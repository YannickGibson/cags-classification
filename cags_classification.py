#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import torchmetrics
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2526.5")
npfl138.global_keras_initializers()

import timm
from npfl138.datasets.cags import CAGS
from npfl138.callbacks import SaveBestWeights

# d3dd8c02-680a-4efb-8cdb-5602f21e7e5f
# 4e29a6d4-1b1d-4232-9669-2a40780fa310
# 26611dc1-9d91-438c-8fb9-8a16ac725fe7
# 1. Vojta, 2. Robin, 3. Yannick


# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs for phase 2 (full model).")
parser.add_argument("--head_epochs", default=0, type=int, help="Number of epochs for phase 1 (head only, encoder frozen).")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--warmup_epochs", default=0, type=int, help="Number of warmup epochs.")
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--encoder", default=["tf_efficientnetv2_b0.in1k"], type=str, nargs="+", help="Timm model name(s) for the encoder backbone. One for all models, or one per ensemble model.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--head_lr", default=None, type=float, help="Learning rate for phase 1 head-only training (default: same as --learning_rate).")
parser.add_argument("--learning_rate_final", default=0.000000001, type=float, help="Final learning rate.")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay strength.")
#parser.add_argument("--freeze_encoder", default=False, action="store_true", help="Whether to freeze encoder part of the model.")
parser.add_argument("--gpus", default=None, type=int, help="Number of GPUs to use (default: all available).")
parser.add_argument("--head_layers", default=[1000], type=int, nargs="+", help="Hidden layer sizes for the classification head.")
parser.add_argument("--gelu", default=False, action="store_true", help="Use GELU instead of ReLU.")
parser.add_argument("--random_augment", default=False, action="store_true", help="Use RandAugment(num_ops=3, magnitude=3) instead of manual augmentations.")
parser.add_argument("--ensemble", default=None, type=int, help="Number of models for ensemble. Seeds will be 101..100+N.")
parser.add_argument("--eval_ensemble", default=None, type=str, help="Path to ensemble logdir to evaluate (skip training). Auto-discovers ensemble_model_*.pt files.")
parser.add_argument("--tta", default=False, action="store_true", help="Enable test-time augmentation with predefined transforms.")
parser.add_argument("--gpu_invariant_lr", default=False, action="store_true", help="Do not scale learning rate by number of GPUs.")


"""

uv run python cags_classification.py --epochs 0 --encoder eva02_base_patch14_448.mim_in22k_ft_in1k --head_epochs 10
Using 1/1 GPU
=== Phase 1: training head only for 10 epochs ===
Epoch 1/10 17.4s lr=0.0010 loss=0.4106 accuracy=0.9034 dev:loss=0.3556 dev:accuracy=0.9183       
Epoch 2/10 16.0s lr=0.0009 loss=0.0718 accuracy=0.9762 dev:loss=0.1976 dev:accuracy=0.9444       
Epoch 3/10 16.0s lr=0.0008 loss=0.0413 accuracy=0.9860 dev:loss=0.1800 dev:accuracy=0.9608       
Epoch 4/10 14.2s lr=0.0007 loss=0.0159 accuracy=0.9953 dev:loss=0.2234 dev:accuracy=0.9542       
Epoch 5/10 14.2s lr=0.0005 loss=0.0058 accuracy=0.9986 dev:loss=0.2102 dev:accuracy=0.9575       
Epoch 6/10 14.1s lr=0.0003 loss=0.0025 accuracy=0.9995 dev:loss=0.1838 dev:accuracy=0.9608       
Epoch 7/10 14.2s lr=0.0002 loss=0.0010 accuracy=1.0000 dev:loss=0.1849 dev:accuracy=0.9608       
Epoch 8/10 14.1s lr=9.55e-05 loss=0.0009 accuracy=1.0000 dev:loss=0.1835 dev:accuracy=0.9608     
Epoch 9/10 14.2s lr=2.45e-05 loss=0.0009 accuracy=1.0000 dev:loss=0.1836 dev:accuracy=0.9608     
Epoch 10/10 14.2s lr=1.00e-09 loss=0.0008 accuracy=1.0000 dev:loss=0.1834 dev:accuracy=0.9608 

Overfit to dev?
uv run python cags_classification.py --epochs 0 --encoder eva02_large_patch14_448.mim_in22k_ft_in1k --head_epochs 1
Epoch 1/1 51.3s lr=1.00e-09 loss=0.3492 accuracy=0.9132 dev:loss=0.1571 dev:accuracy=0.9673 

bash-4.4$ uv run python cags_classification.py --epochs 0 --encoder eva02_large_patch14_448.mim_in22k_ft_in1k --head_epochs 20 --random_augment
Epoch 1/20 56.3s lr=0.0010 loss=0.4028 accuracy=0.9080 dev:loss=0.2373 dev:accuracy=0.9444
Epoch 2/20 52.5s lr=0.0010 loss=0.0980 accuracy=0.9725 dev:loss=0.3592 dev:accuracy=0.9510
Epoch 20/20 47.7s lr=1.00e-09 loss=0.0011 accuracy=0.9995 dev:loss=0.1640 dev:accuracy=0.9804

# Embed 6
DEV 96.41
# Embed 6 TTA
============================================================
Ensemble evaluation on dev:
============================================================
  Ensemble 1/6: dev accuracy = 0.9673
  Ensemble 2/6: dev accuracy = 0.9706
  Ensemble 3/6: dev accuracy = 0.9706
  Ensemble 4/6: dev accuracy = 0.9706
  Ensemble 5/6: dev accuracy = 0.9673
  Ensemble 6/6: dev accuracy = 0.9673


"""


class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, preprocessing_fn=None, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._preprocessing_fn = preprocessing_fn
        self._augmentation_fn = augmentation_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = example["image"], example["label"]
        if self._augmentation_fn:
            image = self._augmentation_fn(image)
        if self._preprocessing_fn:
            image = self._preprocessing_fn(image)
        return image, label


class Encoder(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class Head(npfl138.TrainableModule):
    def __init__(self, head_layers, activation=torch.nn.ReLU) -> None:
        super().__init__()
        layers = []
        for neurons in head_layers:
            layers.append(torch.nn.LazyLinear(neurons))
            layers.append(activation())
        layers.append(torch.nn.LazyLinear(CAGS.LABELS))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EnsembleModel(npfl138.TrainableModule):
    def __init__(self, models: list) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        probs = []
        for m in self.models:
            output = m(x)
            if getattr(m, 'tta_transforms', None) is not None and not m.training:
                probs.append(output)  # TTA output is already probs
            else:
                probs.append(torch.nn.functional.softmax(output, dim=-1))
        return torch.stack(probs).mean(dim=0)


class Model(npfl138.TrainableModule):

    TTA_CROP_SIZE = 192

    @staticmethod
    def get_tta_transforms():
        cs = Model.TTA_CROP_SIZE
        flip = v2.RandomHorizontalFlip(p=1.0)
        crops = [
            v2.Compose([v2.CenterCrop(cs)]),                                                         # center crop
            v2.Lambda(lambda img: v2.functional.crop(img, 0, 0, cs, cs)),                            # top-left
            v2.Lambda(lambda img: v2.functional.crop(img, 0, 224 - cs, cs, cs)),                     # top-right
            v2.Lambda(lambda img: v2.functional.crop(img, 224 - cs, 0, cs, cs)),                     # bottom-left
            v2.Lambda(lambda img: v2.functional.crop(img, 224 - cs, 224 - cs, cs, cs)),              # bottom-right
        ]
        transforms = [v2.Lambda(lambda img: img)]  # original (identity)
        for i, crop in enumerate(crops):
            transforms.append(crop)                                        # crop without flip
            transforms.append(v2.Compose([crop, flip]) if not isinstance(crop, v2.Lambda)
                              else v2.Lambda(lambda img, c=crop, f=flip: f(c(img))))  # crop + flip
        return transforms

    def __init__(self, args: argparse.Namespace, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = Encoder(encoder)
        self.head = Head(args.head_layers, activation=torch.nn.GELU if args.gelu else torch.nn.ReLU)

        self.encoder_input_size = encoder.pretrained_cfg["input_size"][1:]

        # Store normalization params as buffers so they are saved in state_dict
        mean = torch.tensor(encoder.pretrained_cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(encoder.pretrained_cfg["std"]).view(1, 3, 1, 1)
        self.register_buffer("_norm_mean", mean)
        self.register_buffer("_norm_std", std)

        self.tta_transforms = None

        if torch.cuda.is_available():
            torch.cuda.init()
            self.cuda()
            # Initialize LazyLinear parameters before DataParallel wrapping
            with torch.no_grad():
                dummy = torch.zeros(2, 3, *self.encoder_input_size, device="cuda")
                self(dummy)
            available = torch.cuda.device_count()
            num_gpus = min(args.gpus, available) if args.gpus else available
            if num_gpus > 1:
                device_ids = list(range(num_gpus))
                self.encoder = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
                self.head = torch.nn.DataParallel(self.head, device_ids=device_ids)

    def forward(self, x):
        if not self.training and self.tta_transforms is not None:
            all_probs = []
            for tfm in self.tta_transforms:
                aug_x = tfm(x)
                aug_x = (aug_x - self._norm_mean) / self._norm_std
                aug_x = torch.nn.functional.interpolate(aug_x, size=self.encoder_input_size, mode='bilinear', align_corners=False)
                logits = self.head(self.encoder(aug_x))
                all_probs.append(torch.nn.functional.softmax(logits, dim=-1))
            return torch.stack(all_probs).mean(dim=0)

        x = (x - self._norm_mean) / self._norm_std

        x = torch.nn.functional.interpolate(x, size=self.encoder_input_size, mode='bilinear', align_corners=False)
        
        x = self.encoder(x)  # [N, C] (timm with num_classes=0 does global pool already)
        
        return self.head(x)


def main(args: argparse.Namespace) -> None:
    defaults = parser.parse_args([])
    print("\n".join(f"  {k}: {v}" for k, v in vars(args).items() if getattr(defaults, k) != v))
    print()

    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    print(f"Using {num_gpus}/{available} GPU")

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    cags = CAGS(decode_on_demand=False)

    encoder_model = timm.create_model(args.encoder[0], pretrained=True, num_classes=0)

    model = Model(args=args, encoder=encoder_model)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])

    if args.random_augment:
        augmentation_fn = v2.RandAugment(num_ops=3, magnitude=3)
    elif args.augment:
        augmentation_fn = v2.Compose([
            v2.RandomResize(min_size=224, max_size=256),
            v2.Pad(padding=4),
            v2.RandomCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomRotation(degrees=15),
        ])
    else:
        augmentation_fn = None

    train = TransformedDataset(cags.train, preprocessing_fn=preprocessing, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cags.dev, preprocessing_fn=preprocessing)
    test = TransformedDataset(cags.test, preprocessing_fn=preprocessing)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev, test = map(lambda x: torch.utils.data.DataLoader(x, batch_size=args.batch_size), (dev, test))

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    lr0 = args.learning_rate
    if num_gpus and not args.gpu_invariant_lr:  # each gpu gets same batch -> we effectively have batch size of N x bs -> better gradient -> more lr
        lr0 *= num_gpus

    metric_name = "accuracy"
    metrics = {metric_name: torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)}
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Phase 1: Train head only (encoder frozen) ──
    if args.head_epochs > 0:
        print(f"=== Phase 1: training head only for {args.head_epochs} epochs ===")

        # Freeze the params for backbone
        for param in model.encoder.parameters():
            param.requires_grad = False

        head_steps = len(train) * args.head_epochs
        head_lr = (args.head_lr if args.head_lr is not None else args.learning_rate)
        if num_gpus and not args.gpu_invariant_lr:
            head_lr *= num_gpus
        head_optimizer = torch.optim.Adam(model.head.parameters(), weight_decay=args.weight_decay, lr=head_lr)
        head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer, eta_min=args.learning_rate_final, T_max=head_steps
        )

        model.configure(
            scheduler=head_scheduler,
            optimizer=head_optimizer,
            metrics=metrics,
            logdir=logdir,
            loss=loss_fn,
        )

        model.fit(
            train,
            dev=dev,
            epochs=args.head_epochs,
            callbacks=[SaveBestWeights(path="{logdir}/best_acc.pt", metric=f"dev:{metric_name}", mode="max", baseline=0.8)],
        )

        for param in model.encoder.parameters():
            param.requires_grad = True

    # ── Phase 2: Train full model ──
    if args.epochs > 0:
        # if args.freeze_encoder:
        #     for param in model.encoder.parameters():
        #         param.requires_grad = False

        print(f"=== Phase 2: training full model for {args.epochs} epochs ===")
        warmup_steps = len(train) * args.warmup_epochs
        cosine_steps = len(train) * args.epochs - warmup_steps

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay, lr=lr0)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=args.learning_rate_final, T_max=cosine_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        model.configure(
            scheduler=scheduler,
            optimizer=optimizer,
            metrics=metrics,
            logdir=logdir,
            loss=loss_fn,
        )

        model.fit(
            train,
            dev=dev,
            epochs=args.epochs,
            callbacks=[SaveBestWeights(path="{logdir}/best_acc.pt", metric=f"dev:{metric_name}", mode="max")],
        )

    # Enable TTA for evaluation/prediction
    if args.tta:
        model.tta_transforms = Model.get_tta_transforms()

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


def _train_single_model(args, cags, logdir, seed, encoder_name=None):
    """Train a single model with the given seed and save its best weights. Returns the save path."""
    npfl138.startup(seed, args.threads)

    if encoder_name is None:
        encoder_name = args.encoder[0]

    encoder_model = timm.create_model(encoder_name, pretrained=True, num_classes=0)
    model = Model(args=args, encoder=encoder_model)

    # Create preprocessing (normalization is inside Model.forward)
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])

    if args.random_augment:
        augmentation_fn = v2.RandAugment(num_ops=3, magnitude=3)
    elif args.augment:
        augmentation_fn = v2.Compose([
            v2.RandomResize(min_size=224, max_size=256),
            v2.Pad(padding=4),
            v2.RandomCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomRotation(degrees=15),
        ])
    else:
        augmentation_fn = None

    train = TransformedDataset(cags.train, preprocessing_fn=preprocessing, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cags.dev, preprocessing_fn=preprocessing)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    lr0 = args.learning_rate
    if num_gpus and not args.gpu_invariant_lr:
        lr0 *= num_gpus

    metric_name = "accuracy"
    metrics = {metric_name: torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)}
    loss_fn = torch.nn.CrossEntropyLoss()

    model_logdir = os.path.join(logdir, f"model_{seed}")

    # ── Phase 1: Train head only (encoder frozen) ──
    if args.head_epochs > 0:
        print(f"=== Phase 1: training head only for {args.head_epochs} epochs ===")
        for param in model.encoder.parameters():
            param.requires_grad = False

        head_steps = len(train) * args.head_epochs
        head_lr = (args.head_lr if args.head_lr is not None else args.learning_rate)
        if num_gpus and not args.gpu_invariant_lr:
            head_lr *= num_gpus
        head_optimizer = torch.optim.Adam(model.head.parameters(), weight_decay=args.weight_decay, lr=head_lr)
        head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer, eta_min=args.learning_rate_final, T_max=head_steps
        )

        model.configure(
            scheduler=head_scheduler,
            optimizer=head_optimizer,
            metrics=metrics,
            logdir=model_logdir,
            loss=loss_fn,
        )

        model.fit(
            train,
            dev=dev,
            epochs=args.head_epochs,
            callbacks=[SaveBestWeights(path="{logdir}/best_acc.pt", metric=f"dev:{metric_name}", mode="max")],
        )

        for param in model.encoder.parameters():
            param.requires_grad = True

    # ── Phase 2: Train full model ──
    if args.epochs > 0:
        print(f"=== Phase 2: training full model for {args.epochs} epochs ===")
        warmup_steps = len(train) * args.warmup_epochs
        cosine_steps = len(train) * args.epochs - warmup_steps

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay, lr=lr0)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=args.learning_rate_final, T_max=cosine_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        model.configure(
            scheduler=scheduler,
            optimizer=optimizer,
            metrics=metrics,
            logdir=model_logdir,
            loss=loss_fn,
        )

        model.fit(
            train,
            dev=dev,
            epochs=args.epochs,
            callbacks=[SaveBestWeights(path="{logdir}/best_acc.pt", metric=f"dev:{metric_name}", mode="max")],
        )

    # Save final best weights to a known path for ensemble loading
    save_path = os.path.join(logdir, f"ensemble_model_{seed}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model (seed={seed}) to {save_path}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return save_path


def _load_model(args, save_path, encoder_name=None):
    """Recreate a Model and load saved state dict from disk."""
    if encoder_name is None:
        encoder_name = args.encoder[0]
    encoder_model = timm.create_model(encoder_name, pretrained=True, num_classes=0)
    # Force single-GPU to avoid DataParallel wrapping; weights are saved without 'module.' prefix
    load_args = argparse.Namespace(**vars(args))
    load_args.gpus = 1
    model = Model(args=load_args, encoder=encoder_model)
    state_dict = torch.load(save_path, weights_only=True)
    # Strip DataParallel 'module.' prefix if present but model doesn't use DataParallel
    state_dict = {k.replace("encoder.module.", "encoder.").replace("head.module.", "head."): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    if getattr(args, 'tta', False):
        model.tta_transforms = Model.get_tta_transforms()
    return model


def main_ensemble(args: argparse.Namespace) -> None:
    defaults = parser.parse_args([])
    print("\n".join(f"  {k}: {v}" for k, v in vars(args).items() if getattr(defaults, k) != v))
    print()

    N = args.ensemble

    npfl138.startup(args.seed, args.threads)

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    print(f"Using {num_gpus}/{available} GPU")

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    cags = CAGS(decode_on_demand=False)

    # Resolve per-model encoder names
    encoders = args.encoder if len(args.encoder) == N else args.encoder * N

    # ── Train N models ──
    save_paths = []
    for i in range(1, N + 1):
        seed = 100 + i
        enc = encoders[i - 1]
        print(f"\n{'='*60}")
        print(f"Training model {i}/{N} (seed={seed}, encoder={enc})")
        print(f"{'='*60}")
        path = _train_single_model(args, cags, logdir, seed, encoder_name=enc)
        save_paths.append(path)

    # ── Build dev/test loaders for ensemble evaluation (normalization is inside each Model) ──
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])
    dev_ds = TransformedDataset(cags.dev, preprocessing_fn=preprocessing)
    test_ds = TransformedDataset(cags.test, preprocessing_fn=preprocessing)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    # ── Evaluate ensembles of 1..N on dev ──
    metric_name = "accuracy"
    loss_fn = torch.nn.CrossEntropyLoss()

    for k in range(1, N + 1):
        print(f"\n--- Ensemble eval: {k}/{N} models ---")
        models = [_load_model(args, save_paths[j], encoder_name=encoders[j]) for j in range(k)]
        ensemble = EnsembleModel(models)
        ensemble.configure(
            optimizer=torch.optim.Adam(ensemble.parameters(), lr=1e-4),
            loss=loss_fn,
            metrics={metric_name: torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)},
        )
        ensemble.evaluate(dev_loader, dataset_name=f"ensemble_{k}_dev")
        del ensemble, models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Generate test predictions with full N-model ensemble ──
    print(f"\n--- Generating test predictions with {N}-model ensemble ---")
    models = [_load_model(args, save_paths[j], encoder_name=encoders[j]) for j in range(N)]
    ensemble = EnsembleModel(models)
    ensemble.configure(
        optimizer=torch.optim.Adam(ensemble.parameters(), lr=1e-4),
        loss=loss_fn,
        metrics={metric_name: torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)},
    )

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in ensemble.predict(test_loader, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)

    print(f"Test predictions written to {os.path.join(logdir, 'cags_classification.txt')}")


def eval_ensemble(args: argparse.Namespace) -> None:
    import re

    defaults = parser.parse_args([])
    print("\n".join(f"  {k}: {v}" for k, v in vars(args).items() if getattr(defaults, k) != v))
    print()

    logdir = args.eval_ensemble

    # Discover ensemble model files
    import re
    pt_files = sorted(
        [os.path.join(logdir, f) for f in os.listdir(logdir) if re.match(r"ensemble_model_\d+\.pt$", f)],
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    N = len(pt_files)
    if N == 0:
        raise FileNotFoundError(f"No ensemble_model_*.pt files found in {logdir}")
    print(f"Found {N} models: {[os.path.basename(p) for p in pt_files]}")

    # Resolve encoder names
    if len(args.encoder) == 1:
        encoders = args.encoder * N
    elif len(args.encoder) == N:
        encoders = args.encoder
    else:
        raise ValueError(f"--encoder has {len(args.encoder)} values but found {N} models.")

    npfl138.startup(args.seed, args.threads)

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    print(f"Using {num_gpus}/{available} GPU")

    cags = CAGS(decode_on_demand=False)

    preprocessing = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    dev_ds = TransformedDataset(cags.dev, preprocessing_fn=preprocessing)
    test_ds = TransformedDataset(cags.test, preprocessing_fn=preprocessing)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    # Collect dev labels
    dev_labels = torch.tensor([ex["label"] for ex in cags.dev])

    # Collect probs one model at a time
    all_dev_probs = []
    all_test_probs = []

    for i, (pt_path, enc) in enumerate(zip(pt_files, encoders)):
        seed = int(os.path.basename(pt_path).split("_")[-1].split(".")[0])
        print(f"\n--- Evaluating model {i+1}/{N} (seed={seed}, encoder={enc}) ---")

        model = _load_model(args, pt_path, encoder_name=enc)

        # Collect dev probs
        dev_probs = []
        with torch.no_grad():
            for batch in dev_loader:
                x = batch[0]
                if torch.cuda.is_available():
                    x = x.cuda()
                output = model(x)
                probs = output if model.tta_transforms is not None else torch.nn.functional.softmax(output, dim=-1)
                dev_probs.append(probs.cpu())
        dev_probs = torch.cat(dev_probs, dim=0)
        all_dev_probs.append(dev_probs)

        preds = dev_probs.argmax(dim=-1)
        acc = (preds == dev_labels).float().mean().item()
        print(f"  Single model dev accuracy: {acc:.4f}")

        # Collect test probs
        test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0]
                if torch.cuda.is_available():
                    x = x.cuda()
                output = model(x)
                probs = output if model.tta_transforms is not None else torch.nn.functional.softmax(output, dim=-1)
                test_probs.append(probs.cpu())
        test_probs = torch.cat(test_probs, dim=0)
        all_test_probs.append(test_probs)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cumulative ensemble evaluation on dev
    print(f"\n{'='*60}")
    print("Ensemble evaluation on dev:")
    print(f"{'='*60}")
    dev_probs_stack = torch.stack(all_dev_probs)  # [N, 306, 34]
    for k in range(1, N + 1):
        avg_probs = dev_probs_stack[:k].mean(dim=0)
        preds = avg_probs.argmax(dim=-1)
        acc = (preds == dev_labels).float().mean().item()
        print(f"  Ensemble {k}/{N}: dev accuracy = {acc:.4f}")

    # Final test predictions with full ensemble
    test_probs_stack = torch.stack(all_test_probs)  # [N, 612, 34]
    avg_test_probs = test_probs_stack.mean(dim=0)
    test_preds = avg_test_probs.argmax(dim=-1)

    pred_path = os.path.join(logdir, "cags_classification.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        for pred in test_preds:
            print(pred.item(), file=f)
    print(f"\nTest predictions written to {pred_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Validate --augment and --random_augment are not both set
    if main_args.augment and main_args.random_augment:
        parser.error("--augment and --random_augment cannot be used together.")

    # Validate --encoder count
    if len(main_args.encoder) > 1:
        if main_args.ensemble is None and main_args.eval_ensemble is None:
            parser.error("Multiple --encoder values require --ensemble or --eval_ensemble.")
        if main_args.ensemble is not None and len(main_args.encoder) != main_args.ensemble:
            parser.error(f"--encoder has {len(main_args.encoder)} values but --ensemble is {main_args.ensemble}.")

    if main_args.eval_ensemble is not None:
        eval_ensemble(main_args)
    elif main_args.ensemble is not None:
        main_ensemble(main_args)
    else:
        main(main_args)
