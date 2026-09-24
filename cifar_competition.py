#!/usr/bin/env python3
import argparse
import os
import warnings

import numpy as np
import torch
import torchmetrics
from torchvision.transforms import v2
import torchvision
from torch.utils.data import default_collate

import npfl138
npfl138.require_version("2526.4")
from npfl138.callbacks import SaveBestWeights
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
## Additional
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=0.00001, type=float, help="Final learning rate.")
parser.add_argument("--show_images", default=None, const=10, type=int, nargs="?", help="Show augmented images.")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay strength.")
parser.add_argument("--cutmix", default=False, action="store_true", help="Whether to use cutmix evaluation.")
parser.add_argument("--widenet_k", default=2, type=int, help="Widening factor (widenet inspired)")
parser.add_argument("--N", default=2, type=int, help="Amount of repetitions of resnet block in one resnet part (total three) (is called N in the WideResNet paper). Total depth (trainable layers without BN) can be calculated as 6*N+9, (e.g. N=10 -> 69 depth, 28 N -> Depth = 168 + 9 = 177 layers).")
parser.add_argument("--groups", default=1, type=int, help="Number of groups for convolutions")
parser.add_argument("--dropblock", default=0, type=float, help="Dropblock probability to cut out feature maps.")
parser.add_argument("--gpus", default=None, type=int, help="Number of GPUs to use (default: all available).")

"""
Very good setting:
python cifar_competition.py --epochs 160 --batch_size 512 --widenet_k 10 --cutmix --show_images --N 10 --label_smoothing 0.1
... Epoch 150/160 236.5s lr=1.95e-05 loss=1.0324 accuracy=0.8988 dev:loss=0.6454 dev:accuracy=0.9584

Meh
uv run python cifar_competition.py --epochs 160 --batch_size 512 --widenet_k 4 --cutmix --show_images --N 10 --label_smoothing 0.1 --groups 16
Epoch 159/160 53.2s lr=1.01e-05 loss=1.2534 accuracy=0.8098 dev:loss=0.8236 dev:accuracy=0.8906
Epoch 200/200 53.5s lr=1.00e-05 loss=1.3099 accuracy=0.7831 dev:loss=0.8264 dev:accuracy=0.8910 

python cifar_competition.py --epochs 160 --batch_size 512 --widenet_k 10 --cutmix --show_images --N 10 --label_smoothing 0.1 --groups 16 --dropblock 0.1
"""


class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None, cutmix=False) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image, label, = example["image"], example["label"]
        image = image.to(torch.float32) / 255
        if self._augmentation_fn:
            image = self._augmentation_fn(image)
        return image, label
    
    @staticmethod
    def get_collate_fn(use_cutmix: bool):

        cutmix = v2.CutMix(num_classes=CIFAR10.LABELS)
        mixup = v2.MixUp(num_classes=CIFAR10.LABELS)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        def collate_fn(x):

            images, labels = default_collate(x)
            if use_cutmix:
                return cutmix_or_mixup(images, labels.to(torch.int64))
            else:
                return images, labels
        
        return collate_fn

class _ResBlock(torch.nn.Module):
    def __init__(self, *modules: torch.nn.Module, skip_connection: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList(modules)
        self.skip_connection = skip_connection
    
    def append(self, x: torch.nn.Module):
        self.layers.append(x)

    def forward(self, x):
        # Remember input
        inp = x

        # Inner block
        for layer in self.layers:
            x = layer(x)
        
        # Add skip connection with output
        if self.skip_connection:
            res = inp + x
        else:
            res = x
        return res
    
class ResBlock(torch.nn.Module):
    def __init__(self, kernel_size: int, channels: int, skip_connection: bool = True, groups: int = 1):
        super().__init__()

        self.cnn1 = torch.nn.LazyConv2d(kernel_size=kernel_size, out_channels=channels, padding="same", groups=groups)
        self.relu = torch.nn.ReLU()
        self.cnn2 = torch.nn.Conv2d(kernel_size=kernel_size, in_channels=channels, out_channels=channels, padding="same", groups=groups)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=channels)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=channels)

        self.net = _ResBlock(
            self.cnn1,
            self.batch_norm1,
            self.relu,
            self.cnn2,
            self.batch_norm2,
            skip_connection=skip_connection
        )
    def forward(self, x):
        return self.net(x)



class Backbone(npfl138.TrainableModule):
    def __init__(self, in_channels, widenet_k, N, dropblock, groups):
        super().__init__()

        self.res_blocks_16 = torch.nn.Sequential(
            ResBlock(3, channels=16 * widenet_k, skip_connection=False, groups=groups),
            torchvision.ops.DropBlock2d(p=dropblock, block_size=7),
            *
                (
                    torch.nn.Sequential(
                        ResBlock(3, channels=16 * widenet_k, groups=groups),
                        torchvision.ops.DropBlock2d(p=dropblock, block_size=7)
                    )
                    for _ in range(N)
                )
            
        )
        self.res_blocks_32 = torch.nn.Sequential(
            ResBlock(3, channels=32 * widenet_k, skip_connection=False, groups=groups),
            torchvision.ops.DropBlock2d(p=dropblock, block_size=3),
            *
                (
                    torch.nn.Sequential(
                        ResBlock(3, channels=32 * widenet_k, groups=groups),
                        torchvision.ops.DropBlock2d(p=dropblock, block_size=3)
                    )
                    for _ in range(N)
                )
        )

        self.res_blocks_64 = torch.nn.Sequential(
            ResBlock(3, channels=64 * widenet_k, skip_connection=False, groups=groups),
            torchvision.ops.DropBlock2d(p=dropblock, block_size=1),
            *
                (
                    torch.nn.Sequential(
                        ResBlock(3, channels=64 * widenet_k, groups=groups),
                        torchvision.ops.DropBlock2d(p=dropblock, block_size=1)
                    )
                    for _ in range(N)
                )
        )

        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            self.res_blocks_16,
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),

            self.res_blocks_32,
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),

            self.res_blocks_64,
            torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
            torch.nn.Flatten(),

        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            #print(x.shape)
            x = layer(x)
        return x

class Direct(npfl138.TrainableModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.LazyLinear(1000)
        self.output = torch.nn.Linear(in_features=1000, out_features=CIFAR10.LABELS)
        self.relu = torch.nn.ReLU()
        #self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, fmap1: torch.Tensor)-> torch.Tensor:
        x = self.linear(fmap1)
        x = self.relu(x)
        x = self.output(x)
        return x


class SoftLabelAccuracy(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__()
        self._acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        # If CutMix soft labels: [B, C] → argmax to hard labels
        if targets.ndim > 1:
            targets = targets.argmax(dim=1)
        self._acc.update(preds, targets)

    def compute(self):
        return self._acc.compute()
    
    def reset(self):
        self._acc.reset()


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.backbone = Backbone(in_channels=CIFAR10.C, widenet_k=args.widenet_k, N=args.N, dropblock=args.dropblock, groups=args.groups)
        self.direct = Direct()

        if torch.cuda.is_available():
            torch.cuda.init()  # initialize cuda context first -> covers cuBLAS warning
            self.cuda()
            dummy = torch.zeros(2, CIFAR10.C, 32, 32, device="cuda")
            self(dummy)

            # Determine GPU count
            available = torch.cuda.device_count()
            num_gpus = min(args.gpus, available) if args.gpus else available
            print(f"Using {num_gpus}/{available} GPU")
            # Now safe to wrap with DataParallel
            if num_gpus > 1:
                device_ids = list(range(num_gpus))
                self.backbone = torch.nn.DataParallel(self.backbone, device_ids=device_ids)
                self.direct = torch.nn.DataParallel(self.direct, device_ids=device_ids)
        else:
            print("No GPUs available, using CPU.")

    def forward(
        self, first: torch.Tensor
    ) -> tuple[torch.Tensor]:
        fmap1 = self.backbone(first)
        direct_comparison = self.direct(fmap1)
        #if labels.ndim > 1:
        #    labels = labels.argmax(dim=1)
        return direct_comparison
    
    def train_step(self, xs, y):
        y_pred = self(*xs)
        loss = self.track_loss(self.compute_loss(y_pred, y, *xs))
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # GRADIENT CLIPPING
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler is not None and self.scheduler.step()
            metrics = self.compute_metrics(y_pred, y, *xs)
            return {**({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {}), **self.losses, **metrics}


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    augmentation_fn = v2.Compose([
        v2.RandomResize(min_size=28, max_size=36),
        v2.Pad(padding=4),
        v2.RandomCrop(size=(32, 32)),
        v2.RandomHorizontalFlip(),
    ])

    
    # Load the data.
    cifar = CIFAR10()
    # Train: ([45000, 3, 32, 32])

    train = TransformedDataset(cifar.train, augmentation_fn, cutmix=args.cutmix)
    dev = TransformedDataset(cifar.dev)
    test = TransformedDataset(cifar.test)

    model = Model(args)


    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        collate_fn=TransformedDataset.get_collate_fn(args.cutmix)
    )
    dev, test = map(lambda x: torch.utils.data.DataLoader(x, batch_size=args.batch_size), (dev, test))

    # Log augmented train dataset
    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus else available
    gpu_scaled_lr = args.learning_rate * num_gpus  # linear scaling rule
    warmup_steps = len(train) * 5  # warmup over first 5 epochs
    cosine_steps = len(train) * args.epochs - warmup_steps

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=gpu_scaled_lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=args.learning_rate_final, T_max=cosine_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #        optimizer, eta_min=args.learning_rate_final, T_max=len(train) * args.epochs
    #)
    model.configure(
        scheduler = scheduler,
        optimizer=optimizer,
        metrics={
            "accuracy": SoftLabelAccuracy(num_classes=CIFAR10.LABELS),
        },
        logdir=logdir,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),  # solves one hot encoding conversion implicitly
    )

    print(f"Model device: {model.device}")


    if args.show_images:
        GRID, REPEATS, TAG = args.show_images, 5, "augmented"
        batch_images = next(iter(train))[0]
        # 128, 3, 32, 32

        for step in range(REPEATS):
            grid = torchvision.utils.make_grid([batch_images[step * REPEATS + i] for i in range(GRID * GRID)], nrow=GRID)
            model.logger.log_image(TAG, grid, step, data_format="CHW")
        print(f"Saved first {GRID * GRID} training imaged to logs/{TAG}")

    logs = model.fit(
        train,
        dev=dev, 
        epochs=args.epochs,
        callbacks=[SaveBestWeights(path="{logdir}/best_accuracy.pt", metric="dev:accuracy", mode="max")]
    )

    
    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
