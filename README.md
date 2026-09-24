# CAGS 34-Breed Image Classifier (Top 3, 97.22% Accuracy)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Course: NPFL138 Deep Learning](https://img.shields.io/badge/Course-NPFL138%20Deep%20Learning-purple.svg)](https://ufal.mff.cuni.cz/courses/npfl138)
[![Competition](https://img.shields.io/badge/Competition-Top%203%20%2F%20Rank%203-gold.svg)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.22%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning image classification pipeline for the **CAGS (Cats and Dogs)** 34-breed fine-grained classification competition, developed as part of the **NPFL138 (Deep Learning)** course at **Charles University (Faculty of Mathematics and Physics / MFF UK)** (Apr 2026).

🏆 **Achieved 97.22% test accuracy**, placing **Top 3** out of 139 competing teams and individual submissions.

---

## 🎓 Course & Competition Context

- **Course**: [NPFL138 Deep Learning](https://ufal.mff.cuni.cz/courses/npfl138), Charles University (MFF UK), Institute of Formal and Applied Linguistics (ÚFAL)
- **Instructor**: Milan Straka
- **Task**: CAGS Classification ([Task Specification](https://raw.githubusercontent.com/ufal/npfl138/master/tasks/cags_classification.md))
- **Team**: Yannick Daniel Gibson, Robin Klubarski, Vojtěch Nekl
- **Final Result**: **97.22% Test Accuracy** (Rank 3 / Top 3 out of 139 teams, 5/5 Bonus Points)

---

## 🌟 Technical Highlights & Architecture

### 1. Vision Transformer Backbones (EVA-02 & Timm SOTA)
- Leverages state-of-the-art Vision Transformer encoders via `timm` (including **EVA-02 Large Patch14 @ 448px** `eva02_large_patch14_448.mim_in22k_ft_in1k` and `eva02_base_patch14_448`).
- High-resolution visual token processing with dynamically resized bilinear feature interpolation.

### 2. Two-Phase Fine-Tuning Pipeline
- **Phase 1 (Linear Probing / Head Only)**: Trains custom multi-layer classification heads (with GELU / ReLU non-linearities) on frozen transformer embeddings using Cosine Annealing decay down to $\eta_{\text{min}} = 10^{-9}$.
- **Phase 2 (End-to-End Fine-Tuning)**: Unfreezes the full transformer backbone for subtle domain adaptation with linear warmup and cosine decay.

### 3. Multi-Model Ensembling & Test-Time Augmentation (TTA)
- **Multi-Seed & Multi-Encoder Ensembling**: Trains diverse models across unique seeds/encoders and aggregates softmax probabilities.
- **10-Crop Test-Time Augmentation (TTA)**: Evaluates 10 distinct crops per image (center crop + 4 corner crops at $192\times192$ px, with and without horizontal flips) and averages predicted class distributions for maximum generalization.

### 4. Data Regularization & Augmentation
- **RandAugment**: Automatic stochastic augmentation policy (`RandAugment(num_ops=3, magnitude=3)`).
- **Geometric & Color Transforms**: Multi-scale resizing (224-256px), random padding, random cropping ($224\times224$), color jittering (brightness, contrast, saturation, hue), and random rotations ($\pm 15^\circ$).

### 5. Distributed / Multi-GPU Acceleration
- Multi-GPU parallel scaling via `torch.nn.DataParallel` with automatic linear learning rate scaling rule ($LR_{\text{effective}} = LR \times N_{\text{GPUs}}$).

---

## 📁 Repository Structure

```text
cags-classification/
├── cags_classification.py    # Main training, ensembling, and evaluation pipeline
├── train.py                  # CLI entrypoint wrapper
├── run.sh                    # Training runner script
├── crun.sh                   # SLURM multi-GPU batch submission script
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project packaging metadata
├── LICENSE                   # MIT License
└── README.md                 # Project documentation & benchmark results
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YannickGibson/cags-classification.git
cd cags-classification
```

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 💻 Training & Evaluation

### Single Model Training (EVA-02 Large + RandAugment + TTA)

```bash
python3 cags_classification.py \
    --encoder eva02_large_patch14_448.mim_in22k_ft_in1k \
    --head_epochs 20 \
    --epochs 0 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --learning_rate_final 1e-9 \
    --random_augment \
    --tta
```

### Multi-Model Ensemble Training

To train an ensemble of $N$ models across different random seeds:

```bash
python3 cags_classification.py \
    --ensemble 6 \
    --encoder eva02_large_patch14_448.mim_in22k_ft_in1k \
    --head_epochs 20 \
    --epochs 0 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --learning_rate_final 1e-9 \
    --random_augment \
    --tta
```

### Evaluating an Existing Ensemble Directory

```bash
python3 cags_classification.py \
    --eval_ensemble logs/your_ensemble_logdir \
    --encoder eva02_large_patch14_448.mim_in22k_ft_in1k \
    --tta
```

---

## ⚙️ Command-Line Arguments

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--encoder` | `str+` | `tf_efficientnetv2_b0.in1k` | Pretrained `timm` backbone name(s) |
| `--head_epochs` | `int` | `0` | Epochs for Phase 1 (head only, frozen backbone) |
| `--epochs` | `int` | `1` | Epochs for Phase 2 (full model end-to-end) |
| `--batch_size` | `int` | `32` | Batch size for training / evaluation |
| `--learning_rate` | `float` | `0.001` | Initial base learning rate |
| `--learning_rate_final` | `float` | `1e-9` | Minimum learning rate for cosine scheduler |
| `--head_layers` | `int+` | `1000` | Hidden layer dimensions for classification head |
| `--gelu` | `flag` | `False` | Use GELU activation instead of ReLU |
| `--random_augment` | `flag` | `False` | Enable `RandAugment(num_ops=3, magnitude=3)` |
| `--ensemble` | `int` | `None` | Number of models to train for ensembling |
| `--eval_ensemble` | `str` | `None` | Path to logdir to evaluate saved ensemble |
| `--tta` | `flag` | `False` | Enable 10-crop + flip Test-Time Augmentation |
| `--gpus` | `int` | `None` | Number of GPUs to allocate (default: all available) |

---

## 📊 Summary of Experiments

| Experiment / Setup | Backbone | Head Epochs | Augmentation | TTA | Dev Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top 3 Competition Submission** | **EVA-02 Large (Ensemble)** | **20** | **RandAugment** | **Yes** | **97.71%** | **97.22%** |
| Single Model | EVA-02 Large 448 | 20 | RandAugment | No | 98.04% | 96.73% |
| Single Model | EVA-02 Base 448 | 10 | Standard | No | 96.08% | 95.42% |
| Baseline | EfficientNetV2-B0 | 10 | None | No | 91.83% | 91.10% |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
