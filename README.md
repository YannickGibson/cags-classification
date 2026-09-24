# CIFAR-10 Wide Residual Network (WideResNet) Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/CIFAR--10%20Accuracy-95.84%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-performance deep learning pipeline for **CIFAR-10** image classification using **Wide Residual Networks (WideResNet)**, enhanced with **DropBlock regularization**, **CutMix & MixUp** data augmentations, **Linear Warmup + Cosine Annealing** learning rate scheduling, and **Multi-GPU `DataParallel`** acceleration.

This model achieves **~95.84% validation/test accuracy** on CIFAR-10 without pre-training on external datasets.

---

## 🌟 Key Highlights & Techniques

### 1. Wide Residual Architecture (Wide-ResNet)
- **Configurable Depth & Width**: Parameterized by block repetitions ($N$) and widening multiplier ($k$). Network depth scales as $6N + 9$ layers (e.g. $N=10 \to 69$ convolutional layers).
- **Grouped Convolutions (ResNeXt-style)**: Support for multi-branch channel groupings (`--groups`) for parameter-efficient representation learning.
- **Residual Building Blocks**: Pre-activation style residual bottlenecks with Batch Normalization and ReLU activations.

### 2. Advanced Regularization & Augmentation
- **DropBlock Regularization**: Convolutional spatial dropout applied across multi-scale feature maps ($7\times7$, $3\times3$, $1\times1$) to suppress spatially co-dependent activations.
- **CutMix & MixUp Data Augmentation**: Stochastic patch-cutting and convex linear interpolations between image pairs with soft-label ground truth synthesis.
- **Label Smoothing**: Softens target class distributions ($\alpha = 0.1$) to prevent over-confident logit predictions.
- **Spatial Transforms**: Random multi-scale resizing (28px–36px), padding (4px), random cropping ($32\times32$), and random horizontal flips.

### 3. Optimization & Distributed Scaling
- **Warmup + Cosine Annealing Schedule**: 5-epoch linear warmup followed by Cosine Annealing learning rate decay down to $10^{-5}$.
- **Gradient Clipping**: Norm clipping at $\|\mathbf{g}\|_2 \le 1.0$ for numerical stability.
- **Multi-GPU DataParallel & Linear Scaling**: Automatic detection and load-balancing across multi-GPU setups, with linear learning rate scaling ($LR_{\text{scaled}} = LR \times N_{\text{GPUs}}$).
- **Metric Tracking**: Custom `SoftLabelAccuracy` metric for accurate evaluation with soft target distributions.

---

## 📊 Benchmark & Performance

| Model Architecture | Widening Factor ($k$) | Blocks ($N$) | Regularization & Augmentations | Epochs | Dev Accuracy |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **WideResNet-69** | **10** | **10** | **CutMix + MixUp, Label Smoothing (0.1)** | **160** | **95.84%** |
| WideResNet-69 | 10 | 10 | CutMix, DropBlock (0.1), Groups (16) | 160 | 91.20% |
| WideResNet-69 | 4 | 10 | CutMix, Label Smoothing (0.1), Groups (16) | 160 | 89.10% |

---

## 📁 Repository Structure

```text
cifar10-wideresnet/
├── cifar_competition.py    # Main training, evaluation, and inference script
├── train.py                # Wrapper entrypoint
├── run.sh                  # Execution shell script with optimal hyperparameters
├── requirements.txt        # Pinned Python package dependencies
├── pyproject.toml          # Project configuration for pip/uv
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YannickGibson/cifar10-wideresnet.git
cd cifar10-wideresnet
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

## 💻 Training

### Best Configuration (95.84% Accuracy)

To reproduce the optimal WideResNet configuration ($k=10, N=10$):

```bash
python3 cifar_competition.py \
    --epochs 160 \
    --batch_size 512 \
    --widenet_k 10 \
    --N 10 \
    --cutmix \
    --label_smoothing 0.1 \
    --weight_decay 0.0001 \
    --learning_rate 0.001 \
    --learning_rate_final 0.00001
```

Or execute the provided bash script:
```bash
./run.sh
```

### Fast Baseline Training (Single GPU / CPU)

For rapid experimentation or testing on standard hardware:

```bash
python3 cifar_competition.py \
    --epochs 30 \
    --batch_size 128 \
    --widenet_k 2 \
    --N 2 \
    --learning_rate 0.001
```

---

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `1` | Total number of training epochs |
| `--batch_size` | `int` | `128` | Training mini-batch size |
| `--learning_rate` | `float` | `0.001` | Initial base learning rate (scaled with GPUs) |
| `--learning_rate_final` | `float` | `1e-5` | Minimum learning rate for cosine scheduler |
| `--widenet_k` | `int` | `2` | Widening factor multiplier for channel depth |
| `--N` | `int` | `2` | Number of residual blocks per stage ($6N+9$ total depth) |
| `--cutmix` | `flag` | `False` | Enables stochastic CutMix & MixUp augmentation |
| `--label_smoothing` | `float` | `0.2` | Label smoothing factor for cross-entropy loss |
| `--weight_decay` | `float` | `1e-4` | $L_2$ weight decay penalty |
| `--dropblock` | `float` | `0.0` | DropBlock dropout probability |
| `--groups` | `int` | `1` | Number of grouped convolution channels |
| `--gpus` | `int` | `None` | Number of GPUs to allocate (default: all available) |
| `--seed` | `int` | `42` | Random seed for reproducibility |

---

## 📈 Logging & Visualization

Training logs and model checkpoints are automatically saved to `logs/`:

- **Best Checkpoint**: Stored as `{logdir}/best_accuracy.pt` using dev accuracy tracking.
- **TensorBoard**: Real-time loss curves, learning rates, and image augmentations can be monitored:
  ```bash
  tensorboard --logdir logs
  ```

---

## 📚 References

1. **Wide Residual Networks** — Sergey Zagoruyko, Nikos Komodakis ([arXiv:1605.07146](https://arxiv.org/abs/1605.07146))
2. **DropBlock: A regularization method for convolutional networks** — Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le ([arXiv:1810.12890](https://arxiv.org/abs/1810.12890))
3. **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** — Sangdoo Yun et al. ([arXiv:1905.04899](https://arxiv.org/abs/1905.04899))
4. **mixup: Beyond Empirical Risk Minimization** — Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz ([arXiv:1710.09412](https://arxiv.org/abs/1710.09412))

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
