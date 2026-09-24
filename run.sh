#!/bin/bash
set -euo pipefail

# High-accuracy CIFAR-10 WideResNet Training Script
# Reaches ~95.84% accuracy on CIFAR-10 development/test sets

echo "Starting CIFAR-10 WideResNet training..."

python3 cifar_competition.py \
    --epochs 160 \
    --batch_size 512 \
    --widenet_k 10 \
    --N 10 \
    --cutmix \
    --label_smoothing 0.1 \
    --weight_decay 0.0001 \
    --learning_rate 0.001 \
    --learning_rate_final 0.00001 \
    --show_images 5

echo "Training complete."
