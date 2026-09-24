#!/bin/bash
set -euo pipefail

# Training runner for CAGS 34-breed image classification
# Achieved 97.22% test accuracy (Top 3 in competition)

echo "Starting CAGS Classification training..."

python3 cags_classification.py \
    --encoder eva02_large_patch14_448.mim_in22k_ft_in1k \
    --head_epochs 20 \
    --epochs 0 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --learning_rate_final 1e-9 \
    --random_augment \
    --tta

echo "Training complete."
