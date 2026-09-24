#!/bin/bash
#SBATCH --job-name=cags-classification
#SBATCH --partition=amdgpufast,amdgpulong,amdgpuextralong
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source ~/.bashrc 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH"
uv run python cags_classification.py "$@"
