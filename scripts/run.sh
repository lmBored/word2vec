#!/bin/bash
#SBATCH --job-name=word2vec
#SBATCH --partition=...
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

export PYTHONUNBUFFERED=1

module purge
module load uv
uv venv
source .venv/bin/activate
uv sync --active -p .venv

uv run python evaluate.py --model sgns --epochs 100 --dim 300 --neg 10 --lr 0.025 --tokens 17000000
