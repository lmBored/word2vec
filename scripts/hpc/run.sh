#!/bin/bash
#SBATCH --job-name=word2vec
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=mhjjj2113@gmail.com
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

# Disable output buffering for real-time logs
export PYTHONUNBUFFERED=1

module purge
module load uv
uv venv
source .venv/bin/activate
uv sync --active -p .venv

uv run python evaluate.py --model sgns --epochs 100 --dim 300 --neg 10 --lr 0.025 --tokens 17000000
uv run python evaluate.py --model cbow --epochs 100 --dim 300 --neg 10 --lr 0.025 --tokens 17000000
