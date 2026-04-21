#!/bin/bash

set -euo pipefail

HPC_USERNAME="20231193"
HPC_HOST="hpc.tue.nl"
HPC_PROJECT_DIR="/home/20231193/word2vec"
HPC_LOGS_DIR="/home/20231193/word2vec/logs"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "  Local: $LOCAL_PROJECT_DIR"
echo "  Remote: ${HPC_USERNAME}@${HPC_HOST}:${HPC_PROJECT_DIR}"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='logs/*.out' \
    --exclude='logs/*.err' \
    "$LOCAL_PROJECT_DIR/" "${HPC_USERNAME}@${HPC_HOST}:${HPC_PROJECT_DIR}/"

SLURM_JOB_ID=$(ssh "${HPC_USERNAME}@${HPC_HOST}" "cd ${HPC_PROJECT_DIR} && mkdir -p logs && sbatch --parsable run.sh")

if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "Job ID: $SLURM_JOB_ID"
    echo ""
    echo "ssh ${HPC_HOST} 'squeue -u ${HPC_USERNAME}'"
    echo "ssh ${HPC_HOST} 'tail -n 50 ${HPC_LOGS_DIR}/${SLURM_JOB_ID}.out'"
    echo "ssh ${HPC_HOST} 'tail -n 50 ${HPC_LOGS_DIR}/${SLURM_JOB_ID}.err'"
    echo "ssh ${HPC_HOST} 'scancel $SLURM_JOB_ID'"
    echo "ssh ${HPC_HOST} 'scontrol show job $SLURM_JOB_ID'"
    echo ""
    # echo "ssh ${HPC_HOST} 'tail -n 50 ${HPC_LOGS_DIR}/tortuosity_${SLURM_JOB_ID}.out'" | pbcopy
    echo "ssh ${HPC_HOST} 'scancel $SLURM_JOB_ID'" | pbcopy
    # echo "ssh ${HPC_HOST} 'squeue -u ${HPC_USERNAME}'" | pbcopy
else
    echo "ERROR: Job submission failed!"
    exit 1
fi
