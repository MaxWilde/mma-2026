#!/bin/bash
#SBATCH --job-name=eval_transcript_retrieval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/eval_transcript_retrieval-%j.out
#SBATCH --error=slurm/eval_transcript_retrieval-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

nvidia-smi

python scripts/debug_transcript_retrieval.py \
  --eval \
  --index-dir artifacts/transcript_index_day1 \
  --top-k 10 \
  --dense-k 100 \
  --lexical-k 100 \
  --rerank-k 50
