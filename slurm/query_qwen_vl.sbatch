#!/bin/bash
#SBATCH --job-name=query-qwen-vl
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/query_qwen_vl-%j.out
#SBATCH --error=slurm/query_qwen_vl-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch1/shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"

source .venv/bin/activate

QUESTION="${QUESTION:-What are they cooking?}"
INDEX_DIR="${INDEX_DIR:-artifacts/transcript_index}"
TOP_K="${TOP_K:-3}"

export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export QWEN_VL_LOCAL_FILES_ONLY="${QWEN_VL_LOCAL_FILES_ONLY:-1}"
export QWEN_VL_MAX_NEW_TOKENS="${QWEN_VL_MAX_NEW_TOKENS:-96}"

echo "Project: $PROJECT_DIR"
echo "Question: $QUESTION"
echo "Index: $INDEX_DIR"
echo "Qwen model: $QWEN_VL_MODEL"
echo "Offline HuggingFace loading: $QWEN_VL_LOCAL_FILES_ONLY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python scripts/query_index.py "$QUESTION" \
  --index-dir "$INDEX_DIR" \
  --top-k "$TOP_K" \
  --use-vision \
  --vision-mode qwen
