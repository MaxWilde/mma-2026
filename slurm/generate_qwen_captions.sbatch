#!/bin/bash
#SBATCH --job-name=qwen-captions
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/generate_qwen_captions-%j.out
#SBATCH --error=slurm/generate_qwen_captions-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch1/shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"

source .venv/bin/activate

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR}"
OUTPUT="${OUTPUT:-artifacts/qwen_keyframe_captions.jsonl}"
LIMIT="${LIMIT:-5}"

export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export QWEN_VL_LOCAL_FILES_ONLY="${QWEN_VL_LOCAL_FILES_ONLY:-1}"
export QWEN_VL_MAX_NEW_TOKENS="${QWEN_VL_MAX_NEW_TOKENS:-96}"

echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Output: $OUTPUT"
echo "Limit: $LIMIT"
echo "Qwen model: $QWEN_VL_MODEL"
echo "Offline HuggingFace loading: $QWEN_VL_LOCAL_FILES_ONLY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python scripts/generate_qwen_keyframe_captions.py \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT" \
  --limit "$LIMIT"
