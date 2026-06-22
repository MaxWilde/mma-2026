#!/bin/bash
#SBATCH --job-name=qwen-token-bench
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/benchmark_qwen_max_tokens-%j.out
#SBATCH --error=slurm/benchmark_qwen_max_tokens-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"

source .venv/bin/activate

DATASET_ROOT="${DATASET_ROOT:-day1/Stevan/13}"
LIMIT="${LIMIT:-50}"
TOKENS="${TOKENS:-32,48,64,96}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/qwen_token_benchmark}"
QUERIES="${QUERIES:-What color is the refrigerator?|Where is the kettle?|What is on the stove?|What color is the shirt of the person cooking?}"

export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export QWEN_VL_LOCAL_FILES_ONLY="${QWEN_VL_LOCAL_FILES_ONLY:-1}"

echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Limit: $LIMIT"
echo "Tokens: $TOKENS"
echo "Output dir: $OUTPUT_DIR"
echo "Queries: $QUERIES"
echo "Qwen model: $QWEN_VL_MODEL"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python scripts/benchmark_qwen_max_tokens.py \
  --dataset-root "$DATASET_ROOT" \
  --limit "$LIMIT" \
  --tokens "$TOKENS" \
  --output-dir "$OUTPUT_DIR" \
  --queries "$QUERIES"
