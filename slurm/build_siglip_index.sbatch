#!/bin/bash
#SBATCH --job-name=siglip-index
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/build_siglip_index-%j.out
#SBATCH --error=slurm/build_siglip_index-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

DATASET_ROOT="${DATASET_ROOT:-day1/Werner/13}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/siglip_index_werner13}"
MODEL_NAME="${SIGLIP_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Limit: $LIMIT"
echo "Batch size: $BATCH_SIZE"
nvidia-smi || true

python scripts/build_siglip_image_index.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE"
