#!/bin/bash
#SBATCH --job-name=siglip-day1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/build_siglip_index_day1-%j.out
#SBATCH --error=slurm/build_siglip_index_day1-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

DATASET_ROOT="${DATASET_ROOT:-/scratch-shared/group_h/data_goncalo/day1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/siglip_index_day1}"
MODEL_NAME="${SIGLIP_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LIMIT="${LIMIT:-0}"

echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Limit: $LIMIT"
echo "Expected scale estimate: 83718 images at 5.76 img/s is about 4.0 hours"
nvidia-smi || true

python scripts/build_siglip_image_index.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --batch-size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --resume
