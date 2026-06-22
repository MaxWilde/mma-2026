#!/bin/bash
#SBATCH --job-name=siglip-report
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/build_siglip_retrieval_report-%j.out
#SBATCH --error=slurm/build_siglip_retrieval_report-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

INDEX_DIR="${INDEX_DIR:-artifacts/siglip_index_day1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/retrieval_reports/siglip_day1}"
TOP_K="${TOP_K:-10}"
TEXT_MODEL="${SIGLIP_TEXT_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512-text}"

export SIGLIP_DEVICE="${SIGLIP_DEVICE:-cuda}"

echo "Project: $PROJECT_DIR"
echo "Index: $INDEX_DIR"
echo "Output: $OUTPUT_DIR"
echo "Top-k: $TOP_K"
echo "Text model: $TEXT_MODEL"
echo "SigLIP device: $SIGLIP_DEVICE"
nvidia-smi || true

python scripts/build_siglip_retrieval_report.py \
  --index-dir "$INDEX_DIR" \
  --text-model-name "$TEXT_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --query-variants \
  --merge-variants
