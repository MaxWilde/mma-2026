#!/bin/bash
#SBATCH --job-name=siglip-query
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/query_siglip_index-%j.out
#SBATCH --error=slurm/query_siglip_index-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

QUESTION="${QUESTION:-What color is the refrigerator?}"
INDEX_DIR="${INDEX_DIR:-artifacts/siglip_index_day1}"
TOP_K="${TOP_K:-5}"
GROUNDING="${GROUNDING:-none}"

export SIGLIP_TEXT_MODEL_NAME="${SIGLIP_TEXT_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512-text}"
export GROUNDING_DINO_MODEL="${GROUNDING_DINO_MODEL:-/scratch-shared/group_h/models/grounding-dino-base}"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda}"

echo "Project: $PROJECT_DIR"
echo "Question: $QUESTION"
echo "Index: $INDEX_DIR"
echo "Grounding: $GROUNDING"
echo "SigLIP text model: $SIGLIP_TEXT_MODEL_NAME"
echo "GroundingDINO model: $GROUNDING_DINO_MODEL"
nvidia-smi || true

python scripts/query_siglip_index.py "$QUESTION" \
  --index-dir "$INDEX_DIR" \
  --top-k "$TOP_K" \
  --grounding "$GROUNDING" \
  --debug-grounding
