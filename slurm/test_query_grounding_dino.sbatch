#!/bin/bash
#SBATCH --job-name=query-grounding-dino
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/test_query_grounding_dino-%j.out
#SBATCH --error=slurm/test_query_grounding_dino-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"

source .venv/bin/activate

export GROUNDING_DINO_MODEL="${GROUNDING_DINO_MODEL:-/scratch-shared/group_h/models/grounding-dino-base}"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda}"

QUESTION="${QUESTION:-What color is the refrigerator?}"
INDEX_DIR="${INDEX_DIR:-artifacts/multimodal_index_100_uniform_day1}"
TOP_K="${TOP_K:-5}"
GROUNDING_THRESHOLDS="${GROUNDING_THRESHOLDS:-0.25,0.15,0.10,0.05}"
GROUNDING_PROMPTS="${GROUNDING_PROMPTS:-}"
DEBUG_GROUNDING="${DEBUG_GROUNDING:-1}"

echo "Project: $PROJECT_DIR"
echo "Question: $QUESTION"
echo "Index: $INDEX_DIR"
echo "GroundingDINO model: $GROUNDING_DINO_MODEL"
echo "Grounding thresholds: $GROUNDING_THRESHOLDS"
echo "Grounding prompts: ${GROUNDING_PROMPTS:-auto}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

extra_args=()
if [[ -n "$GROUNDING_PROMPTS" ]]; then
  extra_args+=(--grounding-prompts "$GROUNDING_PROMPTS")
fi
if [[ "$DEBUG_GROUNDING" == "1" || "$DEBUG_GROUNDING" == "true" ]]; then
  extra_args+=(--debug-grounding)
fi

python scripts/query_index.py "$QUESTION" \
  --index-dir "$INDEX_DIR" \
  --top-k "$TOP_K" \
  --rerank \
  --grounding-threshold "$GROUNDING_THRESHOLDS" \
  "${extra_args[@]}"
