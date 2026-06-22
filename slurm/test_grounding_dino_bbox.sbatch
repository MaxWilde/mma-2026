#!/bin/bash
#SBATCH --job-name=grounding-dino-bbox
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/test_grounding_dino_bbox-%j.out
#SBATCH --error=slurm/test_grounding_dino_bbox-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"

source .venv/bin/activate

MODEL="${GROUNDING_DINO_MODEL:-/scratch-shared/group_h/models/grounding-dino-base}"
IMAGE="${IMAGE:-day1/Stevan/13/keyframes/shot_0673_frame_116700.jpg}"
TEXT="${TEXT:-person cooking}"
OUTPUT="${OUTPUT:-artifacts/visual_grounding/dino_test_bbox.jpg}"
THRESHOLD="${GROUNDING_DINO_THRESHOLD:-0.25}"

export GROUNDING_DINO_MODEL="$MODEL"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda}"

echo "Project: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Image: $IMAGE"
echo "Text: $TEXT"
echo "Output: $OUTPUT"
echo "Threshold: $THRESHOLD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python scripts/test_grounding_dino_bbox.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --text "$TEXT" \
  --output "$OUTPUT" \
  --threshold "$THRESHOLD"
