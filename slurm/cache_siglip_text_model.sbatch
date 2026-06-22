#!/bin/bash
#SBATCH --job-name=cache-siglip-text
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/cache_siglip_text_model-%j.out
#SBATCH --error=slurm/cache_siglip_text_model-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

MODEL="${SIGLIP_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512}"
OUTPUT="${SIGLIP_TEXT_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512-text}"

echo "Project: $PROJECT_DIR"
echo "Source model: $MODEL"
echo "Text model output: $OUTPUT"

python scripts/cache_siglip_text_model.py \
  --model "$MODEL" \
  --output "$OUTPUT"
