#!/bin/bash
#SBATCH --job-name=cache-siglip
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/cache_siglip_model-%j.out
#SBATCH --error=slurm/cache_siglip_model-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

MODEL_NAME="${SIGLIP_MODEL_NAME:-google/siglip-base-patch16-224}"
OUTPUT_DIR="${SIGLIP_MODEL_DIR:-/scratch-shared/group_h/models/siglip-base-patch16-224}"

echo "Project: $PROJECT_DIR"
echo "Model: $MODEL_NAME"
echo "Output dir: $OUTPUT_DIR"

python - <<'PY'
import os
from transformers import AutoModel, AutoProcessor

model_name = os.environ.get("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224")
output_dir = os.environ.get("SIGLIP_MODEL_DIR", "/scratch-shared/group_h/models/siglip-base-patch16-224")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
processor.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print(f"Cached {model_name} in {output_dir}")
PY
