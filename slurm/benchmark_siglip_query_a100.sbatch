#!/bin/bash
#SBATCH --job-name=siglip-query-bench
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/benchmark_siglip_query_a100-%j.out
#SBATCH --error=slurm/benchmark_siglip_query_a100-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

INDEX_DIR="${INDEX_DIR:-artifacts/siglip_index_day1}"
export SIGLIP_DEVICE="${SIGLIP_DEVICE:-cuda}"
export SIGLIP_TEXT_MODEL_NAME="${SIGLIP_TEXT_MODEL_NAME:-/scratch-shared/group_h/models/siglip2-so400m-patch16-512-text}"

echo "Project: $PROJECT_DIR"
echo "Index: $INDEX_DIR"
echo "SigLIP text model: $SIGLIP_TEXT_MODEL_NAME"
echo "SigLIP device: $SIGLIP_DEVICE"
nvidia-smi || true

python scripts/query_siglip_index.py \
  --index-dir "$INDEX_DIR" \
  --top-k 3 \
  --grounding none \
  --interactive <<'EOF'
Where is the kettle?
What is on the stove?
What color is the refrigerator?
What color is the shirt of the person cooking?
EOF
