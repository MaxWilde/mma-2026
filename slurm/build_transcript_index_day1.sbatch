#!/bin/bash
#SBATCH --job-name=transcript-index
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/build_transcript_index_day1-%j.out
#SBATCH --error=slurm/build_transcript_index_day1-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_DIR="${PROJECT_DIR:-/scratch-shared/group_h/data_goncalo}"
cd "$PROJECT_DIR"
source .venv/bin/activate

TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-all_transcripts}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/transcript_index_day1}"
DAY="${DAY:-day1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

echo "Project: $PROJECT_DIR"
echo "Transcripts: $TRANSCRIPTS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Day: $DAY"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
nvidia-smi || true

python scripts/build_transcript_index.py \
  --transcripts-dir "$TRANSCRIPTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --day "$DAY" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE"
