#!/bin/bash
#SBATCH --job-name=eval_transcript_answer_rerank
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/eval_transcript_answer_rerank-%j.out
#SBATCH --error=slurm/eval_transcript_answer_rerank-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

nvidia-smi

BASELINE_ROOT="artifacts/transcript_rerank_eval_baseline"
RERANK_ROOT="artifacts/transcript_rerank_eval_rerank"
rm -rf "$BASELINE_ROOT" "$RERANK_ROOT"
mkdir -p "$BASELINE_ROOT" "$RERANK_ROOT"

python scripts/batch_query_evidence.py \
  castle_day1_ground_truth_questions.md \
  --output-root "$BASELINE_ROOT" \
  --mixed-top-k 10

python scripts/batch_query_evidence.py \
  castle_day1_ground_truth_questions.md \
  --output-root "$RERANK_ROOT" \
  --mixed-top-k 10 \
  --transcript-answer-rerank-top-n 20

BASELINE_DIR="$(find "$BASELINE_ROOT" -maxdepth 1 -type d -name 'batch_eval_day1_*' | sort | tail -1)"
RERANK_DIR="$(find "$RERANK_ROOT" -maxdepth 1 -type d -name 'batch_eval_day1_*' | sort | tail -1)"

python scripts/compare_transcript_rerank_eval.py \
  --baseline-dir "$BASELINE_DIR" \
  --rerank-dir "$RERANK_DIR" \
  --output artifacts/transcript_rerank_comparison.md
