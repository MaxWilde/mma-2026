#!/bin/bash
#SBATCH --job-name=test_timestamp_refinement
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/test_timestamp_refinement-%j.out
#SBATCH --error=slurm/test_timestamp_refinement-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

nvidia-smi

run_query() {
  local question="$1"
  echo
  echo "============================================================"
  echo "Question: ${question}"
  echo "============================================================"
  python scripts/query_evidence.py \
    "${question}" \
    --visual-index-dir artifacts/siglip_index_day1 \
    --transcript-index-dir artifacts/transcript_index_day1 \
    --debug-router
}

run_query "What is the answer to the question regarding who was the front man of the British rock band The Police, famous for the 1980s hits like Every Breath You Take?"
run_query "How long would the water not boil for?"
