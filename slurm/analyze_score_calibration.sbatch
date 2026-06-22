#!/bin/bash
#SBATCH --job-name=score_calibration
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/analyze_score_calibration-%j.out
#SBATCH --error=slurm/analyze_score_calibration-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

nvidia-smi
python scripts/analyze_score_calibration.py castle_day1_ground_truth_questions.md
