#!/bin/bash
#SBATCH --job-name=cache_qa_model
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/cache_qa_model-%j.out
#SBATCH --error=slurm/cache_qa_model-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

python scripts/cache_qa_model.py
