#!/bin/bash

# One-time copy of import-heavy Python packages and dashboard models from
# shared scratch into the user's home filesystem. Run as a short CPU Slurm job,
# not repeatedly at dashboard startup.

#SBATCH --partition=rome
#SBATCH --job-name=castle-fast-cache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -eu

REPO_ROOT="/home/scur0260/mma-2026"
VENV_SHARED="/gpfs/scratch1/shared/group_h/data_goncalo/.venv"
VENV_HOME="${HOME}/.castle-venv"
PYTHON="${VENV_SHARED}/bin/python3"
PY_VER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SOURCE_SITE="${VENV_SHARED}/lib/python${PY_VER}/site-packages"
TARGET_SITE="${VENV_HOME}/lib/python${PY_VER}/site-packages"
MODEL_TARGET="${REPO_ROOT}/castle-rag-dashboard-2/models"
MODEL_SOURCE="/gpfs/scratch1/shared/group_h/models"

mkdir -p "${TARGET_SITE}" "${MODEL_TARGET}"

echo "[cache] copying import-heavy Python packages to ${TARGET_SITE}"
for pattern in \
    torch torch-* torchgen \
    torchvision torchvision-* \
    transformers transformers-* \
    tokenizers tokenizers-* \
    safetensors safetensors-* \
    huggingface_hub huggingface_hub-* \
    sentence_transformers sentence_transformers-* \
    scipy scipy-* scipy.libs \
    sklearn scikit_learn-* scikit_learn.libs \
    numpy numpy-* numpy.libs \
    PIL pillow-* pillow.libs \
    faiss faiss_cpu-* faiss_cpu.libs \
    regex regex-*; do
    for source in "${SOURCE_SITE}"/${pattern}; do
        [ -e "${source}" ] || continue
        rsync -a "${source}" "${TARGET_SITE}/"
    done
done

echo "[cache] copying SigLIP2 text model"
rsync -a \
    "${MODEL_SOURCE}/siglip2-so400m-patch16-512-text/" \
    "${MODEL_TARGET}/siglip2-so400m-patch16-512-text/"

echo "[cache] copying extractive QA model"
rsync -a \
    "${MODEL_SOURCE}/distilbert-base-cased-distilled-squad/" \
    "${MODEL_TARGET}/distilbert-base-cased-distilled-squad/"

echo "[cache] validating local package imports"
PYTHONPATH="${TARGET_SITE}" "${PYTHON}" -c \
    "import torch, transformers, sentence_transformers, scipy, sklearn, numpy, faiss; print('package cache OK')"

test -f "${MODEL_TARGET}/siglip2-so400m-patch16-512-text/model.safetensors"
test -f "${MODEL_TARGET}/distilbert-base-cased-distilled-squad/model.safetensors"

echo "[cache] fast dashboard cache is ready"
du -sh "${VENV_HOME}" \
    "${MODEL_TARGET}/siglip2-so400m-patch16-512-text" \
    "${MODEL_TARGET}/distilbert-base-cased-distilled-squad"
