#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR}"
CAPTIONS_OUTPUT="${CAPTIONS_OUTPUT:-artifacts/qwen_keyframe_captions_tiny.jsonl}"
INDEX_OUTPUT="${INDEX_OUTPUT:-artifacts/multimodal_index_tiny}"
CAPTION_LIMIT="${CAPTION_LIMIT:-5}"
CHUNK_LIMIT="${CHUNK_LIMIT:-0}"
TOP_K="${TOP_K:-5}"
QUESTION="${QUESTION:-What are they cooking?}"

echo "== Tiny multimodal-caption demo =="
echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Captions: $CAPTIONS_OUTPUT"
echo "Index: $INDEX_OUTPUT"
echo "Caption limit: $CAPTION_LIMIT"
echo "Chunk limit: $CHUNK_LIMIT (0 means all transcript chunks)"
echo "Question: $QUESTION"
echo

echo "== Step 1: Generate Qwen captions =="
python scripts/generate_qwen_keyframe_captions.py \
  --dataset-root "$DATASET_ROOT" \
  --output "$CAPTIONS_OUTPUT" \
  --limit "$CAPTION_LIMIT"

echo
echo "== Step 2: Build CPU-queryable multimodal index =="
if [[ "$CHUNK_LIMIT" -gt 0 ]]; then
  python scripts/build_multimodal_index.py \
    --dataset-root "$DATASET_ROOT" \
    --captions "$CAPTIONS_OUTPUT" \
    --output-dir "$INDEX_OUTPUT" \
    --limit-chunks "$CHUNK_LIMIT"
else
  python scripts/build_multimodal_index.py \
    --dataset-root "$DATASET_ROOT" \
    --captions "$CAPTIONS_OUTPUT" \
    --output-dir "$INDEX_OUTPUT"
fi

echo
echo "== Step 3: Query saved multimodal index without live Qwen =="
CUDA_VISIBLE_DEVICES="" python scripts/query_index.py "$QUESTION" \
  --index-dir "$INDEX_OUTPUT" \
  --top-k "$TOP_K"
