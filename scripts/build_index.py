#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import load_evidence_items
from src.retriever import (
    DEFAULT_MODEL_NAME,
    build_faiss_index,
    embed_texts,
    evidence_to_metadata,
    load_embedding_model,
    save_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over transcript chunks.")
    parser.add_argument("--dataset-root", default="/scratch-shared/group_h/data_goncalo")
    parser.add_argument("--output-dir", default="artifacts/transcript_index")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    items = load_evidence_items(args.dataset_root)
    if not items:
        raise SystemExit("No transcript chunks found.")

    model = load_embedding_model(args.model_name)
    embeddings = embed_texts(model, [item.text for item in items], batch_size=args.batch_size)
    index = build_faiss_index(embeddings)
    save_index(index, evidence_to_metadata(items), args.output_dir, args.model_name)

    print(f"Indexed {len(items)} chunks")
    print(f"Saved FAISS index and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
