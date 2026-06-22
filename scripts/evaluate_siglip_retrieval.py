#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_clip_index import (  # noqa: E402
    collect_variant_results,
    load_faiss_index,
    load_metadata,
    load_synonym_map,
    merge_variant_results,
    print_result,
    print_variant_results,
    query_variants,
    resolve_query_model_name,
)
from src.clip_retrieval import embed_texts_clip_profile, load_clip_text_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual retrieval diagnostics for SigLIP over a JSONL question set.")
    parser.add_argument("--input", required=True, help='JSONL with records like {"question": "...", "expected_object": "..."}')
    parser.add_argument("--index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-variants", action="store_true")
    parser.add_argument("--merge-variants", action="store_true")
    parser.add_argument("--synonyms-file", default=None, help="Optional JSON synonym map for query expansion. Disabled by default.")
    parser.add_argument("--no-visual-templates", action="store_true")
    parser.add_argument("--diversity-window-sec", type=float, default=30.0)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    metadata, index_model_name = load_metadata(args.index_dir)
    index = load_faiss_index(args.index_dir)
    query_model_name = resolve_query_model_name(index_model_name, args.text_model_name, "none")
    model, processor, torch = load_clip_text_model(query_model_name, local_files_only=not args.allow_download)
    print(f"Loaded index entries: {len(metadata)}")
    print(f"Index model: {index_model_name}")
    print(f"Query model: {query_model_name}")
    print(f"Startup time: {time.perf_counter() - start:.3f}s")

    records = load_jsonl(args.input)
    synonym_map = load_synonym_map(args.synonyms_file)
    for qidx, record in enumerate(records, start=1):
        question = str(record.get("question", "")).strip()
        expected_object = str(record.get("expected_object", "")).strip()
        if not question:
            continue
        variants = query_variants(
            question,
            synonym_map,
            include_visual_templates=not args.no_visual_templates,
        ) if args.query_variants else [question]
        search_k = min(max(args.top_k, args.top_k * max(1, args.candidate_multiplier)), len(metadata))
        query_start = time.perf_counter()
        embeddings, profile = embed_texts_clip_profile(variants, model=model, processor=processor, torch=torch)
        scores, ids = index.search(embeddings, search_k)
        variant_results = collect_variant_results(scores, ids, metadata, variants, args.top_k, args.diversity_window_sec)

        print("\n" + "=" * 100)
        print(f"[{qidx}] question: {question}")
        print(f"expected_object: {expected_object or 'n/a'}")
        print(f"variants: {' | '.join(variants)}")
        print(f"tokenizer: {profile['tokenizer_time_sec']:.3f}s")
        print(f"text_forward: {profile['text_forward_time_sec']:.3f}s")
        print(f"normalization: {profile['normalization_time_sec']:.3f}s")
        print(f"query_total: {time.perf_counter() - query_start:.3f}s")
        print_variant_results(variant_results)
        if args.merge_variants:
            merged = merge_variant_results(variant_results, args.top_k, args.diversity_window_sec)
            print("\nMERGED VARIANT RANKING")
            for rank, result in enumerate(merged, start=1):
                print_result(result, rank)


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(data, dict):
                raise SystemExit(f"Expected JSON object on line {line_number}")
            records.append(data)
    return records


if __name__ == "__main__":
    main()
