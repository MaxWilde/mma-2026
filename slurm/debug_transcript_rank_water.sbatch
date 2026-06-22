#!/bin/bash
#SBATCH --job-name=debug_transcript_rank_water
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/debug_transcript_rank_water-%j.out
#SBATCH --error=slurm/debug_transcript_rank_water-%j.err

set -euo pipefail

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch-shared/group_h/data_goncalo
source .venv/bin/activate

nvidia-smi

python - <<'PY'
from __future__ import annotations

from src.transcript_retrieval import candidate_key, retrieve_transcript_evidence_debug

QUESTION = "How long would the water not boil for?"
INDEX_DIR = "artifacts/transcript_index_day1"


def is_known_chunk(item: dict) -> bool:
    text = str(item.get("text", "")).lower()
    source = str(item.get("source_name", "")).lower()
    hour = str(item.get("hour_id", item.get("video_id", "")))
    return (
        source == "stevan"
        and hour == "13"
        and (
            "wouldn't boil the water for an hour" in text
            or "would not boil the water for an hour" in text
            or "boil the water for an hour" in text
        )
    )


def find_rank(items: list[dict], key: str, rank_field: str, score_field: str) -> tuple[object, object]:
    for idx, item in enumerate(items, start=1):
        if candidate_key(item) == key:
            return item.get(rank_field, idx), item.get(score_field, item.get("score"))
    return "not in candidates", None


debug = retrieve_transcript_evidence_debug(
    QUESTION,
    INDEX_DIR,
    top_k=10,
    dense_k=100,
    lexical_k=100,
    rerank_k=50,
    use_cross_encoder=True,
)

known = None
for stage in ("dense", "lexical", "fused", "reranked"):
    for item in debug[stage]:
        if is_known_chunk(item):
            known = item
            break
    if known:
        break

print(f"Question: {QUESTION}")
print(f"Cross-encoder status: {debug['cross_encoder_status']}")
print("Timings:")
for name, value in debug["timings"].items():
    print(f"  {name}: {value:.3f}s")

if known is None:
    print("\nKnown chunk: NOT FOUND in dense/lexical/fused/reranked candidates")
else:
    key = candidate_key(known)
    print("\nKnown chunk:")
    print(f"  source_name: {known.get('source_name')}")
    print(f"  hour_id: {known.get('hour_id', known.get('video_id'))}")
    print(f"  timestamp: {known.get('timestamp')}")
    print(f"  preview: {known.get('transcript_snippet')}")
    dense_rank, dense_score = find_rank(debug["dense"], key, "dense_rank", "dense_score")
    lexical_rank, lexical_score = find_rank(debug["lexical"], key, "lexical_rank", "lexical_score")
    fused_rank, fused_score = find_rank(debug["fused"], key, "fused_rank", "rrf_score")
    reranked_rank, reranked_score = find_rank(debug["reranked"], key, "rerank_rank", "score")
    print(f"  dense rank + score: {dense_rank}, {dense_score}")
    print(f"  lexical rank + score: {lexical_rank}, {lexical_score}")
    print(f"  fused rank + score: {fused_rank}, {fused_score}")
    print(f"  reranked rank + score: {reranked_rank}, {reranked_score}")

print("\nSelected top 10 final results:")
for rank, item in enumerate(debug["reranked"][:10], start=1):
    source = f"{item.get('source_name')} day={item.get('day')} hour={item.get('hour_id', item.get('video_id'))}"
    scores = [
        f"score={float(item.get('score', 0.0)):.4f}",
        f"dense_rank={item.get('dense_rank', '-')}",
        f"dense_score={float(item.get('dense_score', 0.0)):.4f}",
        f"lexical_rank={item.get('lexical_rank', '-')}",
        f"lexical_score={float(item.get('lexical_score', 0.0)):.4f}",
        f"rrf_score={float(item.get('rrf_score', 0.0)):.4f}",
    ]
    if item.get("cross_encoder_score") is not None:
        scores.append(f"cross_encoder_score={float(item['cross_encoder_score']):.4f}")
    if item.get("minilm_passage_score") is not None:
        scores.append(f"minilm_passage_score={float(item['minilm_passage_score']):.4f}")
    if item.get("source_name_boost"):
        scores.append(f"source_name_boost={float(item['source_name_boost']):.4f}")
    print(f"{rank}. {source} {item.get('timestamp')} | " + " | ".join(scores))
    print(f"   {item.get('transcript_snippet', '')}")
PY
