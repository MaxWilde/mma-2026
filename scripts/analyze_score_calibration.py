#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_query_evidence import parse_question_table, resolve_path  # noqa: E402
from scripts.query_evidence import retrieve_visual  # noqa: E402
from src.transcript_retrieval import retrieve_transcript_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw and normalized visual/transcript score calibration.")
    parser.add_argument("questions_md", type=Path)
    parser.add_argument("--output-dir", default="artifacts/score_calibration_analysis")
    parser.add_argument("--visual-index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    rows = parse_question_table(resolve_path(args.questions_md))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['ID']}: {row['Question']}", flush=True)
        records.append(analyze_question(row, args))

    summary = build_summary(records, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)
    print(f"Wrote {output_dir / 'summary.md'}", flush=True)


def analyze_question(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    query_args = argparse.Namespace(
        question=row["Question"],
        visual_index_dir=args.visual_index_dir,
        transcript_index_dir=args.transcript_index_dir,
        text_model_name=None,
        top_k=args.top_k,
        retrieval_top_k=args.top_k,
        visual_query_variants=True,
        synonyms_file=None,
        diversity_window_sec=0.0,
        candidate_multiplier=2,
        allow_download=False,
    )
    visual_results, visual_runtime = retrieve_visual(query_args)
    transcript_results = retrieve_transcript_evidence(
        row["Question"],
        args.transcript_index_dir,
        top_k=args.top_k,
        dense_k=args.top_k,
        lexical_k=args.top_k,
        rerank_k=args.top_k,
        use_cross_encoder=True,
        align_playback=False,
    )
    visual_raw = [float(item.get("score", 0.0)) for item in visual_results[: args.top_k]]
    transcript_raw = [float(item.get("score", 0.0)) for item in transcript_results[: args.top_k]]
    visual_norm = normalize_by_max(visual_raw)
    transcript_norm = normalize_by_max(transcript_raw)
    return {
        "id": row["ID"],
        "expected_type": row["Type"].strip().lower(),
        "question": row["Question"],
        "visual_count": len(visual_raw),
        "transcript_count": len(transcript_raw),
        "visual_raw_distribution": distribution(visual_raw),
        "transcript_raw_distribution": distribution(transcript_raw),
        "visual_normalized_distribution": distribution(visual_norm),
        "transcript_normalized_distribution": distribution(transcript_norm),
        "visual_raw_mean_minus_transcript_raw_mean": safe_mean(visual_raw) - safe_mean(transcript_raw),
        "visual_norm_mean_minus_transcript_norm_mean": safe_mean(visual_norm) - safe_mean(transcript_norm),
        "strongest_visual_raw": max(visual_raw, default=0.0),
        "strongest_transcript_raw": max(transcript_raw, default=0.0),
        "strongest_visual_norm": max(visual_norm, default=0.0),
        "strongest_transcript_norm": max(transcript_norm, default=0.0),
        "visual_runtime_sec": visual_runtime,
    }


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    visual_raw_all = flatten_distribution_values(records, "visual_raw_distribution")
    transcript_raw_all = flatten_distribution_values(records, "transcript_raw_distribution")
    visual_norm_all = flatten_distribution_values(records, "visual_normalized_distribution")
    transcript_norm_all = flatten_distribution_values(records, "transcript_normalized_distribution")
    raw_mean_diffs = [record["visual_raw_mean_minus_transcript_raw_mean"] for record in records]
    norm_mean_diffs = [record["visual_norm_mean_minus_transcript_norm_mean"] for record in records]
    transcript_mean_larger = sum(1 for value in raw_mean_diffs if value < 0)
    visual_mean_larger = sum(1 for value in raw_mean_diffs if value > 0)
    transcript_top_larger = sum(1 for record in records if record["strongest_transcript_raw"] > record["strongest_visual_raw"])
    visual_top_larger = sum(1 for record in records if record["strongest_visual_raw"] > record["strongest_transcript_raw"])

    conclusion = calibration_conclusion(
        records=records,
        transcript_mean_larger=transcript_mean_larger,
        visual_mean_larger=visual_mean_larger,
        transcript_top_larger=transcript_top_larger,
        visual_top_larger=visual_top_larger,
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions": len(records),
        "top_k_requested": args.top_k,
        "visual_index_dir": args.visual_index_dir,
        "transcript_index_dir": args.transcript_index_dir,
        "raw_score_distributions": {
            "visual": aggregate_distribution(records, "visual_raw_distribution"),
            "transcript": aggregate_distribution(records, "transcript_raw_distribution"),
        },
        "per_modality_normalized_distributions": {
            "visual": aggregate_distribution(records, "visual_normalized_distribution"),
            "transcript": aggregate_distribution(records, "transcript_normalized_distribution"),
        },
        "all_candidate_raw_distribution": {
            "visual": distribution(visual_raw_all),
            "transcript": distribution(transcript_raw_all),
        },
        "all_candidate_normalized_distribution": {
            "visual": distribution(visual_norm_all),
            "transcript": distribution(transcript_norm_all),
        },
        "raw_mean_difference_visual_minus_transcript": distribution(raw_mean_diffs),
        "normalized_mean_difference_visual_minus_transcript": distribution(norm_mean_diffs),
        "questions_where_transcript_raw_mean_larger": transcript_mean_larger,
        "questions_where_visual_raw_mean_larger": visual_mean_larger,
        "questions_where_transcript_top_raw_larger": transcript_top_larger,
        "questions_where_visual_top_raw_larger": visual_top_larger,
        "answers": conclusion,
        "per_question": records,
    }


def calibration_conclusion(
    *,
    records: list[dict[str, Any]],
    transcript_mean_larger: int,
    visual_mean_larger: int,
    transcript_top_larger: int,
    visual_top_larger: int,
) -> dict[str, str]:
    total = max(1, len(records))
    transcript_mean_rate = transcript_mean_larger / total
    transcript_top_rate = transcript_top_larger / total
    if transcript_mean_rate >= 0.8 and transcript_top_rate >= 0.8:
        raw_answer = "Yes. Transcript raw scores are systematically larger than visual raw scores for this evaluation set."
    elif visual_mean_larger / total >= 0.8:
        raw_answer = "No. Visual raw scores are usually larger in this run."
    else:
        raw_answer = "Partially. Raw score dominance varies by query, so direct comparison is not stable."
    return {
        "are_transcript_scores_systematically_larger": raw_answer,
        "after_per_modality_normalization_are_distributions_similar": (
            "Per-modality max normalization makes each modality's best candidate equal to 1.0, "
            "but it does not guarantee similar tail distributions. Compare the normalized mean/median/std fields."
        ),
        "is_mixed_ranker_relying_on_incomparable_scores": (
            "Raw scores are produced by different retrieval models/objectives, so direct visual-vs-transcript raw score "
            "comparison is not meaningful. Per-modality normalization avoids direct raw comparison but loses absolute calibration."
        ),
        "would_zscore_or_percentile_be_more_appropriate": (
            "Percentile normalization is likely safer for top-k display because it is robust to modality-specific score scales "
            "and bounded candidate lists. Z-score normalization can help but is more sensitive to small top-k samples and outliers."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Score Calibration Analysis",
        "",
        "This analysis does not modify retrieval, routing, ranking, indexes, timestamps, grounding, or transcript QA.",
        "",
        f"- Questions: {summary['questions']}",
        f"- Top-k requested per modality: {summary['top_k_requested']}",
        "",
        "## Direct Answers",
        "",
    ]
    for key, value in summary["answers"].items():
        lines.append(f"### {key.replace('_', ' ').capitalize()}")
        lines.append("")
        lines.append(value)
        lines.append("")
    lines.extend(
        [
            "## Aggregate Raw Score Distributions",
            "",
            f"- Visual raw: `{format_distribution(summary['all_candidate_raw_distribution']['visual'])}`",
            f"- Transcript raw: `{format_distribution(summary['all_candidate_raw_distribution']['transcript'])}`",
            "",
            "## Per-Modality Normalized Score Distributions",
            "",
            f"- Visual normalized: `{format_distribution(summary['all_candidate_normalized_distribution']['visual'])}`",
            f"- Transcript normalized: `{format_distribution(summary['all_candidate_normalized_distribution']['transcript'])}`",
            "",
            "## Question-Level Mean Difference",
            "",
            "`visual_mean - transcript_mean`:",
            "",
            f"- Raw: `{format_distribution(summary['raw_mean_difference_visual_minus_transcript'])}`",
            f"- Normalized: `{format_distribution(summary['normalized_mean_difference_visual_minus_transcript'])}`",
            "",
            "## Count Summary",
            "",
            f"- Questions where transcript raw mean is larger: {summary['questions_where_transcript_raw_mean_larger']}",
            f"- Questions where visual raw mean is larger: {summary['questions_where_visual_raw_mean_larger']}",
            f"- Questions where transcript top raw score is larger: {summary['questions_where_transcript_top_raw_larger']}",
            f"- Questions where visual top raw score is larger: {summary['questions_where_visual_top_raw_larger']}",
            "",
            "## Per-Question Table",
            "",
            "| ID | Expected | Visual raw mean | Transcript raw mean | Visual norm mean | Transcript norm mean | Visual top raw | Transcript top raw |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in summary["per_question"]:
        lines.append(
            "| {id} | {expected} | {vr:.4f} | {tr:.4f} | {vn:.4f} | {tn:.4f} | {vt:.4f} | {tt:.4f} |".format(
                id=record["id"],
                expected=record["expected_type"],
                vr=record["visual_raw_distribution"]["mean"] or 0.0,
                tr=record["transcript_raw_distribution"]["mean"] or 0.0,
                vn=record["visual_normalized_distribution"]["mean"] or 0.0,
                tn=record["transcript_normalized_distribution"]["mean"] or 0.0,
                vt=record["strongest_visual_raw"],
                tt=record["strongest_transcript_raw"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def aggregate_distribution(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [record[key]["mean"] for record in records if record[key]["mean"] is not None]
    return distribution(values)


def flatten_distribution_values(records: list[dict[str, Any]], key: str) -> list[float]:
    # The per-question records intentionally store summary statistics, not all raw candidate scores.
    # For an all-candidate approximation, weight each question mean by candidate count.
    values: list[float] = []
    for record in records:
        dist = record[key]
        if dist["mean"] is None:
            continue
        values.extend([float(dist["mean"])] * int(dist["n"]))
    return values


def normalize_by_max(values: list[float]) -> list[float]:
    max_value = max(values, default=0.0)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [value / max_value for value in values]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "p01": None, "p05": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = round((len(values) - 1) * p)
    return values[index]


def safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def format_distribution(dist: dict[str, Any]) -> str:
    if not dist.get("n"):
        return "n=0"
    return (
        f"n={dist['n']}, mean={dist['mean']:.4f}, std={dist['std']:.4f}, "
        f"min={dist['min']:.4f}, p25={dist['p25']:.4f}, median={dist['median']:.4f}, "
        f"p75={dist['p75']:.4f}, max={dist['max']:.4f}"
    )


if __name__ == "__main__":
    main()
