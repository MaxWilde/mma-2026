#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_clip_index import (  # noqa: E402
    collect_results,
    collect_variant_results,
    load_faiss_index,
    load_metadata,
    load_synonym_map,
    merge_variant_results,
    query_variants,
    resolve_query_model_name,
)
from src.clip_retrieval import embed_texts_clip_profile, load_clip_text_model  # noqa: E402
from src.evidence_links import youtube_timestamp_url  # noqa: E402
from src.vqa import format_timestamp  # noqa: E402


DEFAULT_QUESTIONS = [
    "What color is the refrigerator?",
    "Where is the kettle?",
    "What is on the stove?",
    "Where is the fireplace?",
    "What is on the couch?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SigLIP retrieval contact-sheet and HTML report.")
    parser.add_argument("--questions", default=None, help="Optional .txt or .jsonl question file. Defaults to five built-in diagnostic questions.")
    parser.add_argument("--index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--output-dir", default="artifacts/retrieval_reports/siglip_day1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-variants", action="store_true")
    parser.add_argument("--merge-variants", action="store_true")
    parser.add_argument("--synonyms-file", default=None, help="Optional JSON synonym map. Disabled by default.")
    parser.add_argument("--no-visual-templates", action="store_true")
    parser.add_argument("--diversity-window-sec", type=float, default=30.0)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sheets_dir = output_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    startup = time.perf_counter()
    metadata, index_model_name = load_metadata(args.index_dir)
    index = load_faiss_index(args.index_dir)
    query_model_name = resolve_query_model_name(index_model_name, args.text_model_name, "none")
    model, processor, torch = load_clip_text_model(query_model_name, local_files_only=not args.allow_download)
    startup_sec = time.perf_counter() - startup

    questions = load_questions(args.questions)
    synonym_map = load_synonym_map(args.synonyms_file)

    report_records: list[dict[str, Any]] = []
    for qidx, question in enumerate(questions, start=1):
        print(f"[{qidx}/{len(questions)}] {question}", flush=True)
        query_start = time.perf_counter()
        variants = (
            query_variants(
                question,
                synonym_map,
                include_visual_templates=not args.no_visual_templates,
            )
            if args.query_variants
            else [question]
        )
        search_k = min(max(args.top_k, args.top_k * max(1, args.candidate_multiplier)), len(metadata))
        embeddings, profile = embed_texts_clip_profile(variants, model=model, processor=processor, torch=torch)
        scores, ids = index.search(embeddings, search_k)

        if args.query_variants:
            variant_results = collect_variant_results(scores, ids, metadata, variants, args.top_k, args.diversity_window_sec)
            results = merge_variant_results(variant_results, args.top_k, args.diversity_window_sec) if args.merge_variants else variant_results[0]["results"]
        else:
            variant_results = []
            results = collect_results(scores, ids, metadata)[: args.top_k]

        normalized_results = [normalize_result(item, rank) for rank, item in enumerate(results, start=1)]
        sheet_path = sheets_dir / f"{qidx:02d}_{slugify(question)}.jpg"
        make_contact_sheet(question, normalized_results, sheet_path)

        report_records.append(
            {
                "question": question,
                "variants": variants,
                "contact_sheet": str(sheet_path.relative_to(output_dir)),
                "query_time_sec": time.perf_counter() - query_start,
                "embedding_profile": profile,
                "results": normalized_results,
                "variant_results": [
                    {
                        "variant": group["variant"],
                        "results": [normalize_result(item, rank) for rank, item in enumerate(group["results"], start=1)],
                    }
                    for group in variant_results
                ],
            }
        )

    summary = {
        "index_dir": args.index_dir,
        "index_entries": len(metadata),
        "index_model_name": index_model_name,
        "query_model_name": query_model_name,
        "top_k": args.top_k,
        "query_variants": args.query_variants,
        "merge_variants": args.merge_variants,
        "diversity_window_sec": args.diversity_window_sec,
        "startup_sec": startup_sec,
        "questions": report_records,
    }
    results_path = output_dir / "retrieval_results.json"
    html_path = output_dir / "index.html"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_html_report(summary, html_path)
    print(f"Results JSON: {results_path}")
    print(f"HTML report: {html_path}")


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    question_path = Path(path)
    if question_path.suffix.lower() == ".jsonl":
        questions: list[str] = []
        with question_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                question = str(data.get("question", "")).strip()
                if not question:
                    raise SystemExit(f"Missing question on line {line_number}: {path}")
                questions.append(question)
        return questions
    with question_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    start_sec = safe_float(item.get("start_sec", item.get("keyframe_time_sec")), 0.0)
    end_sec = safe_float(item.get("end_sec", item.get("keyframe_time_sec")), start_sec)
    return {
        "rank": rank,
        "score": float(item.get("score", 0.0)),
        "day": item.get("day"),
        "source_name": item.get("source_name"),
        "video_id": item.get("video_id"),
        "hour_id": item.get("hour_id", item.get("video_id")),
        "keyframe_path": item.get("keyframe_path"),
        "frame_number": item.get("frame_number"),
        "keyframe_time_sec": safe_float(item.get("keyframe_time_sec", start_sec), start_sec),
        "timestamp": format_timestamp(start_sec, end_sec),
        "youtube_url": youtube_timestamp_url(item),
        "query_variant": item.get("query_variant"),
        "matched_variants": item.get("matched_variants", []),
    }


def make_contact_sheet(question: str, results: list[dict[str, Any]], output_path: Path) -> None:
    thumb_size = (280, 158)
    label_height = 54
    columns = 5
    rows = max(1, (len(results) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + label_height) + 42), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), question, fill=(0, 0, 0), font=font)

    for idx, result in enumerate(results):
        col = idx % columns
        row = idx // columns
        x = col * thumb_size[0]
        y = 42 + row * (thumb_size[1] + label_height)
        image_path = Path(str(result["keyframe_path"]))
        try:
            with Image.open(image_path) as image:
                thumb = ImageOps.contain(image.convert("RGB"), thumb_size)
        except Exception:
            thumb = Image.new("RGB", thumb_size, (235, 235, 235))
        tile = Image.new("RGB", thumb_size, (248, 248, 248))
        tile.paste(thumb, ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1), outline=(60, 60, 60), width=1)
        label = [
            f"#{result['rank']} score={result['score']:.4f}",
            f"{result.get('source_name')} {result.get('day')} {result.get('video_id')} {result['timestamp']}",
            f"frame={result.get('frame_number')}",
        ]
        for line_idx, line in enumerate(label):
            draw.text((x + 4, y + thumb_size[1] + 4 + line_idx * 14), line[:44], fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_html_report(summary: dict[str, Any], output_path: Path) -> None:
    rows = []
    for record in summary["questions"]:
        results_rows = []
        for item in record["results"]:
            youtube = item.get("youtube_url") or ""
            youtube_cell = f'<a href="{html.escape(youtube)}">YouTube</a>' if youtube else ""
            results_rows.append(
                "<tr>"
                f"<td>{item['rank']}</td>"
                f"<td>{item['score']:.4f}</td>"
                f"<td>{html.escape(str(item.get('source_name')))}</td>"
                f"<td>{html.escape(str(item.get('day')))}</td>"
                f"<td>{html.escape(str(item.get('video_id')))}</td>"
                f"<td>{html.escape(str(item.get('timestamp')))}</td>"
                f"<td><code>{html.escape(str(item.get('keyframe_path')))}</code></td>"
                f"<td>{youtube_cell}</td>"
                "</tr>"
            )
        variants = " | ".join(record["variants"])
        rows.append(
            "<section>"
            f"<h2>{html.escape(record['question'])}</h2>"
            f"<p><strong>Variants:</strong> {html.escape(variants)}</p>"
            f"<p><strong>Query time:</strong> {record['query_time_sec']:.3f}s</p>"
            f"<img class=\"sheet\" src=\"{html.escape(record['contact_sheet'])}\" alt=\"contact sheet\">"
            "<table><thead><tr>"
            "<th>Rank</th><th>Score</th><th>Source</th><th>Day</th><th>Hour</th><th>Timestamp</th><th>Keyframe</th><th>YouTube</th>"
            "</tr></thead><tbody>"
            + "\n".join(results_rows)
            + "</tbody></table></section>"
        )

    html_text = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>SigLIP Retrieval Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}"
        "section{margin:0 0 42px 0;padding-bottom:28px;border-bottom:1px solid #ddd}"
        ".sheet{max-width:100%;border:1px solid #ccc}"
        "table{border-collapse:collapse;width:100%;margin-top:12px;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f3f3f3;text-align:left}"
        "code{font-size:12px}"
        "</style></head><body>"
        "<h1>SigLIP Retrieval Report</h1>"
        f"<p><strong>Index:</strong> {html.escape(summary['index_dir'])}</p>"
        f"<p><strong>Entries:</strong> {summary['index_entries']}</p>"
        f"<p><strong>Query model:</strong> {html.escape(summary['query_model_name'])}</p>"
        f"<p><strong>Top-k:</strong> {summary['top_k']}</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:80] or "query"


if __name__ == "__main__":
    main()
