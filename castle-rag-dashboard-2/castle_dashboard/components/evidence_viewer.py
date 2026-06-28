from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from dash import dcc, html

from castle_dashboard.models.schemas import RetrievalResult



def build_evidence_viewer(
    result: RetrievalResult | None,
    grounding: dict[str, Any] | None = None,
    transcript_context: list[dict[str, Any]] | None = None,
    transcript_evidence: dict[str, Any] | None = None,
) -> html.Section:
    if result is None:
        return html.Section(
            className="panel empty-state",
            children=[
                _action_buttons(None),
                "Run a search and select a result to inspect evidence.",
            ],
        )

    youtube_url = _yt_url(result)
    is_transcript = result.modality == "transcript"

    return html.Section(
        className="panel evidence-panel",
        children=[
            # Heading row
            html.Div(
                className="section-heading",
                children=[
                    html.H2("Evidence viewer"),
                    html.Span(
                        f"{result.viewpoint} · {_combined_ts(result)}",
                        className="soft-pill",
                    ),
                    html.A(
                        "▶ YouTube",
                        id="youtube-evidence-link",
                        href=youtube_url,
                        target="_blank",
                        className="yt-link",
                        n_clicks=0,
                    ) if youtube_url else None,
                ],
            ),
            # Body differs by evidence type
            *(_transcript_evidence_body(result, transcript_context, transcript_evidence, youtube_url)
              if is_transcript
              else _visual_evidence_body(result, grounding, transcript_context, transcript_evidence)),
            # Action buttons
            _action_buttons(result),
        ],
    )


def _visual_evidence_body(
    result: RetrievalResult,
    grounding: dict[str, Any] | None,
    transcript_context: list[dict[str, Any]] | None,
    transcript_evidence: dict[str, Any] | None,
) -> list:
    transcript_content = _transcript_tab(result, transcript_context, transcript_evidence)
    return [
        _image_area(result, grounding),
        _grounding_bar(grounding) if grounding else None,
        # Collapsible transcript — hidden by default, no callback needed
        html.Details(
            id="visual-transcript-details",
            className="transcript-details",
            children=[
                html.Summary("Transcript", className="transcript-summary"),
                transcript_content,
            ],
        ),
    ]


def _transcript_evidence_body(
    result: RetrievalResult,
    transcript_context: list[dict[str, Any]] | None,
    transcript_evidence: dict[str, Any] | None,
    youtube_url: str | None,
) -> list:
    embed_src = _yt_embed_url(result)
    player = (
        html.Iframe(
            src=embed_src,
            className="yt-player",
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen",
        )
        if embed_src else None
    )
    transcript_content = _transcript_tab(result, transcript_context, transcript_evidence)
    items = [player] if player else []
    items.append(transcript_content)
    return items


# ── image area ────────────────────────────────────────────────────────────────

def _image_area(result: RetrievalResult, grounding: dict[str, Any] | None) -> html.Div:
    # Use grounding output image when available and matching; else original keyframe
    show_grounding_img = (
        grounding is not None
        and grounding.get("output_image_path")
        and grounding.get("method") == "bbox"
        and grounding.get("keyframe_path") == result.keyframe_path
    )

    if show_grounding_img:
        rel = grounding["output_image_path"]
        img_src = f"/grounding/{rel}"
    elif result.keyframe_path:
        img_src = f"/keyframe?path={quote(result.keyframe_path)}"
    else:
        img_src = ""

    children: list = [
        html.Img(src=img_src, className="keyframe-image", alt=result.title)
        if img_src else html.Div("No keyframe available.", className="no-keyframe"),
    ]

    # "View original" toggle when grounding image is shown
    if show_grounding_img and result.keyframe_path:
        orig_src = f"/keyframe?path={quote(result.keyframe_path)}"
        children.append(
            html.A(
                "View original frame",
                id="original-frame-link",
                href=orig_src,
                target="_blank",
                className="orig-frame-link",
                n_clicks=0,
            )
        )

    return html.Div(className="keyframe-shell", children=children)


def _grounding_bar(grounding: dict[str, Any]) -> html.Div | None:
    if not grounding or grounding.get("method") == "none":
        label = grounding.get("label", "No box found") if grounding else ""
        return html.Div(f"Localization: {label}", className="grounding-bar grounding-miss")

    box = grounding.get("box_xyxy")
    conf = grounding.get("confidence")
    target = grounding.get("grounding_target", "")
    box_str = (
        f"[{', '.join(f'{v:.0f}' for v in box)}]" if box else "n/a"
    )
    conf_str = f"{conf:.3f}" if conf is not None else "n/a"
    return html.Div(
        className="grounding-bar grounding-hit",
        children=[
            html.Span(f"Target: {target}", className="grounding-target"),
            html.Span(f"Box: {box_str}", className="grounding-box"),
            html.Span(f"Conf: {conf_str}", className="grounding-conf"),
        ],
    )


# ── tabs ──────────────────────────────────────────────────────────────────────

def _transcript_tab(
    result: RetrievalResult,
    context: list[dict[str, Any]] | None,
    transcript_evidence: dict[str, Any] | None = None,
) -> html.Div:
    rows = []
    context_rows_added = False

    # QA is an overlay, not a replacement. The highlighted selected passage can
    # be collapsed while the complete ±2 minute transcript remains visible.
    heatmap = (transcript_evidence or {}).get("transcript_heatmap")
    if heatmap and result.modality == "transcript":
        rows.append(
            html.Details(
                open=True,
                className="transcript-highlight-details",
                children=[
                    html.Summary(
                        "QA answer and relevance overlay — click to hide",
                        className="transcript-summary",
                    ),
                    _heatmap_block(transcript_evidence),
                ],
            )
        )

    # Context chunks from transcript index (±2 min window)
    if context:
        for chunk in context:
            start = chunk.get("start_sec", 0)
            end = chunk.get("end_sec", start)
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            ts = f"{_fmt(start)} – {_fmt(end)}"
            is_current = abs(start - result.start_sec) < 5
            row_cls = "transcript-row transcript-current" if is_current else "transcript-row"
            rows.append(
                html.Div(
                    className=row_cls,
                    children=[
                        html.Span(ts, className="time-pill"),
                        html.P(text),
                    ],
                )
            )
            context_rows_added = True

    # Fall back to result's own transcript segments if no context. A heatmap
    # overlay does not count as transcript context for this decision.
    if not context_rows_added:
        for seg in result.transcript:
            rows.append(
                html.Div(
                    className="transcript-row",
                    children=[
                        html.Span(f"{seg.start} – {seg.end}", className="time-pill"),
                        html.Strong(seg.speaker),
                        html.P(seg.text),
                    ],
                )
            )

    if not rows:
        rows = [html.P("No transcript available for this segment.", className="muted")]

    return html.Div(className="transcript-list", children=rows)


def _heatmap_block(transcript_evidence: dict[str, Any]) -> html.Div:
    """Renders transcript_heatmap spans as inline HTML with background
    intensity proportional to score (per backend_method_summary §6), plus
    the extractive QA answer span and n-best candidates (§7)."""
    heatmap = transcript_evidence.get("transcript_heatmap") or []
    answer_span = transcript_evidence.get("answer_span") or {}
    candidates = transcript_evidence.get("answer_candidates") or []

    spans = [
        html.Span(
            item.get("text", ""),
            className="heatmap-span",
            style={"backgroundColor": _heat_color(float(item.get("score", 0.0)))},
        )
        for item in heatmap
    ]

    children: list = [html.Div(spans, className="heatmap-text")]

    if answer_span.get("text"):
        children.append(
            html.Div(
                [
                    html.Span("Best answer span: ", className="answer-span-label"),
                    html.Strong(f'"{answer_span["text"]}"'),
                    html.Span(
                        f" (confidence {float(answer_span.get('score', 0.0)) * 100:.0f}%)",
                        className="answer-span-meta",
                    ),
                ],
                className="answer-span-block",
            )
        )

    if candidates:
        children.append(html.P("Other candidate spans:", className="answer-candidates-label"))
        children.append(
            html.Ul(
                [
                    html.Li(f'"{c.get("text", "")}" ({float(c.get("score", 0.0)) * 100:.0f}%)')
                    for c in candidates[1:6]
                ],
                className="answer-candidates-list",
            )
        )

    return html.Div(children, className="transcript-heatmap-block")


def _heat_color(score: float) -> str:
    score = max(0.0, min(1.0, score))
    if score <= 0:
        return "transparent"
    # amber wash, intensity proportional to score
    return f"rgba(245, 158, 11, {0.12 + 0.55 * score:.3f})"


def _metadata_tab(
    result: RetrievalResult,
    grounding: dict[str, Any] | None,
) -> html.Div:
    youtube_url = _yt_url(result)
    score_label = "Cosine similarity (SigLIP2)" if result.modality == "visual" else "Rerank score (cross-encoder/MiniLM)"
    rows = {
        "Source": result.source_name or result.viewpoint,
        "Hour / POV": result.camera,
        "Day": result.day,
        "Timestamp": result.timestamp,
        "Start sec": f"{result.start_sec:.1f}",
        "Confidence (mixed ranker)": f"{result.score * 100:.0f}%",
        score_label: f"{result.raw_score:.4f}",
        "Modality": result.modality,
        "Routed evidence": "yes" if result.is_routed_choice else "no",
        "Keyframe": result.keyframe_path or "n/a",
    }
    if grounding and grounding.get("method") == "bbox":
        rows["DINO confidence"] = f"{grounding['confidence']:.3f}" if grounding.get("confidence") else "n/a"
        rows["DINO target"] = grounding.get("grounding_target") or "n/a"

    items = [html.Div([html.Span(k), html.Strong(v)]) for k, v in rows.items()]
    if youtube_url:
        items.append(
            html.Div([
                html.Span("YouTube"),
                html.A(youtube_url, href=youtube_url, target="_blank", className="yt-link-meta"),
            ])
        )
    return html.Div(className="metadata-grid", children=items)


# ── helpers ───────────────────────────────────────────────────────────────────

def _action_buttons(result: "RetrievalResult | None") -> html.Div:
    localize_disabled = result is None or result.modality != "visual"
    highlight_disabled = result is None or result.modality != "transcript"
    return html.Div(
        className="evidence-actions",
        children=[
            html.Button(
                "Localize (GroundingDINO)",
                id="localize-button",
                n_clicks=0,
                className="secondary-button",
                disabled=localize_disabled,
            ),
            html.Button(
                "Highlight answer (QA)",
                id="highlight-answer-button",
                n_clicks=0,
                className="secondary-button",
                disabled=highlight_disabled,
            ),
        ],
    )


def _fmt(sec: float) -> str:
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _combined_ts(result: RetrievalResult) -> str:
    """Combine camera (hour) with MM:SS from timestamp to get e.g. '16:52:58'."""
    ts = result.timestamp  # "HH:MM:SS"
    mm_ss = ts[3:] if len(ts) == 8 and ts[2] == ":" else ts
    cam = str(result.camera).strip()
    return f"{cam}:{mm_ss}" if cam else ts


def _yt_url(result: RetrievalResult) -> str | None:
    url = result.youtube_url
    if not url:
        return None
    t = int(max(0, result.start_sec))
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={t}"


def _yt_embed_url(result: RetrievalResult) -> str | None:
    """Convert a YouTube watch URL to an embeddable iframe src with start time."""
    url = result.youtube_url
    if not url:
        return None
    t = int(max(0, result.start_sec))
    try:
        parsed = urlparse(url)
        if "youtu.be" in parsed.netloc:
            video_id = parsed.path.lstrip("/")
        else:
            qs = parse_qs(parsed.query)
            ids = qs.get("v", [])
            video_id = ids[0] if ids else ""
        if not video_id:
            return None
        return f"https://www.youtube.com/embed/{video_id}?start={t}"
    except Exception:
        return None
