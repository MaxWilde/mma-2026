from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go

from castle_dashboard.models.schemas import RetrievalResult

PLOT_TEMPLATE = "plotly_white"
PRIMARY = "#2563eb"
ACCENT = "#14b8a6"
MUTED = "#94a3b8"
WARNING = "#f59e0b"
TEXT = "#0f172a"


def _base(fig: go.Figure, height: int = 260) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=18, r=18, t=26, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
    )
    return fig


def _empty(message: str = "No results") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig)


def make_score_figure(results: list[RetrievalResult], selected_id: str | None) -> go.Figure:
    if not results:
        return _empty("Run a search to see scores")
    ordered = sorted(results, key=lambda r: r.score)
    colors = [PRIMARY if r.id == selected_id else MUTED for r in ordered]
    label_max = 40
    labels = [
        (r.title[:label_max] + "…" if len(r.title) > label_max else r.title)
        for r in ordered
    ]
    fig = go.Figure(
        go.Bar(
            x=[r.score * 100 for r in ordered],
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=[r.id for r in ordered],
            text=[r.evidence_type for r in ordered],
            hovertemplate="%{y}<br>confidence=%{x:.1f}%% (%{text})<extra></extra>",
        )
    )
    fig.update_xaxes(title="Confidence % (mixed evidence ranker)", range=[0, 100])
    fig.update_yaxes(title="")
    return _base(fig, height=max(220, len(results) * 22))


def make_timeline_figure(results: list[RetrievalResult], selected_id: str | None) -> go.Figure:
    if not results:
        return _empty("Run a search to see the timeline")
    colors = [PRIMARY if r.id == selected_id else ACCENT for r in results]
    sizes = [14 if r.id == selected_id else 9 for r in results]
    fig = go.Figure(
        go.Scatter(
            x=[r.timestamp_seconds / 3600 for r in results],
            y=[r.viewpoint for r in results],
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            customdata=[r.id for r in results],
            text=[r.title for r in results],
            hovertemplate="%{text}<br>%{y}<br>hour=%{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Video hour", showgrid=True)
    fig.update_yaxes(title="POV")
    return _base(fig, height=230)


def make_modality_figure(results: list[RetrievalResult]) -> go.Figure:
    if not results:
        return _empty()
    counts = Counter(r.modality for r in results)
    ordered = sorted(counts.items(), key=lambda p: p[1], reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[name for name, _ in ordered],
            y=[count for _, count in ordered],
            marker_color=ACCENT,
        )
    )
    fig.update_xaxes(title="Modality")
    fig.update_yaxes(title="Results", rangemode="tozero")
    return _base(fig, height=230)
