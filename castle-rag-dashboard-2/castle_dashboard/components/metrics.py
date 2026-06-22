from dash import html

from castle_dashboard.models.schemas import DashboardStats


def build_metrics(stats: DashboardStats, result_count: int) -> html.Div:
    metrics = [
        ("Visible results", str(result_count)),
        ("Keyframes indexed", f"{stats.total_keyframes:,}"),
        ("Transcript chunks", f"{stats.transcript_chunks:,}"),
        ("Last query", f"{stats.avg_latency_ms} ms" if stats.avg_latency_ms else "—"),
    ]
    return html.Div([_card(label, value) for label, value in metrics], className="metric-grid")


def _card(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Strong(value)], className="metric-card")
