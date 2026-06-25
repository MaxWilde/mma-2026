from __future__ import annotations

import os
import sys
import time as _time
from pathlib import Path

_t0_module = _time.perf_counter()
print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: module start (sys.path setup)", flush=True)

# Ensure the live backend repo (src.*, scripts.*) is importable, taking
# priority over multimedia-rag/, which only holds a stale, pre-backend-update
# copy of src/. Without this order, `import src.transcript_retrieval` (and
# other new modules) would silently resolve to the outdated copy or fail.
_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent   # castle-rag-dashboard-2/
_MMA_ROOT = _DASHBOARD_ROOT.parent                         # multimedia-rag/
_BACKEND_ROOT = Path(
    os.getenv("CASTLE_BACKEND_ROOT", "/gpfs/scratch1/shared/group_h/data_goncalo")
)
for _root in (_MMA_ROOT, _BACKEND_ROOT):  # inserted in reverse priority order
    _root_str = str(_root)
    if _root_str in sys.path:
        sys.path.remove(_root_str)
    sys.path.insert(0, _root_str)

print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: importing dash …", flush=True)
from dash import Dash
print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: importing dotenv …", flush=True)
from dotenv import load_dotenv
print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: importing callbacks …", flush=True)
from castle_dashboard.callbacks.dashboard_callbacks import register_callbacks
print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: importing layout …", flush=True)
from castle_dashboard.components.layout import build_layout
print(f"[t_mod={_time.perf_counter()-_t0_module:.2f}s] castle_dashboard.app: all imports done", flush=True)


def create_app(t_script_start: float = 0.0) -> Dash:
    def _log(msg: str) -> None:
        elapsed = _time.perf_counter() - t_script_start if t_script_start else _time.perf_counter() - _t0_module
        print(f"[t={elapsed:.2f}s] create_app: {msg}", flush=True)

    _log("load_dotenv …")
    load_dotenv()
    _log("Dash() init …")
    app = Dash(
        __name__,
        title="CASTLE RAG Dashboard",
        suppress_callback_exceptions=True,
        assets_folder=str(_DASHBOARD_ROOT / "assets"),
    )
    _log("build_layout() …")
    app.layout = build_layout()
    _log("register_callbacks() …")
    register_callbacks(app)
    _log("_register_image_routes() …")
    _register_image_routes(app)
    _log("done")
    return app


def _register_image_routes(app: Dash) -> None:
    """Flask routes for serving keyframe and grounding images from absolute paths."""
    from flask import abort, request, send_file

    _ALLOWED_PREFIXES = (
        "/gpfs/scratch1/shared/group_h/data_goncalo/day1/",
        "/scratch-shared/group_h/data_goncalo/day1/",
    )

    @app.server.route("/keyframe")
    def serve_keyframe():
        path = request.args.get("path", "")
        if not any(path.startswith(p) for p in _ALLOWED_PREFIXES):
            abort(403)
        if not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype="image/jpeg")

    @app.server.route("/grounding/<path:relpath>")
    def serve_grounding(relpath):
        full = _BACKEND_ROOT / relpath
        if not full.is_file():
            abort(404)
        return send_file(str(full), mimetype="image/jpeg")
