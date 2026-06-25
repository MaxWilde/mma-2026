import time as _time
_t_script_start = _time.perf_counter()
print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] app.py: starting Python interpreter", flush=True)

print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] app.py: importing castle_dashboard.app …", flush=True)
from castle_dashboard.app import create_app
print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] app.py: import done — calling create_app() …", flush=True)

app = create_app(_t_script_start)
server = app.server
print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] app.py: create_app() done", flush=True)

if __name__ == "__main__":
    import threading
    from castle_dashboard import pipeline as _pl

    print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] [startup] Loading FAISS indexes, SigLIP2 text model, and transcript retrieval resources…", flush=True)
    t0 = _time.perf_counter()

    print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] [startup] _load_visual_index …", flush=True)
    _pl._load_visual_index()
    print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] [startup] _load_transcript_index …", flush=True)
    _pl._load_transcript_index()
    print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] [startup] _load_siglip_text_model …", flush=True)
    _pl._load_siglip_text_model()
    print(f"[t={_time.perf_counter()-_t_script_start:.2f}s] [startup] Core search ready in {_time.perf_counter() - t0:.1f}s — serving on :13209", flush=True)

    # Transcript retrieval pulls in sentence_transformers on first use — a heavy,
    # cold import (~3–4 min on a fresh compute node) that used to block serving.
    # Warm it in the background instead (like minilm/dino below) so the dashboard
    # is reachable immediately; the first transcript query waits only if it
    # arrives before this finishes.
    def _warmup_transcript() -> None:
        _t = _time.perf_counter()
        _pl.retrieve_transcript("warmup", top_k=1)
        _pl.add_transcript_heatmap("warmup", {"text": "warmup", "evidence_type": "transcript"})
        print(f"[startup] transcript retrieval warm in {_time.perf_counter()-_t:.1f}s", flush=True)

    threading.Thread(target=_warmup_transcript, daemon=True, name="transcript-warmup").start()
    threading.Thread(target=_pl.warmup_minilm, daemon=True, name="minilm-warmup").start()
    threading.Thread(target=_pl.warmup_grounding_dino, daemon=True, name="dino-warmup").start()

    app.run(debug=False, host="127.0.0.1", port=13209)
