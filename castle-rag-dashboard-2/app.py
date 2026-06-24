from castle_dashboard.app import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    import threading
    import time
    from castle_dashboard import pipeline as _pl

    print("[startup] Loading FAISS indexes, SigLIP2 text model, and transcript retrieval resources…", flush=True)
    t0 = time.perf_counter()
    _pl._load_visual_index()
    _pl._load_siglip_text_model()
    # Warms transcript_retrieval's internal dense/lexical/cross-encoder caches
    # (and answer_span_highlight's QA + MiniLM caches) with a throwaway query,
    # so the first real search from the UI isn't the one paying the load cost.
    _pl.retrieve_transcript("warmup", top_k=1)
    _pl.add_transcript_heatmap("warmup", {"text": "warmup", "evidence_type": "transcript"})
    print(f"[startup] Core models ready in {time.perf_counter() - t0:.1f}s — serving on :13209", flush=True)

    # Pre-load MiniLM (keyword ranking) and GroundingDINO in background daemon
    # threads so neither delays the HTTP server before their first use.
    threading.Thread(target=_pl.warmup_minilm, daemon=True, name="minilm-warmup").start()
    threading.Thread(target=_pl.warmup_grounding_dino, daemon=True, name="dino-warmup").start()

    app.run(debug=False, host="127.0.0.1", port=13209)
