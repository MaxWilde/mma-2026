from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Callable


class StartupManager:
    """Tracks background dashboard warmup for the browser and terminal."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "not_started",
            "ready": False,
            "stage": "Waiting to start",
            "detail": "",
            "completed": 0,
            "total": 3,
            "percent": 0,
            "elapsed_seconds": 0.0,
            "error": None,
            "viewpoints": ["All"],
            "grounding_status": "not_started",
        }
        self._started_at: float | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._started_at = time.perf_counter()
            self._state.update(
                {
                    "status": "loading",
                    "ready": False,
                    "stage": "Starting background warmup",
                    "detail": "",
                    "completed": 0,
                    "percent": 0,
                    "error": None,
                }
            )
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="castle-core-warmup",
            )
            self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = deepcopy(self._state)
            if self._started_at is not None:
                state["elapsed_seconds"] = round(
                    time.perf_counter() - self._started_at,
                    1,
                )
            return state

    def _run(self) -> None:
        from castle_dashboard import pipeline as pipeline

        stages: list[tuple[str, str, Callable[[], Any]]] = [
            (
                "Loading visual FAISS index",
                "Reading keyframe embeddings and metadata",
                pipeline._load_visual_index,
            ),
            (
                "Loading SigLIP2 text encoder",
                "Preparing visual-query embeddings",
                pipeline._load_siglip_text_model,
            ),
            (
                "Loading transcript retrieval",
                "Dense index, lexical resources and transcript reranker",
                lambda: pipeline.retrieve_transcript("warmup", top_k=1),
            ),
        ]

        try:
            for index, (stage, detail, operation) in enumerate(stages, start=1):
                self._set_stage(stage, detail, index - 1)
                started = time.perf_counter()
                operation()
                elapsed = time.perf_counter() - started
                self._complete_stage(index, stage, elapsed)

            viewpoints = ["All", *pipeline.get_available_viewpoints()]
            with self._lock:
                self._state.update(
                    {
                        "status": "ready",
                        "ready": True,
                        "stage": "Dashboard ready",
                        "detail": (
                            "Search is available. QA, keyword resources and "
                            "GroundingDINO continue warming in the background."
                        ),
                        "completed": len(stages),
                        "percent": 100,
                        "viewpoints": viewpoints,
                        "grounding_status": "loading",
                    }
                )
            print(
                f"[startup] [100%] Dashboard ready in {self.snapshot()['elapsed_seconds']:.1f}s",
                flush=True,
            )
            threading.Thread(
                target=self._warm_grounding,
                args=(pipeline,),
                daemon=True,
                name="dino-warmup",
            ).start()
            threading.Thread(
                target=self._warm_optional_resources,
                args=(pipeline,),
                daemon=True,
                name="optional-model-warmup",
            ).start()
        except Exception as exc:
            with self._lock:
                self._state.update(
                    {
                        "status": "failed",
                        "ready": False,
                        "stage": "Startup failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            print(
                f"[startup] FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _set_stage(self, stage: str, detail: str, completed: int) -> None:
        with self._lock:
            self._state.update(
                {
                    "stage": stage,
                    "detail": detail,
                    "completed": completed,
                    "percent": round(100 * completed / self._state["total"]),
                }
            )
        print(
            f"[startup] [{completed}/{self._state['total']}] {stage} — {detail}",
            flush=True,
        )

    def _complete_stage(self, completed: int, stage: str, elapsed: float) -> None:
        percent = round(100 * completed / self._state["total"])
        with self._lock:
            self._state.update(
                {
                    "completed": completed,
                    "percent": percent,
                    "detail": f"Completed in {elapsed:.1f}s",
                }
            )
        width = 24
        filled = round(width * completed / self._state["total"])
        bar = "█" * filled + "░" * (width - filled)
        print(
            f"[startup] [{bar}] {percent:3d}% {stage} completed in {elapsed:.1f}s",
            flush=True,
        )

    def _warm_grounding(self, pipeline: Any) -> None:
        try:
            pipeline.warmup_grounding_dino()
            with self._lock:
                self._state["grounding_status"] = "ready"
        except Exception as exc:
            with self._lock:
                self._state["grounding_status"] = f"unavailable: {exc}"

    def _warm_optional_resources(self, pipeline: Any) -> None:
        """Warm non-search features after the dashboard becomes usable."""
        try:
            pipeline._load_minilm_model()
            pipeline.add_transcript_heatmap(
                "warmup",
                {"text": "warmup", "evidence_type": "transcript"},
            )
        except Exception as exc:
            print(
                f"[startup] optional model warmup unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )


startup_manager = StartupManager()
