# CASTLE RAG Dashboard — Technical Implementation Report

## 1. Project Overview

The CASTLE RAG Dashboard is an interactive multimedia evidence-retrieval system built for the **CASTLE 2024** multi-camera video archive. It lets a user type a natural-language query and surface timestamped video evidence from two complementary modalities: raw video keyframes (visual retrieval via SigLIP2 + FAISS) and speech-to-text transcripts (transcript retrieval via MiniLM + BM25 + cross-encoder). The system runs as a single-process Python web application on a GPU-equipped compute node of the **Snellius HPC cluster** (SURF), accessible via SSH port-forwarding.

**Entry-point:** `castle-rag-dashboard-2/app.py`
**Serving port:** `127.0.0.1:13209`
**Python venv:** `/gpfs/scratch1/shared/group_h/data_goncalo/.venv`

---

## 2. Repository Layout

```
multimedia-rag/
├── castle-rag-dashboard-2/          # Dashboard application
│   ├── app.py                       # Process entry-point
│   └── castle_dashboard/
│       ├── app.py                   # Dash factory + Flask image routes
│       ├── pipeline.py              # In-process retrieval adapter
│       ├── callbacks/
│       │   └── dashboard_callbacks.py  # All Dash callbacks (event handlers)
│       ├── components/              # UI component builders
│       │   ├── layout.py            # Top-level page skeleton
│       │   ├── search_panel.py      # Sidebar: query input + filters
│       │   ├── result_list.py       # Ranked result cards
│       │   ├── evidence_viewer.py   # Evidence detail + media panel
│       │   ├── charts.py            # Chart card wrapper
│       │   ├── router_banner.py     # Router decision banner
│       │   └── metrics.py           # Index stats footer
│       ├── services/
│       │   └── dashboard_service.py # Stateful service layer + result cache
│       ├── models/
│       │   └── schemas.py           # Dataclasses: RetrievalResult, SearchFilters, …
│       └── utils/
│           ├── figures.py           # Plotly chart builders
│           └── formatting.py        # Display helpers
├── src/                             # Backend retrieval library
│   ├── retriever.py                 # FAISS I/O + generic dense retrieval
│   ├── clip_retrieval.py            # SigLIP2/CLIP model loading + embedding
│   ├── transcript_retrieval.py      # Dense+BM25+RRF+rerank transcript pipeline
│   ├── evidence_router.py           # Heuristic modality router
│   ├── mixed_evidence_ranker.py     # Cross-modal score fusion + deduplication
│   ├── visual_grounding.py          # GroundingDINO object localization
│   ├── answer_span_highlight.py     # Extractive QA (RoBERTa/DistilBERT) + heatmap
│   ├── evidence_links.py            # YouTube timestamp URL generation
│   ├── dataset_loader.py            # CASTLE dataset I/O
│   └── vqa.py                       # Timestamp formatting + VQA utilities
└── scripts/
    ├── query_evidence.py            # CLI retrieval script (mirrored by pipeline.py)
    ├── build_siglip_image_index.py  # Index builder for visual FAISS
    ├── build_transcript_index.py    # Index builder for transcript FAISS
    └── …                            # Evaluation and analysis scripts
```

---

## 3. Startup Sequence

`app.py` is the OS-level entry point. On startup it:

1. Calls `create_app()` (defined in `castle_dashboard/app.py`) which instantiates the Dash/Flask app, wires the layout and callbacks, and registers two custom Flask routes for image serving.
2. **Eagerly** loads (on the main thread, blocking):
   - The visual FAISS index + metadata → `_load_visual_index()` (cached with `@lru_cache(maxsize=1)`)
   - The SigLIP2 text-only tower → `_load_siglip_text_model()`
   - The transcript dense/lexical resources + cross-encoder → warmed by a throwaway `retrieve_transcript("warmup", top_k=1)` call
3. **Lazily in a daemon thread** → loads GroundingDINO into GPU memory via `warmup_grounding_dino()`, so the first "Localize" click doesn't pay the cold-load penalty.
4. Calls `app.run(debug=False, host="127.0.0.1", port=13209)`.

This design ensures the HTTP server is up and responding in < 1 s while the GroundingDINO GPU transfer (~10–60 s) happens in the background.

---

## 4. sys.path and Module Resolution

`castle_dashboard/app.py` explicitly manages `sys.path` to ensure the **live** backend repo (`multimedia-rag/src/`, `multimedia-rag/scripts/`) takes priority. The two roots inserted are:

- `_MMA_ROOT` = `multimedia-rag/` (the git repo)
- `_BACKEND_ROOT` = `$CASTLE_BACKEND_ROOT` or `/gpfs/scratch1/shared/group_h/data_goncalo`

Both are inserted at index 0 (with any existing entry removed first) so the live `src.*` namespace always shadows anything else. This allows `pipeline.py` to do `from src.retriever import …` unambiguously.

---

## 5. Front-End Architecture (Dash / React)

### 5.1 Framework

The UI is built with **Plotly Dash 4.2**, which compiles a React SPA from Python component descriptors. There is no hand-written JavaScript. The Dash server-side rendering model means every interactive element is described in Python and re-rendered by callbacks.

### 5.2 Page Layout (`components/layout.py`)

The page shell (`html.Div(className="app-shell")`) contains:

| Zone | Component | Purpose |
|------|-----------|---------|
| Sidebar | `build_search_panel()` | Query input, modality/viewpoint/score filters, search & refine buttons |
| Main > top | `router-banner` | Evidence router decision badge |
| Main > content-grid left | `evidence-panel` | Selected result's keyframe / transcript / grounding view |
| Main > content-grid right | `ranked-results` | Scrollable ranked result cards |
| Main > chart-grid | Three chart cards | Score bar chart, timeline scatter, modality donut |
| Footer | `metrics-panel` | Index stats (keyframes indexed, chunks, latency) |

**Hidden `dcc.Store` elements** act as the client-side state bus:

| Store ID | Content |
|----------|---------|
| `filtered-result-ids` | Ordered list of `RetrievalResult.id` for the current search |
| `selected-result-id` | ID of the currently highlighted result |
| `grounding-result` | GroundingDINO JSON output (bounding box + confidence) |
| `transcript-evidence` | QA answer span + heatmap token scores |
| `feedback-store` | `{liked: [...], disliked: [...]}` — relevance feedback state |
| `query-embedding-store` | L2-normalised SigLIP2 query vector (for Rocchio) |

### 5.3 Search Panel (`components/search_panel.py`)

- `dcc.Textarea` for the natural-language query
- `dcc.Checklist` for modality filter: `["transcript", "visual"]`
- `dcc.RangeSlider` for confidence range [0, 1], step 0.01
- `dcc.Dropdown` for viewpoint (populated dynamically from `dashboard_service.get_available_viewpoints()`, which reads unique `source_name` values from the visual FAISS metadata)
- "Search" button (primary), "Refine with feedback" button (secondary, disabled until feedback exists + embedding is stored)

### 5.4 Result Cards (`components/result_list.py`)

Each result card is a `html.Button` with a **pattern-matching id** `{"type": "result-card", "index": result.id}`. This lets the callback system fire on any card click without registering one callback per card. Cards show:
- Rank pill, confidence score pill, "Router pick" badge (if `is_routed_choice`)
- Title (transcript snippet or `source/video@timestamp`), viewpoint, caption preview
- Thumbs-up / thumbs-down buttons (also pattern-matched IDs) for relevance feedback

### 5.5 Evidence Viewer (`components/evidence_viewer.py`)

Adapts its content based on `result.modality`:

**Visual results:**
- Keyframe image served via `/keyframe?path=<absolute-path>` Flask route
- When GroundingDINO has run: swaps to the annotated bounding-box image from `/grounding/<relpath>`
- "View original frame" link when grounding image is active
- Collapsible `<details>` element showing ±2 min transcript context chunks, or (if heatmap was computed) the token-level heatmap with amber color intensity proportional to relevance score

**Transcript results:**
- YouTube embed iframe (`https://www.youtube.com/embed/{video_id}?start={t}`) with start time set to `result.start_sec`
- Transcript context rendering — or the heatmap block when "Highlight answer (QA)" was clicked

**Action buttons:**
- "Localize (GroundingDINO)" — only enabled for visual results
- "Highlight answer (QA)" — only enabled for transcript results

**Heatmap rendering** (`_heatmap_block`): each token is wrapped in `html.Span` with `backgroundColor: rgba(245, 158, 11, α)` where α = `0.12 + 0.55 * score`. The best extractive QA span is displayed separately with its sigmoid-normalised confidence, plus up to 5 alternative candidate spans.

### 5.6 Plotly Charts (`utils/figures.py`)

Three charts share a common `_base()` style (white template, no background, Inter font):

| Chart | Type | X / Y | Interactivity |
|-------|------|--------|---------------|
| Score chart | Horizontal `go.Bar` | Confidence % / result title | Click fires `select_result` callback via `customdata=result.id` |
| Timeline chart | `go.Scatter` (markers) | Video hour / viewpoint (POV) | Click fires `select_result` |
| Modality chart | Vertical `go.Bar` | Modality name / count | Informational only |

The selected result is highlighted in `PRIMARY` blue; others in `MUTED` gray (score chart) or `ACCENT` teal (timeline).

### 5.7 Image Serving Routes

Two Flask routes are registered directly on the Dash/Flask `app.server`:

- **`GET /keyframe?path=<abs_path>`** — validates that the path starts with an allowed prefix (`/gpfs/scratch1/shared/group_h/data_goncalo/day1/` or the symlink-resolved path), then `send_file`. This is a security control: the path allowlist prevents directory traversal.
- **`GET /grounding/<relpath>`** — serves annotated bounding-box images from `_BACKEND_ROOT / relpath`, generated by GroundingDINO and written to `artifacts/visual_grounding/`.

---

## 6. Callback Architecture (`callbacks/dashboard_callbacks.py`)

Dash callbacks are registered via `register_callbacks(app)`. There are **8 callbacks**:

### Callback 1 — `run_search`
**Trigger:** "Search" button click
**Outputs:** `filtered-result-ids`, `query-embedding-store`, `feedback-store`, `search-error`
**Flow:**
1. Builds `SearchFilters` dataclass from UI state
2. Calls `dashboard_service.search(filters)` (see §8)
3. Calls `pipeline.get_query_embedding(query)` — encodes query with SigLIP2 text tower and stores the raw embedding vector in `dcc.Store` for later Rocchio use
4. Returns ordered list of result IDs, resets feedback

### Callback 2 — `update_keyword_suggestions`
**Trigger:** `filtered-result-ids` changes (i.e., after search completes)
**Output:** `keyword-suggestions` div
Calls `dashboard_service.compute_keyword_suggestions()`, which delegates to `src/transcript_keyword_recommender.py`. Renders clickable `html.Button` chips with pattern-matched IDs `{"type": "keyword-chip", "term": term}`.

### Callback 3 — `append_keyword_to_query`
**Trigger:** Any keyword chip click (pattern-matched `ALL`)
**Output:** `query-input` value
Appends the clicked keyword to the query text box (only if not already present).

### Callback 4 — `select_result`
**Trigger:** Any result card click OR chart click
**Output:** `selected-result-id`
Uses `callback_context.triggered_id` to determine source. For chart clicks, extracts `result_id` from `click_data["points"][0]["customdata"]`.

### Callback 5 — `toggle_feedback`
**Trigger:** Any thumbs-up or thumbs-down click (pattern-matched `ALL`)
**Output:** `feedback-store`
Maintains a mutual-exclusion invariant: a result can be liked or disliked, not both. Toggling the same direction removes it (toggle-off).

### Callback 6 — `update_refine_controls`
**Trigger:** `feedback-store` or `query-embedding-store` changes
**Output:** `refine-button.disabled`, `feedback-count` label
Enables the Refine button only when there is at least one piece of feedback AND an embedding is available.

### Callback 7 — `refine_search` (Rocchio)
**Trigger:** "Refine with feedback" button click
**Outputs:** `filtered-result-ids`, `feedback-store`
1. Translates liked/disliked result IDs → FAISS row indices (via `result.faiss_row_id`)
2. Calls `pipeline.rocchio_refine(embedding, liked_rows, disliked_rows)` to compute a refined query vector
3. Calls `pipeline.retrieve_visual_from_embedding(refined_embedding)` for new visual results
4. Re-routes and re-ranks with cached transcript results (no re-embedding)
5. Resets feedback to empty

### Callback 8a — `run_grounding`
**Trigger:** "Localize (GroundingDINO)" button click
**Output:** `grounding-result`
Packages the selected result as a `chunk` dict and calls `pipeline.run_grounding(query, chunk)`.

### Callback 8b — `run_transcript_evidence`
**Trigger:** "Highlight answer (QA)" button click
**Output:** `transcript-evidence`
Calls `dashboard_service.compute_transcript_evidence(result_id)`, which in turn calls `pipeline.add_transcript_heatmap(query, item)`.

### Callback 9a — `update_result_feedback`
**Trigger:** `feedback-store` changes (thumb clicks only)
**Output:** `ranked-results` (partial re-render)
Separate from the main dashboard update so thumb-state changes don't wastefully re-render charts or the evidence panel.

### Callback 9b — `update_dashboard`
**Trigger:** `filtered-result-ids`, `selected-result-id`, `grounding-result`, or `transcript-evidence` changes
**Outputs:** All 7 main display panels simultaneously
The "god callback" — re-renders everything. Fetches results by IDs from cache, fetches selected result, computes transcript context, and distributes to all component builders.

---

## 7. Backend Pipeline (`castle_dashboard/pipeline.py`)

This module is the **adapter layer** between the dashboard and the raw `src.*` backend. All models and indexes are kept **resident in memory** via `@lru_cache(maxsize=1)` — no per-request disk I/O after startup.

### Key functions:

**`retrieve_visual(question, top_k=20)`**
1. Calls `query_variants(question)` to generate noun-phrase and prompt-template variants of the query (from `scripts/query_evidence.py`)
2. Embeds all variants with `embed_texts_clip_profile` (SigLIP2 text tower)
3. Searches the FAISS index with `search_k = top_k × CANDIDATE_MULTIPLIER` (default 5×)
4. Calls `collect_variant_results` + `merge_variant_results` to deduplicate and diversify across a 30-second temporal window (`DIVERSITY_WINDOW_SEC`)
5. Enriches each result with `evidence_type="visual"`, formatted timestamp, and YouTube URL
6. Adds `faiss_row_id` (positional index in FAISS) for later Rocchio use

**`retrieve_transcript(question, top_k=20)`**
Delegates entirely to `src.transcript_retrieval.retrieve_transcript_evidence` (see §9).

**`route(question, visual_results, transcript_results)`**
Delegates to `src.evidence_router.route_evidence` (see §10).

**`get_mixed_evidence(question, visual_results, transcript_results, router_debug, top_k=20)`**
Delegates to `src.mixed_evidence_ranker.build_mixed_evidence_list` (see §11).

**`rocchio_refine(query_embedding, liked_row_ids, disliked_row_ids)`**
Implements the classic Rocchio formula:

```
q_new = α·q_orig + β·mean(liked_vectors) − γ·mean(disliked_vectors)
```

Default weights: α=1.0, β=0.15, γ=0.05. Liked/disliked vectors are fetched from the FAISS index using `index.reconstruct(row_id)`. Result is L2-normalised before returning.

**`retrieve_visual_from_embedding(query_embedding, top_k=20)`**
Takes a pre-computed embedding (e.g., Rocchio-refined), searches FAISS directly, applies the same diversity window filtering as `retrieve_visual`.

**`run_grounding(query, chunk)`**
Calls `src.visual_grounding.ground_visual_evidence` with a `GroundingConfig()`. Converts the `VisualGroundingResult` dataclass to a JSON-serialisable dict, computing the relative path from `_BASE` for the output image (served via `/grounding/<relpath>`).

**`add_transcript_heatmap(question, item)`**
Delegates to `scripts/query_evidence.add_transcript_heatmap`, which calls both `answer_span_highlight.find_answer_span` and `src.transcript_heatmap.compute_transcript_heatmap`.

**Path remapping (`_fix_path`, `_fix_item_paths`)**
The FAISS metadata was built on the old `/scratch-shared/` mount point. All absolute paths in metadata items are rewritten to `/gpfs/scratch1/shared/` on load to handle the mount rename.

---

## 8. Service Layer (`services/dashboard_service.py`)

`DashboardService` is a **singleton** (`dashboard_service = DashboardService()`) holding:
- `_cache: dict[str, RetrievalResult]` — result objects keyed by ID
- `_raw_cache: dict[str, dict]` — raw pipeline dicts (needed for on-demand QA enrichment)
- `_last_transcript_results` — saved for Rocchio refine (avoids re-running transcript retrieval)
- `_last_query`, `_last_router_debug`, `_last_query_ms`

**`search(filters)`:**
1. Calls both `retrieve_visual` and `retrieve_transcript` unconditionally (the router needs both top scores even if the UI is filtered to one modality)
2. Calls `route` → gets the router's single "best answer" choice + `router_debug`
3. Calls `get_mixed_evidence` → merged ranked list
4. Applies UI filters in Python: modality filter, confidence range filter, viewpoint filter
5. Converts raw dicts → `RetrievalResult` dataclasses via `_visual_to_result` / `_transcript_to_result`
6. Sets `is_routed_choice=True` on whichever result matches the router's pick (`_evidence_key` comparison)
7. Stores results in `_cache` + `_raw_cache`

**`refine_visual(filters, refined_visual)`:**
Re-runs routing and mixing with Rocchio-refined visual results but **reuses cached transcript results** — so only the visual side is refreshed.

---

## 9. Transcript Retrieval (`src/transcript_retrieval.py`)

This is a multi-stage pipeline operating entirely in Python on the loaded FAISS transcript index (41,769 chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`):

### Stage 1 — Dense retrieval (MiniLM)
`dense_candidates(question, index, metadata, model, dense_k=100)`
- Encodes the query with MiniLM (`encode`, normalise)
- Searches the FAISS IndexFlatIP for top 100 candidates by cosine similarity

### Stage 2 — Lexical retrieval (BM25)
`lexical_candidates(question, index_dir, lexical_k=100)`
- Loads lexical resources from `@lru_cache`: builds sliding-window passages (80 tokens, stride 40) from all transcript text, computes per-window token counts and corpus-level IDF
- Scores each passage using BM25 (k1=1.2, b=0.75), takes best window per chunk
- Returns top 100 by BM25 score

### Stage 3 — Reciprocal Rank Fusion (RRF)
`rrf_fuse(dense, lexical, rrf_k=60)`
- Computes RRF score = Σ 1/(60 + rank) across both ranked lists
- Merges into a single deduplicated list ranked by RRF score

### Stage 4 — Reranking
If `cross-encoder/ms-marco-MiniLM-L-6-v2` is found locally:
- `cross_encoder_rerank`: scores each (question, passage) pair with the cross-encoder; replaces RRF score with cross-encoder logit

If cross-encoder is unavailable:
- `minilm_passage_rerank`: fallback — encodes question + passages with MiniLM, scores by dot product, re-sorts

### Stage 5 — Source name boost
`apply_source_name_boost`: if any CASTLE camera name appears literally in the query text (tokenised), boost matching results by +0.25.

### Optional — Timestamp refinement
`refine_transcript_timestamp`: loads the raw JSON transcript file for a result, builds short windows (≤30 s) inside the broad FAISS chunk boundary, re-ranks them by a weighted combination of query BM25 score + broad-text BM25 score + content density. Replaces the broad timestamp with a more precise one.

### Optional — Playback alignment
`align_playback_start_across_povs`: finds the same spoken content in a different camera's transcript file (same day + hour_id), and outputs a `playback_youtube_timestamp_url` pointing to that alternate POV if it scores better than the original.

---

## 10. Evidence Router (`src/evidence_router.py`)

Receives the top-1 visual result and top-1 transcript result, returns one as the "chosen" evidence item.

**Scoring formula:**
```
route_score(mode) = heuristic_score(question, mode) + normalize(raw_retrieval_score, mode)
```

**Heuristic scores** are based on keyword matching:
- Visual cues: `color`, `where`, `visible`, `wearing`, `holding`, spatial prepositions (+1.2 each)
- Visual regex bonuses: "where is/are", "what color", "what is/are on/in/near" (+2.0 each)
- Transcript cues: `say`, `mention`, `talk`, `why`, `how long`, `what did` (+1.2 or +2.0–2.5)

**Override rule:** If `visual_heuristic == 0.0` AND `raw_transcript_score - raw_visual_score ≥ 0.25`, transcript wins unconditionally.

**Router confidence** is derived as `margin / (margin + 1.0)` (a bounded sigmoid on the score gap, reported as a display-only % — not a calibrated probability).

The full `router_debug` dict (heuristic scores, raw scores, combined scores, margin, confidence, reason string) is stored in the service layer and shown in the `router_banner` component.

---

## 11. Mixed Evidence Ranker (`src/mixed_evidence_ranker.py`)

Takes visual + transcript candidates and fuses them onto a single 0–1 confidence scale.

### Step 1 — Diversity filtering (before scoring)
- `diverse_visual_candidates`: greedily removes keyframes within 30 s of an already-selected one from the same source/video
- `diverse_transcript_candidates`: greedily removes transcript chunks with temporal IoU > 0.7 from the same source/video

### Step 2 — Channel weights (softmax)
```
visual_logit = combined_visual_score / temperature (1.5)
transcript_logit = combined_transcript_score / temperature
visual_weight, transcript_weight = softmax(visual_logit, transcript_logit)
```

### Step 3 — Scoring each candidate
```
quality_component = 0.75 × calibrated_quality_score
prior_component   = 0.25 × channel_weight
diversity_penalty = item-level penalty from pre-filtering
final_score = (quality_component + prior_component) × (1 - diversity_penalty)
```

`calibrated_quality_score` in `"max"` mode = raw score / max score in channel. In `"percentile"` mode = linear rank percentile.

### Step 4 — Cross-modal duplicate suppression
After sorting the merged list, `suppress_event_duplicates` removes items that are "the same event" as an already-kept result:
- **Same-modality transcript:** temporal IoU > 0.5 on same day/hour, or Jaccard similarity of token sets > 0.75
- **Same-modality visual:** within 30 s on same source, or within 10 s across POVs on same day/hour
- **Cross-modal:** within 3 s on same source (conservative)

### Step 5 — Confidence assignment
Each item gets `confidence = min(final_score, 1.0)`. This is the value displayed in the UI (score pill, score chart).

---

## 12. Visual Grounding (`src/visual_grounding.py`)

On-demand, triggered by clicking "Localize (GroundingDINO)".

**Model:** `grounding-dino-base` (Hugging Face `AutoModelForZeroShotObjectDetection`), loaded with `@lru_cache(maxsize=1)` — GPU-resident after warmup.

**Grounding target derivation (`derive_grounding_plan`):**
1. Try regex patterns for location queries ("where is X"), attribute queries ("what color is X"), relation queries ("what is on/near X")
2. Extract noun phrases from the question and caption
3. Prefer caption-confirmed phrases (phrases that appear in both the question and the keyframe's visual caption)
4. Fall back to generic "person" or "object"

**Detection (`detect_with_grounding_dino`):**
For each prompt in priority order, runs GroundingDINO at decreasing confidence thresholds `(0.25, 0.15, 0.10, 0.05)` until at least one bounding box is returned. The first successful detection is returned.

**Post-processing:** Uses `processor.post_process_grounded_object_detection` with introspection to handle both old (`threshold`) and new (`box_threshold`) API signatures.

**Output image:** Draws a red bounding box + label with confidence using Pillow's `ImageDraw`. Output is saved to `artifacts/visual_grounding/<stem>_<sha1>_bbox.jpg` and served via `/grounding/<relpath>`.

**Optional SigLIP crop-reranking** (`ground_visual_evidence_with_rerank`): for multi-frame grounding, crops each detected box, re-embeds crops with SigLIP, and scores as `α·frame_score + β·dino_score + γ·siglip_crop_score`.

---

## 13. Extractive QA + Transcript Heatmap (`src/answer_span_highlight.py`)

Triggered by "Highlight answer (QA)".

**Model:** `deepset/roberta-base-squad2` (primary, tried first) or DistilBERT SQuAD fallback. Loaded with `@lru_cache(maxsize=1)`.

**Context selection:**
Long transcripts are split into overlapping 60-word windows (stride 30). The best window is selected by MiniLM embedding similarity to the query (optionally combined 65/35 with an anchor text similarity). Adjacent windows are merged.

**Span extraction (`run_manual_qa`):**
The tokenizer is called with `return_offsets_mapping=True`. Start/end logit pairs for all context tokens are enumerated (max 20 tokens apart). Top candidates are deduplicated by normalised text. The raw logit sum is passed through a sigmoid for display confidence.

**Heatmap:** Separate from answer-span QA — computed by `src/transcript_heatmap.py` which assigns per-token relevance scores. In the UI each token is rendered as `html.Span` with amber background intensity `rgba(245, 158, 11, 0.12 + 0.55 × score)`.

---

## 14. Relevance Feedback and Rocchio Refinement

### Collection
Each result card has thumbs-up / thumbs-down buttons. Clicking stores the result's ID in `feedback-store.liked` or `feedback-store.disliked` (mutually exclusive, toggle-off supported).

### Embedding storage
After each search, the SigLIP query embedding (shape `[1, D]`, L2-normalised) is stored in `query-embedding-store` as `list[list[float]]`.

### Refinement
When "Refine with feedback" is clicked:
1. Liked/disliked result IDs → FAISS row indices via `result.faiss_row_id`
2. Liked/disliked vectors fetched from FAISS with `index.reconstruct(i)` — no disk I/O
3. Rocchio update:
   `q_new = 1.0·q_orig + 0.15·mean(liked) − 0.05·mean(disliked)`
   Result is L2-normalised
4. Direct FAISS search with the refined vector
5. New visual results → re-route + re-mix with **cached** transcript results
6. Feedback store reset to empty

---

## 15. Data Layer

### Visual Index
- Format: FAISS `IndexFlatIP` (inner product = cosine similarity on L2-normalised vectors)
- Size: 83,718 keyframe embeddings
- Dimension: 1152 (SigLIP2 `so400m-patch16-512` text embedding dimension)
- Metadata: JSON sidecar with fields: `keyframe_path`, `source_name`, `video_id`, `day`, `start_sec`, `end_sec`, `keyframe_time_sec`, `visual_caption`, `youtube_url`, `source_id`

### Transcript Index
- Format: FAISS `IndexFlatIP`
- Size: 41,769 chunks
- Embedding: `all-MiniLM-L6-v2` (384-dim)
- Metadata: JSON sidecar with fields: `text`, `source_name`, `video_id`, `hour_id`, `start_sec`, `end_sec`, `transcript_path`, `youtube_url`, `source_id`

### Index file layout (per index directory):
```
transcript.faiss    ← FAISS binary index
metadata.json       ← {"model_name": "...", "items": [...]}
```

---

## 16. Key Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| FAISS `IndexFlatIP` (flat) | Exact nearest-neighbour — no recall degradation vs approximate methods. Feasible for 83k vectors. |
| `@lru_cache(maxsize=1)` for models | Models are process-global. Flask/Dash is single-process, so this is safe and avoids repeated disk I/O. |
| Always retrieve both modalities | The router and ranker need both channels' top scores to compute weights — filtering happens after mixing. |
| RRF for transcript fusion | Score spaces of dense (cosine similarity) and BM25 are incomparable. RRF uses rank positions, avoiding calibration. |
| Rocchio β=0.15, γ=0.05 | Conservative weights — feedback should steer, not override the query. Weak feedback signal from small sample. |
| `suppress_callback_exceptions=True` | Pattern-matched IDs (result cards, chips) don't exist on page load; Dash raises without this flag. |
| Separate `update_result_feedback` callback | Thumb clicks should not re-run charts or re-fetch the evidence panel — only the result list needs updating. |
| Daemon thread for GroundingDINO warmup | Keeps HTTP server startup fast. If the warmup fails (no GPU, model missing), it logs and silently skips without crashing. |
| Path prefix allowlist on `/keyframe` route | Prevents arbitrary filesystem reads via URL manipulation — only paths under the known data directories are served. |
| Playback alignment via `align_playback_start_across_povs` | The retrieved transcript chunk may be from a secondary POV. The system finds the same spoken moment in a different camera's transcript and provides an alternate YouTube link. |

---

## 17. Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Web framework | Plotly Dash 4.2 (Flask under the hood) |
| Charts | Plotly `graph_objects` |
| Visual retrieval | SigLIP2 `so400m-patch16-512` (HuggingFace `transformers`) + FAISS IndexFlatIP |
| Transcript retrieval | `all-MiniLM-L6-v2` (sentence-transformers) + BM25 + RRF + cross-encoder |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Object localization | `grounding-dino-base` (HuggingFace `transformers`) |
| Extractive QA | `deepset/roberta-base-squad2` or DistilBERT SQuAD |
| Relevance feedback | Rocchio algorithm on FAISS-resident vectors |
| Image manipulation | Pillow |
| Numerics | NumPy |
| Infrastructure | Snellius HPC (SURF), SLURM, SSH port-forwarding, GPU compute node |
