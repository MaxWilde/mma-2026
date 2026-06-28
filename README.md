# CASTLE RAG Dashboard

Interactive multimedia evidence retrieval for the CASTLE multi-view video archive.

This project implements an evidence-first retrieval dashboard for the CASTLE Day 1 recordings. A user asks a natural-language question, and the system returns timestamped visual keyframes and transcript snippets rather than a single generated answer. The goal is to help an analyst inspect, verify, and refine evidence across cameras, time, and modalities.

The repository contains the dashboard, retrieval/indexing code, and evaluation utilities. It does **not** include the full CASTLE media archive, extracted Day 1 keyframes, transcripts, FAISS indexes, or large model weights. To run the system on another machine, prepare equivalent local artifacts and point the app to them through environment variables or `castle-rag-dashboard-2/.env`.

![CASTLE RAG dashboard demo](docs/assets/mma-rag-demo-x5.gif)

## What the dashboard does

The default dashboard combines the final agreed implementation from the team:

- visual retrieval over CASTLE keyframes using SigLIP2 text/image embeddings and FAISS;
- transcript retrieval over ASR transcript chunks using MiniLM dense retrieval, BM25 lexical retrieval, reciprocal-rank fusion, and MiniLM reranking over 50 fused candidates;
- a lightweight evidence router that estimates whether a query is better answered by visual or transcript evidence;
- a mixed ranked list that still shows both modalities instead of hard-switching to only one;
- ranked result cards with modality, viewpoint, timestamp, confidence-like score, and source links;
- an evidence viewer for inspecting keyframes, transcript snippets, surrounding transcript context, and YouTube timestamp links;
- on-demand GroundingDINO localization for selected visual keyframes;
- on-demand transcript answer highlighting for selected transcript evidence;
- relevance feedback with Rocchio-style visual query refinement and feedback-aware routing;
- MiniLM-based recommended keyword chips for query steering;
- optional evaluation logging controlled by an external script.

The default study configuration intentionally uses MiniLM for transcript reranking and keyword suggestions. The heavier BGE reranker and older Qwen-based experiments remain in the repository as optional experiments, but they are not part of the default dashboard path.

## Architecture

At runtime, the system is a single Dash application that keeps the retrieval indexes and models resident in memory. Search and enrichment are separated: the dashboard first retrieves ranked evidence quickly, then only runs expensive enrichment models when the user requests them.

```text
User query
   │
   ▼
Dash frontend
   │  search / filter / select / feedback / localize / highlight
   ▼
DashboardService
   │
   ├── Visual branch
   │     ├── query variants
   │     ├── SigLIP2 text embedding
   │     ├── FAISS search over keyframe embeddings
   │     └── temporal/source deduplication
   │
   ├── Transcript branch
   │     ├── MiniLM dense retrieval
   │     ├── BM25 lexical retrieval
   │     ├── reciprocal-rank fusion
   │     └── MiniLM passage reranking over 50 candidates
   │
   ├── Evidence router
   │     └── combines retrieval scores, query cues, and feedback preference
   │
   ├── Mixed evidence ranker
   │     └── merges visual and transcript candidates into one ranked list
   │
   └── On-demand enrichment
         ├── GroundingDINO bounding boxes for selected visual evidence
         └── DistilBERT-style extractive QA / heatmap for selected transcript evidence
```

The core design principle is that retrieval returns inspectable evidence. The system does not hide uncertainty behind a generated answer; it exposes keyframes, transcript text, timestamps, viewpoints, scores, and source-video links so that the analyst can judge the evidence directly.

## Repository structure

```text
mma-2026/
├── README.md                         # this file
├── requirements.txt                  # default Python environment
├── docs/assets/
│   └── mma-rag-demo-x5.gif           # README demo GIF
├── castle-rag-dashboard-2/
│   ├── app.py                        # dashboard entrypoint
│   ├── requirements.txt              # Python package requirements
│   ├── .env.example                  # main environment variables
│   ├── assets/styles.css             # dashboard styling
│   ├── castle_dashboard/
│   │   ├── app.py                    # Dash app factory + Flask image routes
│   │   ├── callbacks/                # Dash callbacks and clientside behaviour
│   │   ├── components/               # UI components
│   │   ├── models/                   # dataclasses / typed UI schemas
│   │   ├── services/                 # dashboard service, startup, evaluation logger
│   │   └── pipeline.py               # adapter around the backend retrieval modules
│   └── scripts/
│       └── submit_dashboard.job      # Snellius Slurm job for the default dashboard
├── src/
│   ├── clip_retrieval.py             # SigLIP/CLIP model loading and embedding
│   ├── retriever.py                  # FAISS loading and dense retrieval helpers
│   ├── transcript_retrieval.py       # MiniLM + BM25 + RRF + reranking
│   ├── evidence_router.py            # visual/transcript route decision
│   ├── mixed_evidence_ranker.py      # unified visual/transcript ranking
│   ├── query_steering.py             # Rocchio feedback helpers
│   ├── transcript_keyword_recommender.py
│   ├── answer_span_highlight.py      # transcript QA / heatmap support
│   └── visual_grounding.py           # GroundingDINO localization
├── scripts/
│   ├── build_siglip_image_index.py
│   ├── build_transcript_index.py
│   ├── query_evidence.py             # CLI version of the retrieval flow
│   ├── setup_fast_dashboard_home.sh  # optional Snellius package/model cache
│   ├── evaluation_control.py         # optional evaluation-mode controller
│   └── three_part_evaluation.py      # optional three-part study logger helper
└── slurm/
    └── *.sbatch                      # Snellius indexing, caching, testing, and evaluation jobs
```

The `slurm/` jobs and some default paths reflect the course cluster environment. For public use, treat them as examples and override the paths for your own dataset/model locations.

## Installation

Create a Python environment from the repository root:

```bash
git clone <repository-url>
cd mma-2026

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA-enabled machine, install the PyTorch build that matches your CUDA version before installing the remaining requirements. See the official PyTorch installation selector for the correct command for your system.

The nested file `castle-rag-dashboard-2/requirements.txt` is kept for the dashboard package, but the root `requirements.txt` is the recommended starting point for reproducing the default project environment.

## Data preparation

This repository is designed around Day 1 of the CASTLE dataset. The course version used:

- extracted keyframes from Day 1 videos;
- transcript chunks aligned to the corresponding YouTube/source-video timestamps;
- a visual FAISS index over SigLIP2 keyframe embeddings;
- a transcript FAISS/BM25 index over MiniLM transcript embeddings.

These artifacts are not redistributed here. To reproduce the system, you need to create or obtain equivalent artifacts.

### Keyframes

The visual branch expects folders of Day 1 keyframes. In our course setup, these keyframes were already provided as extracted image frames from the CASTLE videos. If starting from the official CASTLE videos or another live copy of the dataset, extract representative keyframes first, for example with FFmpeg or a similar video-processing tool.

The keyframe loader searches for `keyframes/` folders under a dataset root. A typical local layout can look like:

```text
data/
└── day1/
    ├── <viewpoint-or-camera>/<hour-or-video>/
    │   ├── video.mp4                 # optional, used for provenance if available
    │   ├── manifest.json             # optional metadata
    │   └── keyframes/
    │       ├── frame_000001.jpg
    │       └── ...
    └── ...
```

Build a visual index:

```bash
python scripts/build_siglip_image_index.py \
  --dataset-root data/day1 \
  --output-dir artifacts/siglip_index_day1 \
  --model-name google/siglip2-so400m-patch16-512 \
  --allow-download
```

For offline/cluster runs, download or cache the model first and pass the local model path instead of the Hugging Face model name.

### Transcripts

The transcript branch expects JSON files with a `chunks` list. Each chunk should contain text and a timestamp interval:

```json
{
  "chunks": [
    {
      "text": "example transcript text",
      "timestamp": [12.3, 15.8]
    }
  ]
}
```

In the course setup, transcripts were already provided. If starting from the original videos, transcripts can be generated from the audio using an ASR system such as [OpenAI Whisper](https://github.com/openai/whisper), then converted into the JSON chunk format above.

The default transcript-index script expects file names of the form:

```text
day1_<source_name>_<hour_id>.json
```

For example:

```text
all_transcripts/day1_Kitchen_16.json
```

Build a transcript index:

```bash
python scripts/build_transcript_index.py \
  --transcripts-dir all_transcripts \
  --output-dir artifacts/transcript_index_day1 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --allow-download
```

### Expected runtime artifacts

After preparation, the dashboard needs at least:

```text
artifacts/
├── siglip_index_day1/
│   ├── transcript.faiss              # generic FAISS index filename used by the shared loader
│   ├── metadata.json
│   └── ...
└── transcript_index_day1/
    ├── transcript.faiss
    ├── metadata.json
    └── ...
```

The exact locations do not matter as long as the environment variables below point to them.

## Data and model resources

The course/Snellius setup uses the following paths. They are examples, not requirements:

| Resource | Example Snellius path |
|---|---|
| Visual FAISS index | `/gpfs/scratch1/shared/group_h/data_goncalo/artifacts/siglip_index_day1` |
| Transcript FAISS/BM25 index | `/gpfs/scratch1/shared/group_h/data_goncalo/artifacts/transcript_index_day1` |
| Shared Python environment | `/gpfs/scratch1/shared/group_h/data_goncalo/.venv` |
| Model cache root | `/gpfs/scratch1/shared/group_h/models` |
| Grounding outputs | `artifacts/visual_grounding/` |

For a local or public installation, set the paths yourself:

```bash
export VISUAL_INDEX_DIR=/path/to/artifacts/siglip_index_day1
export TRANSCRIPT_INDEX_DIR=/path/to/artifacts/transcript_index_day1
export SIGLIP_TEXT_MODEL_NAME=/path/to/siglip2-so400m-patch16-512-text
export MINILM_MODEL_DIR=/path/to/all-MiniLM-L6-v2
export ANSWER_SPAN_QA_MODEL=/path/to/distilbert-base-cased-distilled-squad
export GROUNDING_DINO_MODEL=/path/to/grounding-dino-base
```

Alternatively, put the same values in:

```text
castle-rag-dashboard-2/.env
```

using `castle-rag-dashboard-2/.env.example` as a template.

Important model families used in the default or optional pipeline:

- [Google SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-512) for visual keyframe retrieval.
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for transcript dense retrieval and default semantic keyword suggestions.
- [GroundingDINO base](https://huggingface.co/IDEA-Research/grounding-dino-base) for on-demand visual localization.
- [DistilBERT SQuAD](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) for transcript answer-span highlighting.
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) as an optional experimental transcript reranker.
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) in older/experimental reasoning and keyword-generation scripts, not in the default dashboard path.

Core software libraries:

- [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/) for the dashboard.
- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/docs/transformers) for model inference.
- [Sentence-Transformers](https://www.sbert.net/) for MiniLM embedding.
- BM25 and reciprocal-rank fusion for lexical/semantic transcript retrieval.

## Default configuration

The default dashboard configuration is the stable, teammate-compatible version:

```bash
CASTLE_USE_BGE_RERANKER=0
TRANSCRIPT_RERANKER_MODEL=minilm
TRANSCRIPT_RERANK_K=50
```

In this mode:

- transcript candidates are reranked with the fast MiniLM passage-similarity fallback;
- keyword suggestions are ranked with MiniLM embeddings rather than Qwen generation;
- visual keyword suggestions and extra answerability reranking are not part of the default search path;
- BGE reranking remains available only when explicitly enabled.

The main environment variables are listed in `castle-rag-dashboard-2/.env.example`.

## Run on Snellius with Slurm

If you are on the original course Snellius environment, the dashboard can be launched through the provided GPU Slurm job. The job uses the shared virtual environment, shared indexes, shared model cache, and the default MiniLM reranking configuration.

The provided `submit_dashboard.job` is currently configured for the course account layout (`~/mma-2026` plus shared `group_h` scratch resources). If you clone the repository under a different account or path, update `REPO_ROOT` and the resource paths in `castle-rag-dashboard-2/scripts/submit_dashboard.job` or override the corresponding environment variables.

```bash
cd ~/mma-2026

JOB_ID=$(sbatch --parsable castle-rag-dashboard-2/scripts/submit_dashboard.job)
echo "Job ID: ${JOB_ID}"

tail -f "slurm-${JOB_ID}.out"
```

Wait until the log shows the dashboard URL, for example:

```text
Dash is running on http://127.0.0.1:13209/
```

Find the compute node:

```bash
squeue -j "${JOB_ID}"
NODE=$(squeue -h -j "${JOB_ID}" -o "%N")
echo "${NODE}"
```

From your laptop, create an SSH tunnel. Replace `NODE_NAME` with the node printed above:

```bash
ssh -N \
  -L 14002:127.0.0.1:13209 \
  -J <username>@snellius.surf.nl \
  <username>@NODE_NAME
```

Then open:

```text
http://127.0.0.1:14002
```

If port `14002` is already in use locally, choose another local port, for example `14003`.

## Optional fast startup cache on Snellius

The default job can run directly from the shared environment. For faster startup, run the helper once to copy frequently used packages and selected model files into home storage:

```bash
cd ~/mma-2026
sbatch scripts/setup_fast_dashboard_home.sh
```

After the copy exists, `submit_dashboard.job` automatically prefers the home-local package cache and warms it before starting the dashboard. This reduces slow reads from shared GPFS during model startup.

## Run locally or interactively

Local execution is possible once the machine can access the configured indexes and model directories. On a laptop or new server, prepare local copies of the keyframes/transcripts, FAISS indexes, and model weights first.

Install dependencies:

```bash
cd mma-2026
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set the resource paths:

```bash
export PYTHONPATH="$PWD:$PWD/castle-rag-dashboard-2"
export VISUAL_INDEX_DIR=/path/to/siglip_index_day1
export TRANSCRIPT_INDEX_DIR=/path/to/transcript_index_day1
export SIGLIP_TEXT_MODEL_NAME=/path/to/siglip2-so400m-patch16-512-text
export MINILM_MODEL_DIR=/path/to/all-MiniLM-L6-v2
export ANSWER_SPAN_QA_MODEL=/path/to/distilbert-base-cased-distilled-squad
export GROUNDING_DINO_MODEL=/path/to/grounding-dino-base
export GROUNDING_DINO_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CASTLE_USE_BGE_RERANKER=0
export TRANSCRIPT_RERANKER_MODEL=minilm
export TRANSCRIPT_RERANK_K=50
```

Run:

```bash
python castle-rag-dashboard-2/app.py
```

The app defaults to:

```text
http://127.0.0.1:13209
```

To use another port:

```bash
export DASH_PORT=13210
python castle-rag-dashboard-2/app.py
```

## Optional and experimental features

These features are present in the repository but are not the default dashboard path.

### BGE transcript reranker

BGE can improve some transcript reranking cases but loads an additional model and may increase latency. Enable it explicitly:

```bash
export CASTLE_USE_BGE_RERANKER=1
export TRANSCRIPT_RERANKER_MODEL=/gpfs/scratch1/shared/group_h/models/bge-reranker-v2-m3
export TRANSCRIPT_RERANK_K=20
```

Then run the dashboard normally.

### Evaluation logging mode

Evaluation controls are external to the main dashboard. When disabled, no extra evaluation buttons are shown. To enable logging:

```bash
cd ~/mma-2026
python scripts/evaluation_control.py start --session-id P1-example-session
```

Start a task from the dashboard evaluation controls, interact with the system, then stop:

```bash
python scripts/evaluation_control.py stop
```

For the three-part think-aloud protocol:

```bash
python scripts/three_part_evaluation.py begin --participant-id P1 --participant-role "author/evaluator"
python scripts/three_part_evaluation.py part 1
python scripts/three_part_evaluation.py part 2
python scripts/three_part_evaluation.py part 3
python scripts/three_part_evaluation.py finalize
```

Logs are written under `X_evaluation/logs/` outside the application code.

### Standalone CLI retrieval

The backend can also be tested without the dashboard:

```bash
cd ~/mma-2026
export PYTHONPATH="$PWD"

# Use the active project environment. On Snellius group_h this is:
/gpfs/scratch1/shared/group_h/data_goncalo/.venv/bin/python \
  scripts/query_evidence.py "What color is the refrigerator?"
```

This is useful for debugging retrieval, routing, and enrichment separately from the Dash UI.

## Implementation notes

- The visual index contains 83,718 Day 1 keyframes in the course setup.
- The transcript index contains 41,769 transcript chunks in the course setup.
- Search retrieves both visual and transcript candidates even when the user filters the displayed modalities. This is necessary because routing and mixed ranking need both channels.
- GroundingDINO is deliberately on demand. Running object localization over every retrieved result would make search too slow.
- Transcript answer highlighting is also on demand. The retrieved transcript remains the primary evidence; the highlight is an inspection aid, not a generated answer.
- Confidence values are relative ranking scores, not calibrated probabilities.
- Compute nodes are treated as offline: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` prevent slow Hugging Face network retries.

## Acknowledgements

This project was developed for the Multimedia Analytics course at the University of Amsterdam. It builds on the CASTLE dataset and course-provided CASTLE preprocessing artifacts, including Day 1 keyframes, transcript chunks, metadata, and YouTube timestamp mappings.

We acknowledge the open-source and model communities behind Dash, Plotly, FAISS, PyTorch, Transformers, Sentence-Transformers, SigLIP/SigLIP2, GroundingDINO, DistilBERT, MiniLM, BGE, and Qwen. Model weights and datasets remain subject to their original licenses and usage terms.

## Limitations

- The repository does not include the full CASTLE media archive, large FAISS indexes, or model weights.
- The current dashboard is optimized for the Day 1 course subset, not the full CASTLE benchmark.
- The evidence router is lightweight and heuristic-assisted; it should be treated as a guide rather than ground truth.
- GroundingDINO can localize the wrong object or fail when the retrieved frame does not contain the requested object.
- Transcript chunks may include surrounding context beyond the exact answer span.
- The included evaluation tooling supports descriptive interaction analysis, but larger studies with external participants are needed for stronger general usability claims.
