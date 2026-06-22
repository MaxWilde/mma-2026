# CASTLE RAG Video Search Dashboard

Frontend-only multimedia analytics dashboard for the CASTLE RAG project, rewritten as a Python **Dash + Plotly** application following the workshop stack.

The app uses structured mock data now. It does not implement retrieval, embeddings, FAISS, answer generation, video playback, or SAM segmentation. Those pieces can be connected later through the service layer without changing the UI components.

## Features

- Natural-language search panel
- Modality, viewpoint, and relevance-score filters
- Ranked retrieval result cards
- Generated answer panel
- Keyframe viewer with optional segmentation overlay
- Transcript, caption, and metadata tabs
- Timeline and viewpoint exploration
- Plotly score, modality, embedding-projection, and evaluation figures
- Responsive layout for desktop and smaller screens
- Backend-ready service architecture

## Project structure

```text
castle-rag-dashboard/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ README.md
├─ assets/
│  ├─ styles.css
│  └─ keyframes/
├─ castle_dashboard/
│  ├─ app.py
│  ├─ callbacks/
│  ├─ components/
│  ├─ data/
│  ├─ models/
│  ├─ services/
│  └─ utils/
└─ docs/
   ├─ backend-integration.md
   └─ feature-decisions.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally

```bash
python app.py
```

Open the local Dash URL shown in the terminal, usually:

```text
http://127.0.0.1:13209
```

## Mock data

Mock data lives in:

```text
castle_dashboard/data/mock_data.py
```

The data is typed through dataclasses in:

```text
castle_dashboard/models/schemas.py
```

The UI never imports mock data directly. Components receive data through:

```text
castle_dashboard/services/dashboard_service.py
```

## Backend integration later

Copy the environment example:

```bash
cp .env.example .env
```

When the backend exists, set:

```env
DASH_USE_MOCK_DATA=false
DASH_API_BASE_URL=http://localhost:8000/api
```

Then implement backend calls in:

```text
castle_dashboard/services/dashboard_service.py
```

Expected backend responsibility:

- query embedding
- FAISS retrieval
- transcript/caption lookup
- answer synthesis
- segmentation mask generation
- video/audio evidence URLs

The Dash callbacks and UI components already consume typed service outputs, so backend integration should not require major refactoring.
