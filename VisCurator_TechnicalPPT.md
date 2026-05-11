# VisCurator — Technical Presentation Content
### Architecture · Modules · Workflow

---

## SLIDE 1 — Title Slide

**Title:** VisCurator / CVAgent  
**Subtitle:** Autonomous AI-Powered Computer Vision Engineering Platform  
**Tagline:** From Raw Data to Trained Model — Fully Automated

**Speaker Notes:**  
VisCurator is a full-stack SaaS platform that automates the entire computer vision pipeline — from intelligent dataset discovery and curation, to visual model design, PyTorch code generation, and real-time training monitoring. It is powered by NVIDIA NIM (LLaMA 3.1 70B).

---

## SLIDE 2 — Problem Statement

### The CV Engineering Bottleneck

| Pain Point | Manual Reality |
|---|---|
| 🔍 Dataset Discovery | Hours searching HuggingFace, Kaggle, Roboflow manually |
| 🧹 Data Quality | Duplicates, blurry images, class imbalance — hidden issues |
| 🏗️ Model Design | Writing PyTorch boilerplate from scratch |
| 📊 Training Insight | No real-time visibility into training metrics |
| ⚖️ Validation | No scientific way to compare curated vs raw datasets |

> **VisCurator solves all five — with a single platform.**

---

## SLIDE 3 — System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 BROWSER (React + TypeScript)             │
│  Login │ Dashboard │ Dataset │ Builder │ Analytics       │
└───────────────────────────┬─────────────────────────────┘
                 REST + WebSockets + SSE
┌───────────────────────────▼─────────────────────────────┐
│              FASTAPI BACKEND  (Python 3.11)              │
│                                                          │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │  Dataset     │  │  Builder /       │  │  Auth /    │  │
│  │  Pipeline    │  │  Code Generator  │  │  Health    │  │
│  │  (Agent)     │  │  (Deterministic) │  │  Telemetry │  │
│  └──────┬───────┘  └────────┬────────┘  └────────────┘  │
│         │                   │                            │
│  ┌──────▼───────────────────▼─────────────────────────┐  │
│  │             NVIDIA NIM (LLaMA 3.1 70B)              │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │                         │
   HuggingFace API           Training Subprocess
   Kaggle API                (PyTorch, SQLite metrics)
   Roboflow API
```

---

## SLIDE 4 — Technology Stack

### Backend
| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI + Uvicorn (ASGI) |
| **Real-time** | WebSockets + Server-Sent Events (SSE) |
| **AI / LLM** | NVIDIA NIM — `meta/llama-3.1-70b-instruct` |
| **CV Processing** | PyTorch, OpenCV, Albumentations, Pillow |
| **Data Quality** | `imagehash` (perceptual dedup), Laplacian blur detection |
| **Dataset Sources** | HuggingFace `datasets`, Kaggle API, Roboflow API |
| **Auth** | JWT (HS256), OAuth2 password flow via `python-jose` |
| **Telemetry** | `psutil`, `pynvml` (GPU stats) |
| **Storage** | SQLite (training metrics), Filesystem (runs, datasets) |

### Frontend
| Layer | Technology |
|---|---|
| **Framework** | React 19 + TypeScript + Vite |
| **Styling** | Tailwind CSS (dark-themed design system) |
| **State** | Zustand (with persistence) |
| **Graph Engine** | XYFlow / React Flow (drag-and-drop model builder) |
| **Charting** | Recharts (training curves, quality analysis) |
| **Routing** | React Router v6 (protected routes) |

---

## SLIDE 5 — Backend Module Map

### Directory Structure

```
backend/
├── main.py                 # FastAPI app: all routes, WS manager, job store
├── auth.py                 # JWT auth, single-user store, OAuth2 dependency
│
├── agent/
│   ├── dataset_agent.py    # ReAct agent: conversation loop, tool dispatching
│   ├── tools.py            # Tool registry: 7 callable functions for the agent
│   ├── nim_client.py       # Async NVIDIA NIM wrapper (OpenAI SDK)
│   └── annotator.py        # SAM + CLIP auto-annotation subprocess
│
├── builder/
│   └── code_generator.py   # Deterministic PyTorch nn.Module compiler
│
├── models/
│   └── schemas.py          # Pydantic models: requests, responses, WS messages
│
└── training/
    ├── runtime.py          # Generates self-contained training directories
    └── db.py               # SQLite CRUD for per-epoch training metrics
```

---

## SLIDE 6 — Core Module: DatasetAgent (ReAct Pattern)

### What is it?
A **Conversational ReAct (Reasoning + Action) Agent** that orchestrates the full dataset curation lifecycle. It drives the LLM in a loop, parses tool calls, executes them, and feeds results back — pausing whenever human input is needed.

### Key Behaviors
- Max **20 reasoning iterations** per session
- Parses `<tool_call>JSON</tool_call>` blocks from LLM output
- **Pausable** — sends `waiting_for_user` when dataset selection is needed
- **Resumable** — agent session persists in memory (`agent_sessions` dict)
- Uses structured text formatting (not raw JSON) for tool results → prevents context truncation

### System Prompt Strategy
```
STEP 1: SEARCH FIRST — always call search_datasets immediately
STEP 2: PRESENT OPTIONS — call present_dataset_options
STEP 3: POST-SELECTION — quality estimation → planning → cleaning
STEP 4: SUMMARIZE — final curation report
```

### Agent Loop (Simplified)
```python
while iteration < MAX_ITERATIONS:
    response = await nim_client.chat(messages)          # LLM call
    tool_calls = _extract_tool_calls(response.content)  # Parse <tool_call>
    for call in tool_calls:
        result = await execute_tool(call.name, call.args)
        if result["status"] == "waiting_for_user":
            return {"paused": True}                     # Pause for input
    messages.append(format_result_for_llm(result))      # Feed back
```

---

## SLIDE 7 — Core Module: Agent Tools Registry

### 7 Tools Available to the Agent

| Tool | Description | Key Libraries |
|---|---|---|
| `search_datasets` | Multi-source search with intelligent query expansion | aiohttp, HF API, Kaggle API |
| `get_dataset_info` | Fetch splits, features, size, license for a dataset | aiohttp, HF Datasets Server |
| `estimate_dataset_quality` | Sample images: blur score, duplicates, class balance | OpenCV (Laplacian), imagehash, Pillow |
| `analyze_dataset_and_plan_processing` | Rule-based preprocessing plan from quality results | Pure Python logic |
| `clean_and_augment_dataset` | Execute image cleaning/augmentation as a subprocess | Albumentations, PyTorch |
| `generate_and_run_script` | Download HuggingFace datasets and export to disk | HuggingFace `datasets` lib |
| `present_dataset_options` | Pause pipeline, surface ranked options to the user | WebSocket emit |

### Query Expansion Strategy
```
User query: "apple leaf disease"
 → Domain map lookup → ["plant-disease", "apple disease", "leaf disease"]
 → Hyphenated form   → "apple-leaf-disease"
 → Content words     → ["apple", "leaf", "disease"]
 → 4 parallel HF API calls → merge + relevance-rank → top 8 results
```

---

## SLIDE 8 — Core Module: PyTorch Code Generator

### What is it?
A **deterministic graph compiler** — no LLM involved. Converts a React Flow visual node graph into a syntactically valid, runnable `nn.Module`.

### 4-Stage Compilation Pipeline

```
Stage 1: Topological Sort
  → Kahn's algorithm on directed graph (adj list)
  → Detects cycles; reports warnings

Stage 2: Shape Propagation
  → Tracks (channels, H, W) tensor shape through every layer
  → Auto-fixes BatchNorm feature mismatch
  → Auto-inserts Flatten before Linear layers

Stage 3: Code Emission
  → Template-based per-layer code generation
  → Estimates total parameter count
  → Emits extra classes (e.g., ResidualBlock_N)

Stage 4: Assembly
  → Combines imports + extra classes + CVAgentModel class
  → Returns code, model_summary, warnings[]
```

### Supported Layer Types
`Conv2d` · `BatchNorm2d` · `MaxPool2d` · `Linear` · `MultiheadAttention` · `ResidualBlock`

---

## SLIDE 9 — Core Module: Training Runtime

### Training Lifecycle

```
POST /api/builder/train
      │
      ▼
PyTorchCodeGenerator.generate(nodes, edges)
      │
      ▼
write_training_runtime(run_dir, model_code, ...)
  → Creates: runs/<run_id>/
              ├── model.py       (generated model class)
              ├── config.json    (epochs, lr, batch_size, dataset_path)
              └── train.py       (self-contained training script)
      │
      ▼
asyncio.create_task(_run_training_process(run_id))
  → Spawns: subprocess.Popen([python, train.py])
  → Thread reads stdout line-by-line
  → Parses METRIC:{...} and EVENT:{...} prefixed lines
  → Inserts metrics into SQLite via insert_training_metric()
  → Broadcasts to WS room via manager.broadcast(run_id, ...)
```

### Compare Mode (Curated vs Raw)
```
compare_mode=True → spawns TWO parallel training runs:
  runs/<run_id>_curated/   ← trained on AI-curated dataset
  runs/<run_id>_raw/       ← trained on original raw dataset

After both complete → generates comparison_report.json with:
  - accuracy_delta, map_delta
  - Side-by-side final metrics
```

---

## SLIDE 10 — Communication Architecture

### Three Real-time Channels

#### 1. WebSocket: Dataset Pipeline `/ws/pipeline/{job_id}`
```
Frontend ──WS connect──▶ Backend
              ◀─── catch-up logs (history replay) ───
              ◀─── THOUGHT: "Searching for datasets..." ───
              ◀─── TOOL_CALL: search_datasets({...}) ───
              ◀─── TOOL_RESULT: "Found 12 datasets" ───
              ◀─── DONE (paused=true): dataset options ───
Frontend ──POST /api/dataset/reply/{job_id}──▶
              ◀─── TOOL_CALL: estimate_dataset_quality ───
              ◀─── DONE (paused=false): curation complete ───
```

#### 2. WebSocket: Training `/ws/train/{run_id}`
```
Frontend ──WS connect──▶ Backend
              ◀─── LOG: "Training started" ───
              ◀─── TOOL_RESULT: "Epoch 1/10 | loss=0.42" ───
              ◀─── DONE: "Training completed" ───
```

#### 3. SSE: Copilot `/api/copilot/analyze`
```
Frontend ──POST (nodes, edges)──▶ Backend
              ◀─── data: {"chunk": "Your model has..."} ───
              ◀─── data: [DONE] ───
```

---

## SLIDE 11 — WebSocket Message Schema

### `PipelineMessage` — The Universal Envelope

```json
{
  "id": "uuid-v4",
  "type": "thought | tool_call | tool_result | log | script_log | done | error",
  "message": "Human-readable description",
  "data": { /* Tool-specific payload */ },
  "timestamp": "2026-05-11T14:30:00.000Z"
}
```

### Message Type Breakdown

| Type | When Emitted | Frontend Renders |
|---|---|---|
| `thought` | Agent reasoning text | Italic italic in terminal |
| `tool_call` | Before executing a tool | Tool badge with args |
| `tool_result` | After tool completes | Success/error result card |
| `log` | Pipeline step info | Plain log line |
| `script_log` | Live subprocess stdout | Monospace terminal output |
| `done` | Pipeline complete/paused | Dataset chooser or summary |
| `error` | Any failure | Red error card |

---

## SLIDE 12 — Authentication Architecture

### JWT-Based Single-User Auth

```
POST /api/auth/login
  Body: { username, password }
  ↓
  bcrypt.verify(password, hashed_password)
  ↓
  jwt.encode({ sub: username, exp: now + 24h }, JWT_SECRET, HS256)
  ↓
  Returns: { access_token, token_type: "bearer", username, role }

Protected routes:
  Authorization: Bearer <token>   (HTTP header)
  ?token=<token>                  (WebSocket query param)

FastAPI dependency: Depends(get_current_user)
  → decodes token, verifies sub, returns user dict
  → raises HTTP 401 if missing/invalid/expired
```

> **Config:** Credentials in `.env` as `VISCURATOR_USER` / `VISCURATOR_PASS`

---

## SLIDE 13 — Frontend Architecture

### Page → Component → State Map

```
/login         → LoginPage.tsx
                  └─ POST /api/auth/login → saves token in Zustand

/ (dashboard)  → DashboardPage.tsx
                  └─ System telemetry, job overview, quick actions

/dataset       → DatasetPage.tsx
                  └─ DatasetUploadZone, AgentTerminal, DatasetChooser
                  └─ WebSocket: /ws/pipeline/{job_id}

/builder       → BuilderPage.tsx
                  └─ React Flow canvas + CopilotPanel + CodeModal
                  └─ SSE: /api/copilot/analyze
                  └─ POST: /api/builder/compile, /api/builder/train

/analytics     → AnalyticsPage.tsx
                  └─ TrainingRunList, MetricCharts, CompareRunsView
                  └─ WS: /ws/train/{run_id}
                  └─ GET: /api/builder/train/{run_id}/metrics

/library       → LibraryPage.tsx
                  └─ Curated dataset cards, search, clone, download
```

### Zustand Store Slices

| Slice | Key State |
|---|---|
| **Auth** | `user`, `token`, `isAuthenticated`, `login()`, `logout()` |
| **Dataset** | `jobId`, `terminalLogs`, `isPaused`, `pendingQuestion`, `processingPlan` |
| **Flow** | `nodes`, `edges`, `setNodes()`, `setEdges()` |
| **Training** | `trainingMetrics`, `trainingLogs`, `activeRunId`, `runStatus` |

---

## SLIDE 14 — End-to-End Dataset Curation Workflow

### Full Pipeline: 8 Steps

```
1. USER  ──── Types query: "apple leaf disease detection, 2000 images"
               POST /api/dataset/search { query, target_size }
               ← Returns job_id (UUID)

2. FRONTEND ── Opens WebSocket /ws/pipeline/{job_id}
               Displays real-time terminal

3. AGENT   ── Step 1: Calls search_datasets("apple leaf disease")
               ├─ 4 parallel HF API queries
               ├─ Relevance-scored, deduplicated
               └─ Returns top 8 matches

4. AGENT   ── Step 2: Calls present_dataset_options(options=[...])
               Pipeline PAUSES → frontend shows DatasetChooser UI

5. USER    ── Selects: "plantvillage/plant-disease"
               POST /api/dataset/reply/{job_id} { reply: "dataset_id" }

6. AGENT   ── Step 3: Calls estimate_dataset_quality("plantvillage/plant-disease")
               ├─ Downloads 30 sample images
               ├─ Calculates blur (Laplacian variance), dedup (dHash)
               ├─ Analyzes class balance
               └─ Returns quality score / 100

7. AGENT   ── Step 4: Calls analyze_dataset_and_plan_processing(quality_result)
               Creates deterministic preprocessing plan

8. AGENT   ── Step 5: Calls clean_and_augment_dataset(dataset_id, plan, target_size=2000)
               ├─ Runs Python subprocess
               ├─ Filters blur, removes duplicates
               ├─ Applies Albumentations augmentations
               ├─ Exports to ./cvagent_output/<slug>/processed/
               └─ Saves preprocessing_report.json
```

---

## SLIDE 15 — End-to-End Training Workflow

### Full Pipeline: 5 Steps

```
1. USER  ──── Designs model on React Flow canvas
               (Drag-drop: Conv → BatchNorm → Pool → Linear)

2. COPILOT ── (Optional) POST /api/copilot/analyze
               NIM streams architecture advice via SSE

3. COMPILE ── POST /api/builder/compile
               PyTorchCodeGenerator runs deterministic compiler:
               → Topological sort → Shape propagation → Code emit
               ← Returns { code, model_summary, warnings }

4. TRAIN  ──── POST /api/builder/train { nodes, edges, task_type, dataset_path }
               ├─ write_training_runtime() creates runs/<run_id>/
               │   ├─ model.py  (generated PyTorch class)
               │   ├─ config.json
               │   └─ train.py  (self-contained training script)
               └─ subprocess.Popen(train.py)

5. MONITOR ── WebSocket /ws/train/{run_id}
               Per-epoch: METRIC:{"epoch":1,"loss":0.42,"accuracy":0.78,...}
               Stored in SQLite → rendered in Recharts
               Download: GET /api/builder/train/{run_id}/download → .zip
```

---

## SLIDE 16 — Data Flow Diagram

```
                   ╔══════════════╗
                   ║  HuggingFace ║
                   ║  Kaggle      ║
                   ║  Roboflow    ║
                   ╚══════╤═══════╝
                          │ search_datasets()
                          ▼
╔══════════════╗   ╔══════════════════╗   ╔════════════════╗
║   Browser    ║   ║  FastAPI Backend  ║   ║  NVIDIA NIM    ║
║  (React)     ║◄──║  /api/*           ║◄──║  LLaMA 3.1 70B ║
║              ║──►║  /ws/*            ║──►║                ║
╚══════════════╝   ╚══════╤═══════════╝   ╚════════════════╝
                          │
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
    ╔══════════════╗ ╔══════════╗ ╔══════════════╗
    ║ cvagent_output║║  SQLite   ║ ║ runs/<run_id>║
    ║  /raw         ║║  metrics  ║ ║ /model.py    ║
    ║  /processed   ║║  DB       ║ ║ /train.py    ║
    ╚══════════════╝ ╚══════════╝ ╚══════════════╝
```

---

## SLIDE 17 — Key Design Decisions

| Decision | Rationale |
|---|---|
| **ReAct Agent Pattern** | Allows LLM to reason + act in a loop; enables multi-step autonomous behavior |
| **Deterministic Code Generator** | Prevents LLM hallucination in critical PyTorch code; 100% reproducible output |
| **Training via Subprocesses** | Isolates GPU memory; allows true parallel runs; avoids blocking the event loop |
| **WebSocket Rooms** | Multiple browser tabs/users can observe the same job simultaneously |
| **History Replay on WS Connect** | Late-joining clients catch up instantly — no lost messages |
| **Structured Tool Result Formatting** | Converts large JSON to readable text before feeding to LLM — prevents context truncation |
| **Query Expansion** | HuggingFace keyword search is weak; domain synonym expansion dramatically improves recall |
| **Relevance Scoring** | Prevents popular-but-unrelated datasets from outranking crop-specific matches |
| **Compare Mode** | Scientific validation of AI curation value via parallel training runs |

---

## SLIDE 18 — API Endpoint Reference

### Dataset Pipeline
| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/dataset/browse` | ❌ | Quick dataset search (no job) |
| `POST` | `/api/dataset/search` | ✅ | Create curation job |
| `POST` | `/api/dataset/upload` | ✅ | Upload local ZIP dataset |
| `POST` | `/api/dataset/annotate` | ✅ | SAM + CLIP auto-annotation |
| `GET` | `/api/dataset/status/{job_id}` | ✅ | Poll job status |
| `POST` | `/api/dataset/reply/{job_id}` | ✅ | Send user reply to paused agent |
| `GET` | `/api/dataset/download/{slug}` | ❌ | Download processed dataset ZIP |
| `WS` | `/ws/pipeline/{job_id}` | Optional | Stream agent pipeline |

### Builder & Training
| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/builder/compile` | ✅ | Graph → PyTorch code |
| `POST` | `/api/builder/train` | ✅ | Launch training run |
| `GET` | `/api/builder/train/runs` | ✅ | List all training runs |
| `GET` | `/api/builder/train/{run_id}/metrics` | ✅ | Fetch per-epoch metrics |
| `GET` | `/api/builder/train/{run_id}/download` | ✅ | Download model + training files |
| `WS` | `/ws/train/{run_id}` | Optional | Stream training metrics |

### Copilot & System
| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `POST` | `/api/copilot/analyze` | ❌ | SSE architecture analysis |
| `GET` | `/api/health` | ❌ | Service health + diagnostics |
| `GET` | `/api/system/telemetry` | ❌ | CPU/RAM/GPU stats |
| `POST` | `/api/auth/login` | ❌ | Get JWT token |

---

## SLIDE 19 — Environment Configuration

### Required `.env` Variables

```ini
# NVIDIA NIM — Required for all AI features
NVIDIA_API_KEY=nvapi-...

# Auth (optional, defaults shown)
JWT_SECRET=viscurator-dev-secret-change-in-prod
VISCURATOR_USER=admin
VISCURATOR_PASS=viscurator

# Dataset Sources (optional but expand capabilities)
HF_TOKEN=hf_...                    # HuggingFace private datasets
KAGGLE_USERNAME=your_username      # Enables Kaggle search
KAGGLE_KEY=your_api_key

# Logging
LOG_LEVEL=info
```

### Startup Diagnostic Banner
On every startup, the backend prints a health summary:
- ✅/❌ NIM connection status
- ✅/❌ Each Python package (torch, cv2, albumentations, etc.)
- Each env var set/missing status

---

## SLIDE 20 — Deployment Architecture

### Launch Modes

```
Option 1: Unified Launcher (Development)
  python VisCurator_launch.py
    ├─ Starts FastAPI backend (port 8000)
    └─ Starts Vite dev server (port 5173)

Option 2: Separate Processes (Production)
  Backend:  python -m uvicorn backend.main:app --port 8000
  Frontend: npm run build → serve dist/ via nginx

Option 3: Docker (Planned)
  docker-compose up
    ├─ viscurator-api  (port 8000)
    └─ viscurator-ui   (port 80)
```

### CORS Policy
- Allowed origins: `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`
- Static file mount: `/data` → `./cvagent_output/` (dataset files served statically)

---

## SLIDE 21 — Scalability & Limitations

### Current Constraints
| Area | Current State | Path to Scale |
|---|---|---|
| **Job Store** | In-memory Python dict | Replace with Redis |
| **Agent Sessions** | In-memory dict | Redis-backed session store |
| **Training** | Single machine subprocess | Kubernetes + Ray Train |
| **Auth** | Single hardcoded user | Full user DB (PostgreSQL) |
| **Dataset Storage** | Local filesystem | S3 / GCS blob storage |
| **Telemetry** | Single GPU (index 0) | Multi-GPU via NVML |

### Strengths
- **Fully async** — FastAPI + asyncio + thread-based subprocess I/O
- **Windows compatible** — `WindowsProactorEventLoopPolicy` explicitly set
- **Graceful degradation** — Copilot falls back if NIM unavailable; training falls back to synthetic data
- **History replay** — WebSocket late-joiners get full message catch-up

---

## SLIDE 22 — Summary

### What VisCurator Delivers

```
┌─────────────────────────────────────────────────────┐
│   DISCOVER          CURATE              TRAIN        │
│                                                      │
│  Multi-source   → Quality Analysis  → Visual Model  │
│  Dataset Search    Blur Detection      Builder       │
│                    Deduplication                     │
│  HuggingFace       Class Balance    → Deterministic  │
│  Kaggle            Augmentation        Code Gen      │
│  Roboflow                                            │
│                 → Auto-Annotation   → Real-time      │
│  AI-Powered        SAM + CLIP          Training      │
│  (LLaMA 70B)                           Monitoring   │
│                                                      │
│  Conversational → Preprocessing     → Compare       │
│  Agent Pipeline    Report              Curated vs    │
│  (ReAct)                               Raw Dataset  │
└─────────────────────────────────────────────────────┘
```

**Stack:** FastAPI · React · NVIDIA NIM · PyTorch · Albumentations · XYFlow · Zustand · Recharts  
**Model:** `meta/llama-3.1-70b-instruct` via NVIDIA NIM API

---
*Generated from VisCurator codebase analysis — May 2026*
