# VisCurator: Project Technical Documentation

VisCurator is an AI-assisted computer vision (CV) engineering platform designed to automate the transition from raw data discovery to model training. It leverages NVIDIA NIM for agentic reasoning and a deterministic compiler for model architecture generation.

---

## 1. Project Overview & Goals
- **Autonomous Dataset Curation**: A ReAct agent discovers, evaluates, and preprocesses datasets from HuggingFace, Kaggle, and Roboflow.
- **Local Dataset Upload & Auto-Annotation**: Users can upload raw image ZIPs, which are annotated autonomously using SAM + CLIP foundation models.
- **Quick Annotator**: A fully in-browser zero-dependency annotator using Transformers.js (DETR) for object detection with YOLO export.
- **Dataset Augmentation Agent**: A standalone, class-aware augmentation pipeline with 5 strategy profiles and class balancing.
- **Visual Model Builder**: A drag-and-drop interface (React Flow) for designing PyTorch architectures.
- **Reliable Code Generation**: Deterministic conversion of visual graphs to valid PyTorch code (no LLM hallucination in code generation).
- **Integrated Training**: Real-time training monitoring via WebSockets with metric visualization.
- **Dataset Export**: Curated datasets can be exported in YOLO Classification, YOLO Detection, and COCO JSON formats.

---

## 2. Technical Stack

### Backend (Python/FastAPI)
- **Web Framework**: FastAPI, Uvicorn, WebSockets, SSE.
- **AI/LLM**: NVIDIA NIM (via OpenAI-compatible async client), meta/llama-3.1-70b-instruct.
- **Foundation Models**: SAM (`facebook/sam-vit-base`), CLIP (`openai/clip-vit-base-patch32`).
- **Data/ML**: PyTorch, HuggingFace `datasets`, Albumentations (augmentation), OpenCV (quality analysis), `imagehash` (deduplication).
- **Auth**: JWT (HS256) via `python-jose`, password hashing via `passlib[bcrypt]`.
- **Telemetry**: `psutil`, `pynvml`.
- **Persistence**: SQLite (training metrics via `backend/data/training_metrics.db`).

### Frontend (React/TypeScript)
- **Framework**: React 19, Vite, Tailwind CSS.
- **State Management**: Zustand (with persistence).
- **Graph Engine**: XYFlow (React Flow).
- **Visualization**: Recharts (for training metrics and quality analysis).
- **In-browser ML**: Transformers.js (`@xenova/transformers`) — DETR for Quick Annotator, runs entirely client-side.

---

## 3. Architecture Overview

```text
[Frontend: React] <---> [Backend: FastAPI] <---> [NVIDIA NIM (Agent Reasoning)]
       |                      |
       | (REST / WebSockets)  |---> [Agent Tools (HF, Kaggle, CV Analysis)]
       |                      |
       |                      |---> [PyTorch Code Generator (Deterministic)]
       |                      |
       |                      |---> [Training Runtime (Subprocesses)]
       |                      |
       |                      |---> [annotator.py subprocess (SAM + CLIP)]
       |                      |
       |                      |---> [augmenter.py subprocess (Anti-blur Recovery)]
       |                      |
       |                      |---> [augmentation_agent.py subprocess (Class-aware Aug)]
```

---

## 4. Startup & Launch

### `VisCurator_launch.py` (Recommended entry point)
A zero-dependency, self-healing launch manager. Run from the project root:

```bash
python VisCurator_launch.py
```

**What it does (10 steps):**
1. Checks Python 3.10+ and Node.js 18+.
2. Validates `backend/.env` — auto-creates from `.env.example` if missing; checks for spaces and placeholder keys.
3. Verifies all required Python packages and auto-installs any that are missing.
4. Checks `node_modules` — runs `npm install` if absent.
5. Creates runtime directories (`runs/`, `cvagent_output/`, `backend/data/`) and initialises the SQLite DB.
6. Starts the FastAPI backend (uvicorn) and polls `/api/health` until healthy (30 s timeout).
7. Starts the Vite dev server and polls until ready (120 s timeout).
8. Prints a summary banner and opens the browser to `http://localhost:5173`.
9. Watches both processes in a loop — restarts the backend up to 3 times on crash.
10. Clean shutdown on Ctrl-C.

**Default credentials**: `admin` / `viscurator` (configurable via `VISCURATOR_USER` / `VISCURATOR_PASS` env vars).

### Manual startup
```bash
# Backend (FastAPI on port 8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend (Vite on port 5173)
npm run dev
```

### Environment Variables (`backend/.env`)
| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | Yes (for agent) | NVIDIA NIM API key (prefix: `nvapi-`) |
| `HF_TOKEN` | Optional | HuggingFace token for gated datasets |
| `KAGGLE_USERNAME` | Optional | Kaggle username for Kaggle search |
| `KAGGLE_KEY` | Optional | Kaggle API key |
| `ROBOFLOW_API_KEY` | Optional | Roboflow API key |
| `VISCURATOR_USER` | Optional | Login username (default: `admin`) |
| `VISCURATOR_PASS` | Optional | Login password (default: `viscurator`) |
| `JWT_SECRET` | Optional | JWT signing secret (dev default built-in) |
| `HOST` | Optional | Bind host (default: `0.0.0.0`) |
| `PORT` | Optional | Server port (default: `8000`) |
| `LOG_LEVEL` | Optional | Logging level (default: `info`) |

---

## 5. Backend Functional Documentation

### `backend/main.py`
The central entry point for the API and WebSocket services (FastAPI v0.2.0).

#### System Endpoints
| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | None | Package checks, NIM status, env var status, active job count. |
| `GET` | `/api/system/telemetry` | None | Real-time CPU %, RAM, disk, GPU stats (via psutil / pynvml). |

#### Auth Endpoints (from `backend/auth.py`)
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/login` | None | OAuth2 password form → JWT bearer token. |

#### Dataset Endpoints
| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/dataset/browse` | None | Direct multi-source dataset search (no job/agent). |
| `GET` | `/api/dataset/jobs` | JWT | List all pipeline jobs (newest first). |
| `GET` | `/api/dataset/list` | JWT | List locally processed datasets in `cvagent_output/`. |
| `GET` | `/api/dataset/inspect` | JWT | Sample images from a dataset to detect real dimensions and classes. |
| `POST` | `/api/dataset/search` | JWT | Create a new agent curation job → returns `job_id`. |
| `POST` | `/api/dataset/upload` | JWT | Upload a ZIP of raw images → creates a local upload job. |
| `POST` | `/api/dataset/seed/{job_id}/{class_name}` | JWT | Upload seed images (or ZIPs) for a specific class for CLIP annotation. |
| `POST` | `/api/dataset/annotate` | JWT | Trigger SAM+CLIP auto-annotation subprocess for a local upload job. |
| `GET` | `/api/dataset/annotation-report/{job_id}` | JWT | Fetch the `annotation_report.json` for a completed annotation run. |
| `POST` | `/api/dataset/augment` | JWT | Trigger anti-blur recovery + Albumentations augmentation subprocess. |
| `POST` | `/api/dataset/augment-agent` | JWT | Run the standalone Augmentation Agent on any ImageFolder dataset. |
| `GET` | `/api/dataset/augment-agent/report/{job_id}` | JWT | Fetch the `augmentation_report.json` for an augmentation agent run. |
| `GET` | `/api/dataset/download-augmented/{job_id}` | JWT | Stream a ZIP of augmented + recovered images. |
| `GET` | `/api/dataset/download-augmented-agent/{job_id}` | JWT | Stream a ZIP of the Augmentation Agent's output ImageFolder. |
| `GET` | `/api/dataset/processed` | None | List all processed datasets available for download across output roots. |
| `GET` | `/api/dataset/status/{job_id}` | JWT | Poll a pipeline job's status. |
| `POST` | `/api/dataset/reply/{job_id}` | JWT | Send user input to a paused agent (re-activates pipeline). |
| `GET` | `/api/dataset/test-run` | JWT | Sanity-check the pipeline with `ylecun/mnist` bypassing agent reasoning. |
| `GET` | `/api/dataset/export/{slug}` | None | Export processed dataset as ZIP, COCO JSON, or YOLO format. |

#### Builder / Training Endpoints
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/copilot/analyze` | JWT | Stream LLM analysis of the current model graph (SSE). |
| `POST` | `/api/builder/compile` | JWT | Compile React Flow graph to PyTorch code. |
| `POST` | `/api/builder/train` | JWT | Launch async training; returns `run_id`. Supports compare mode. |
| `GET` | `/api/builder/train/runs` | JWT | List all training runs. |
| `GET` | `/api/builder/train/{run_id}/metrics` | JWT | Fetch all metric points for a specific run. |

#### WebSockets
| Path | Description |
|---|---|
| `/ws/pipeline/{job_id}` | Bi-directional stream: agent thoughts, tool calls, tool results, user interaction. |
| `/ws/train/{run_id}` | Streams real-time epoch metrics and training logs. |

#### Static Mounts
- `/data` → `./cvagent_output/` — serves processed dataset images statically.

---

### `backend/auth.py`
Minimal single-user JWT authentication module.

- **Algorithm**: HS256, 24-hour token expiry.
- **User store**: Single configurable user. Password hashed with bcrypt at import time.
- **`get_current_user`**: FastAPI dependency accepting Bearer token from header or `?token=` query param (needed for WebSocket auth).
- **`get_ws_user`**: Non-raising variant for WebSocket connections.
- **`POST /api/auth/login`**: Accepts OAuth2 password form; returns `access_token`, `username`, `name`, `role`.

---

### `backend/agent/dataset_agent.py`
Implements the `DatasetAgent` class, a ReAct (Reasoning + Action) agent.

- **`run(query, target_size, user_reply)`**: Main loop. Iteratively calls NIM, parses `<tool_call>` blocks, executes tools, and feeds results back to the LLM. Can "pause" by returning a `waiting_for_user` status when `present_dataset_options` tool is called.

---

### `backend/agent/tools.py`
A registry of functions exposed to the agent.

- **`search_datasets(query, sources)`**: Multi-source search (HF, Kaggle, Roboflow, PWC).
- **`get_dataset_info(dataset_id)`**: Fetches metadata, features, and split sizes.
- **`estimate_dataset_quality(dataset_id)`**: Downloads a sample to calculate blur, class balance, and deduplication potential.
- **`analyze_dataset_and_plan_processing(...)`**: Deterministic rule-based logic to create a preprocessing plan.
- **`clean_and_augment_dataset(...)`**: Generates and executes a Python script to perform actual image processing (resize, filter, augment).

---

### `backend/agent/annotator.py`
Heavyweight subprocess for autonomous auto-annotation of raw image uploads.
- Utilises **SAM** (`facebook/sam-vit-base`) for foreground segmentation and **CLIP** (`openai/clip-vit-base-patch32`) for zero-shot classification based on few-shot seed images.
- Filters out blurry images (Laplacian variance threshold) and exact duplicates (dHash) before annotation.
- Emits `EVENT:`-prefixed JSON lines to stdout, relayed by the backend to the WebSocket.
- Writes `annotation_report.json` to `cvagent_output/<slug>/`.

---

### `backend/agent/augmenter.py`
Isolated subprocess for recovering and expanding rejected/blurry images.
- Applies a 3-pass anti-blur pipeline: Gaussian Unsharp Mask → Richardson-Lucy Deconvolution → CLAHE.
- Generates N augmented variants (using `albumentations`) for each recovered image to compensate for dataset size reduction.
- Emits `AUG_EVENT:`-prefixed JSON lines; writes `augmentation_report.json`.

---

### `backend/agent/augmentation_agent.py`
Standalone, general-purpose augmentation agent for any ImageFolder dataset.

**Features:**
- **5 strategy profiles**: `light`, `medium`, `heavy`, `medical`, `adversarial` — each with tailored Albumentations pipelines (10+ transforms).
- **Class-aware balancing**: Minority classes receive more augmentation variants to match the majority class count.
- **Perceptual hash deduplication** (`imagehash.phash`): Near-duplicate augmented images are dropped before saving.
- **Parallel execution**: `ThreadPoolExecutor` with configurable `max_workers`.
- **Output**: ImageFolder-format output directory; preserves original images alongside augmented variants; writes `augmentation_report.json`.
- Emits `AUGAGENT_EVENT:`-prefixed JSON lines to stdout.

**CLI Arguments** (called by FastAPI via subprocess):
```
--job-id      <uuid>
--input-dir   <path/to/ImageFolder>
--out-dir     <path/to/output>
--strategy    light|medium|heavy|medical|adversarial  (default: medium)
--multiplier  <float>   (default: 3.0)
--target-size <int>     (0 = use multiplier; >0 = exact images per class)
--target-px   <int>     (default: 224, resize to NxN)
--balance     true|false (default: true)
--max-workers <int>     (default: 4)
```

---

### `backend/agent/export_utils.py`
Dataset format conversion utilities.

- **`export_to_yolo_classification(processed_dir, output_zip)`**: Converts ImageFolder to YOLO Classification format (`train/`+`val/` split at 80/20) with `data.yaml`.
- **`export_to_coco_classification(processed_dir, output_zip)`**: Converts to COCO JSON format with `annotations.json` and `images/` directory.
- **`export_to_yolo_detection(processed_dir, output_zip)`**: Converts to YOLO Detection format with full-image bounding boxes (legacy/compatibility).

---

### `backend/agent/nim_client.py`
The `NIMClient` uses an async `httpx` client to communicate with NVIDIA's OpenAI-compatible endpoint (`meta/llama-3.1-70b-instruct`). Supports streaming and standard chat completions.

---

### `backend/builder/code_generator.py`
The `PyTorchCodeGenerator` compiler.

- **`generate(nodes, edges)`**:
  1. Performs a topological sort of the graph.
  2. Infers tensor shapes across layers.
  3. Emits a standard `nn.Module` class with `__init__` and `forward` methods.
  4. Detects and warns about dimension mismatches (e.g., BatchNorm feature mismatch).

---

### `backend/training/runtime.py`
Manages the lifecycle of a training run.

- **`write_training_runtime(...)`**: Generates a self-contained directory with `model.py`, `config.json`, and a `train.py` script.
- **`_build_train_script()`**: Returns the template for a training script that supports synthetic data fallback, real ImageFolder datasets, and metric reporting via `METRIC:`-prefixed stdout lines.
- **Compare Mode**: When `compare_mode=True`, launches two parallel training subprocesses — Run A (curated) and Run B (raw) — streaming metrics for both via the same WebSocket.

---

### `backend/training/db.py`
SQLite persistence for training metrics.

- **`init_training_db()`**: Creates the `runs` table if absent.
- **`insert_training_metric(...)`**: Persists a `TrainingMetricPoint`.
- **`get_training_metrics(run_id)`**: Retrieves all metric points for a run.
- **`list_training_runs()`**: Returns a list of all unique run IDs.

---

## 6. Frontend Functional Documentation

### Application Routes (`src/App.tsx`)

| Route | Page | Auth Required |
|---|---|---|
| `/login` | `LoginPage` | No |
| `/` | `DashboardPage` | Yes |
| `/dataset` | `DatasetPage` | Yes |
| `/annotator` | `AnnotatorPage` | Yes |
| `/quick-annotator` | `QuickAnnotatorPage` | Yes |
| `/augmentation` | `AugmentationPage` | Yes |
| `/builder` | `BuilderPage` | Yes |
| `/analytics` | `AnalyticsPage` | Yes |
| `/library` | `LibraryPage` | Yes |
| `/settings` | `SettingsPage` (stub) | Yes |

A `BootScreen` component renders once per browser session on first load.

---

### Page Descriptions

#### `DashboardPage` (`src/pages/DashboardPage.tsx`)
Overview hub: recent activity, dataset stats, and infrastructure summary card. Hardware telemetry is fetched from `GET /api/system/telemetry`. The infrastructure card is partially presentational.

#### `DatasetPage` (`src/pages/DatasetPage.tsx`)
Dataset search and curation via the CVAgent pipeline. Supports HuggingFace / Kaggle / Roboflow queries. WebSocket-connected to `/ws/pipeline/{job_id}`. Displays agent thoughts, tool calls, preprocessing report, blur chart, and class distribution.

#### `AnnotatorPage` (`src/pages/AnnotatorPage.tsx`)
Full upload-and-annotate workflow:
1. Upload a ZIP of raw images (`POST /api/dataset/upload`).
2. Add seed images per class for CLIP zero-shot classification.
3. Trigger SAM + CLIP annotation (`POST /api/dataset/annotate`).
4. Optionally trigger anti-blur recovery + augmentation (`POST /api/dataset/augment`).
5. WebSocket-connected to `/ws/pipeline/{job_id}` for live progress.
6. Download final annotated dataset as ZIP.

#### `QuickAnnotatorPage` (`src/pages/QuickAnnotatorPage.tsx`)
Lightweight, fully in-browser annotator requiring no backend:
- Loads **DETR** (`Xenova/detr-resnet-50`, quantized ~50 MB) via Transformers.js from CDN.
- Users drop images, optionally define custom class names.
- DETR runs object detection at `threshold=0.4`; model labels are mapped to user-defined classes.
- Detections are rendered as bounding boxes on HTML Canvas with interactive highlighting.
- Classes can be corrected per-detection via a dropdown.
- Exports YOLO-format ZIP (`data.yaml` + per-image `labels/*.txt`) via JSZip (loaded from CDN).

#### `AugmentationPage` (`src/pages/AugmentationPage.tsx`)
Standalone augmentation interface using the Augmentation Agent:
- Select strategy (light / medium / heavy / medical / adversarial), multiplier, target resolution, and class balancing.
- Submits to `POST /api/dataset/augment-agent`.
- WebSocket-connected for live progress.
- Displays augmentation report (class summary, expansion ratio).
- Download augmented dataset via `GET /api/dataset/download-augmented-agent/{job_id}`.

#### `BuilderPage` (`src/pages/BuilderPage.tsx`)
Visual drag-and-drop model architecture designer:
- Powered by React Flow (XYFlow).
- Clone architecture templates from Library (YOLOv8n, ResNet50, EfficientDet, DETR).
- AI Copilot panel streams architectural advice via SSE (`POST /api/copilot/analyze`).
- Compile to PyTorch code (`POST /api/builder/compile`).
- Launch training (`POST /api/builder/train`) with optional Compare Mode (curated vs. raw dataset A/B).
- Training terminal streams live logs and metrics via `/ws/train/{run_id}`.

#### `AnalyticsPage` (`src/pages/AnalyticsPage.tsx`)
Training run history and metric visualization:
- Lists all training runs from `GET /api/builder/train/runs`.
- Displays loss curves, accuracy, precision, recall, and mAP per epoch via Recharts.
- Compare Mode surfaces side-by-side charts for Run A (curated) vs. Run B (raw).

#### `LibraryPage` (`src/pages/LibraryPage.tsx`)
Curated catalogue of pre-defined architecture templates:
- YOLOv8n, ResNet50, EfficientDet, DETR.
- One-click "Clone to Builder" to load a full node/edge graph into the Builder.

---

### `src/store/useAppStore.ts`
Global state via Zustand (with persistence).

- **Slices**:
  - **Auth**: User profile (`name`, `role`, `avatarInitials`), session state, `isAuthenticated`.
  - **Dataset**: `jobId`, `terminalLogs`, `processingPlan`, `preprocessingReport`, agent interaction state (`isPaused`, `pendingQuestion`).
  - **Flow**: React Flow nodes and edges state.
  - **Training**: `trainingMetrics`, `trainingLogs`, and active `runId`.

---

### `src/lib/api.ts`
The API client library.

- **`connectPipelineWebSocket(jobId, onMessage)`**: Manages the agent communication socket.
- **`connectTrainingWebSocket(runId, onMessage)`**: Manages the training metric socket.
- **`startTrainingRun(nodes, edges, taskType, datasetPath)`**: Triggers backend compilation and training launch.
- **Token management**: `setToken`, `clearToken` — JWT stored in memory and sent as `Authorization: Bearer` header.

---

## 7. Data Models (`backend/models/schemas.py`)

### Enums
| Enum | Values |
|---|---|
| `MessageType` | `thought`, `tool_call`, `tool_result`, `log`, `script_log`, `done`, `error` |
| `JobState` | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `DatasetSource` | `huggingface`, `openimages`, `roboflow`, `kaggle`, `web` |
| `AugmentationStrategy` | `light`, `medium`, `heavy`, `medical`, `adversarial` |

### Key Models
| Model | Description |
|---|---|
| `PipelineMessage` | Standard WebSocket envelope (`id`, `type`, `message`, `data`, `timestamp`). |
| `JobStatus` | Snapshot of a pipeline job (state, progress 0–100, result dict). |
| `DatasetSearchRequest` | Body for `/api/dataset/search` (`query`, `source`, `target_size`). |
| `DatasetSearchResponse` | Response with `job_id`, `status`, `created_at`. |
| `DatasetAnnotateRequest` | Body for `/api/dataset/annotate` (`job_id`, `min_confidence`, `blur_threshold`). |
| `AugmentationAgentRequest` | Body for `/api/dataset/augment-agent` (all augmentation parameters). |
| `CopilotAnalyzeRequest` | Body for `/api/copilot/analyze` (`nodes`, `edges`). |
| `BuilderCompileRequest` | Body for `/api/builder/compile` (`nodes`, `edges`). |
| `BuilderTrainRequest` | Body for `/api/builder/train` (nodes, edges, task type, dataset path, compare mode, epochs). |
| `BuilderTrainResponse` | Response with `run_id`, `status`, `task_type`. |
| `TrainingMetricPoint` | Per-epoch metric (`run_id`, `epoch`, `loss`, `accuracy`, `precision`, `recall`, `map`). |
| `TokenResponse` | Auth login response (`access_token`, `token_type`, `username`, `name`, `role`). |

---

## 8. Communication Flows

### Dataset Curation Pipeline (HuggingFace / Kaggle)
1. Frontend POSTs search query to `POST /api/dataset/search`.
2. Backend returns `job_id`.
3. Frontend opens WebSocket to `/ws/pipeline/{job_id}`.
4. Agent sends `thought` messages and `tool_call` updates.
5. If a tool needs user input (`present_dataset_options`), backend sends a `done` message with `paused: true`.
6. Frontend displays options; user clicks; frontend POSTs reply to `POST /api/dataset/reply/{job_id}`.
7. Agent resumes via the same WebSocket logic.

### Local Upload & Annotation Pipeline
1. User uploads ZIP → `POST /api/dataset/upload` → `job_id` returned with status `COMPLETED`.
2. User uploads seed images per class → `POST /api/dataset/seed/{job_id}/{class_name}`.
3. Frontend opens WebSocket to `/ws/pipeline/{job_id}`.
4. User triggers annotation → `POST /api/dataset/annotate`.
5. Backend spawns `annotator.py` subprocess; stdout lines are relayed as WebSocket messages.
6. On completion, `annotation_report.json` is readable via `GET /api/dataset/annotation-report/{job_id}`.

### Training Pipeline
1. Frontend calls `POST /api/builder/train` with graph nodes/edges + dataset path.
2. Backend compiles PyTorch code, writes runtime files to `runs/<run_id>/`, launches subprocess.
3. Frontend opens WebSocket to `/ws/train/{run_id}`.
4. Training script emits `METRIC:` lines → backend parses and relays as WebSocket messages + persists to SQLite.
5. Analytics page reads stored metrics from `GET /api/builder/train/{run_id}/metrics`.

---

## 9. Development Commands

```bash
# Launch everything (recommended)
python VisCurator_launch.py

# Frontend only
npm run dev                         # Vite on port 5173

# Backend only
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Backend syntax check
python -m compileall backend

# Frontend production build
npm run build
```

---

## 10. Testing & Quality Assurance

### Integration & Health Checks
- **`GET /api/health`**: Validates availability of all required Python packages (e.g., `torch`, `transformers`, `albumentations`) and environment variables.
- **`GET /api/dataset/test-run`**: Programmatically sanity-checks the pipeline with a minimal dataset (`ylecun/mnist`) bypassing the agentic reasoning loop.
- **`run_demo.py`**: A root-level script to simulate a full end-to-end dataset curation pipeline run.

### Quantitative Quality Metrics
- **Dataset Level**: The system self-evaluates via the `estimate_dataset_quality` tool, measuring blur ratio, duplicate percentage, class balance (Gini coefficient), and average resolution.
- **Augmentation Level**: Calculates Laplacian variance delta before and after the anti-blur recovery pipeline (`avg_blur_before` vs `avg_blur_after`). The Augmentation Agent reports `expansion_ratio` and per-class `total` counts.
- **Model Training**: Evaluates training quality via SQLite-backed per-epoch metrics: loss, accuracy, precision, recall, and Mean Average Precision (mAP).

### Comparative Experiment Design (A/B Testing)
The most rigorous validation mechanism is the **Compare Mode** in the Builder:
- Trains the exact same model architecture and hyperparameters on two datasets simultaneously.
- **Run A**: Curated/Annotated dataset.
- **Run B**: Raw/Unprocessed dataset.
- Surfaces side-by-side loss curves and accuracy/mAP deltas. A statistical improvement signal is displayed if the accuracy delta exceeds a threshold (e.g., 2%).

---

## 11. Directory Structure (Key Paths)

```
VisCurator/
├── VisCurator_launch.py        # Self-healing launcher (entry point)
├── backend/
│   ├── .env                    # Secret keys (git-ignored)
│   ├── .env.example            # Template for .env
│   ├── requirements.txt        # Python dependencies
│   ├── main.py                 # FastAPI app, all endpoints & WebSockets
│   ├── auth.py                 # JWT authentication module
│   ├── check_env.py            # Standalone environment diagnostics
│   ├── agent/
│   │   ├── dataset_agent.py    # ReAct CVAgent
│   │   ├── tools.py            # Agent tool registry
│   │   ├── annotator.py        # SAM + CLIP annotation subprocess
│   │   ├── augmenter.py        # Anti-blur + Albumentations subprocess
│   │   ├── augmentation_agent.py  # Standalone augmentation agent
│   │   ├── export_utils.py     # COCO / YOLO export helpers
│   │   └── nim_client.py       # NVIDIA NIM async HTTP client
│   ├── builder/
│   │   └── code_generator.py   # Deterministic PyTorch code compiler
│   ├── training/
│   │   ├── runtime.py          # Training run file generator
│   │   └── db.py               # SQLite metric persistence
│   ├── models/
│   │   └── schemas.py          # All Pydantic models
│   └── data/
│       └── training_metrics.db # SQLite database (auto-created)
├── src/
│   ├── App.tsx                 # Routes and boot screen
│   ├── pages/
│   │   ├── DashboardPage.tsx
│   │   ├── DatasetPage.tsx
│   │   ├── AnnotatorPage.tsx   # Upload + SAM/CLIP annotation
│   │   ├── QuickAnnotatorPage.tsx  # In-browser DETR annotator
│   │   ├── AugmentationPage.tsx    # Augmentation Agent UI
│   │   ├── BuilderPage.tsx
│   │   ├── AnalyticsPage.tsx
│   │   ├── LibraryPage.tsx
│   │   └── LoginPage.tsx
│   ├── components/
│   │   ├── layout/             # AppLayout, Sidebar
│   │   ├── auth/               # ProtectedRoute
│   │   ├── builder/            # CopilotPanel, TrainingTerminal, nodes
│   │   ├── dataset/            # DatasetControls, TerminalPanel
│   │   ├── onboarding/         # SetupModal, BootScreen
│   │   └── ui/                 # Shared UI primitives
│   ├── store/
│   │   └── useAppStore.ts      # Zustand global state
│   └── lib/
│       └── api.ts              # REST + WebSocket client helpers
├── cvagent_output/             # Agent-downloaded and processed datasets
├── runs/                       # Training run artifacts (model.py, train.py, logs)
└── test_suite/                 # Integration test helpers
```
