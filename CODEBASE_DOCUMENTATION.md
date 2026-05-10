# VisCurator: Project Technical Documentation

VisCurator is an AI-assisted computer vision (CV) engineering platform designed to automate the transition from raw data discovery to model training. It leverages NVIDIA NIM for agentic reasoning and a deterministic compiler for model architecture generation.

---

## 1. Project Overview & Goals
- **Autonomous Dataset Curation**: A ReAct agent discovers, evaluates, and preprocesses datasets from HuggingFace, Kaggle, and Roboflow.
- **Visual Model Builder**: A drag-and-drop interface (React Flow) for designing PyTorch architectures.
- **Reliable Code Generation**: Deterministic conversion of visual graphs to valid PyTorch code (no LLM hallucination in code generation).
- **Integrated Training**: Real-time training monitoring via WebSockets with metric visualization.

---

## 2. Technical Stack

### Backend (Python/FastAPI)
- **Web Framework**: FastAPI, Uvicorn, WebSockets.
- **AI/LLM**: NVIDIA NIM (via OpenAI-compatible async client).
- **Data/ML**: PyTorch, HuggingFace `datasets`, Albumentations (augmentation), OpenCV (quality analysis), `imagehash` (deduplication).
- **Telemetry**: `psutil`, `pynvml`.

### Frontend (React/TypeScript)
- **Framework**: React 19, Vite, Tailwind CSS.
- **State Management**: Zustand (with persistence).
- **Graph Engine**: XYFlow (React Flow).
- **Visualization**: Recharts (for training metrics and quality analysis).

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
```

---

## 4. Backend Functional Documentation

### `backend/main.py`
The central entry point for the API and WebSocket services.

- **Endpoints**:
    - `POST /api/dataset/search`: Creates a new dataset curation job.
    - `GET /api/dataset/status/{job_id}`: Polls job status.
    - `POST /api/dataset/reply/{job_id}`: Sends user input to a paused agent.
    - `POST /api/copilot/analyze`: Streams LLM analysis of the current model graph (SSE).
    - `POST /api/builder/compile`: Compiles graph to PyTorch code.
    - `POST /api/builder/train`: Launches an asynchronous training process.
- **WebSockets**:
    - `/ws/pipeline/{job_id}`: Bi-directional stream for agent thoughts, tool logs, and user interaction.
    - `/ws/train/{run_id}`: Streams real-time training metrics and logs.

### `backend/agent/dataset_agent.py`
Implements the `DatasetAgent` class, a ReAct (Reasoning + Action) agent.

- **`run(query, target_size, user_reply)`**: Main loop. Iteratively calls NIM, parses `<tool_call>` blocks, executes tools, and feeds results back to the LLM. It can "pause" by returning a `waiting_for_user` status.

### `backend/agent/tools.py`
A registry of functions exposed to the agent.

- **`search_datasets(query, sources)`**: Multi-source search (HF, Kaggle, Roboflow, PWC).
- **`get_dataset_info(dataset_id)`**: Fetches metadata, features, and split sizes.
- **`estimate_dataset_quality(dataset_id)`**: Downloads a sample to calculate blur, class balance, and deduplication potential.
- **`analyze_dataset_and_plan_processing(...)`**: Deterministic rule-based logic to create a preprocessing plan.
- **`clean_and_augment_dataset(...)`**: Generates and executes a Python script to perform actual image processing (resize, filter, augment).

### `backend/builder/code_generator.py`
The `PyTorchCodeGenerator` compiler.

- **`generate(nodes, edges)`**: 
    1. Performs a topological sort of the graph.
    2. Infers tensor shapes across layers.
    3. Emits a standard `nn.Module` class with `__init__` and `forward` methods.
    4. Detects and warns about dimension mismatches (e.g., BatchNorm feature mismatch).

### `backend/training/runtime.py`
Manages the lifecycle of a training run.

- **`write_training_runtime(...)`**: Generates a self-contained directory with `model.py`, `config.json`, and a `train.py` script.
- **`_build_train_script()`**: Returns the template for a training script that supports synthetic data fallback, real ImageFolder datasets, and metric reporting via `METRIC:`-prefixed stdout lines.

---

## 5. Frontend Functional Documentation

### `src/store/useAppStore.ts`
The global state managed via Zustand.

- **Slices**:
    - **Auth**: User profile and session state.
    - **Dataset**: `jobId`, `terminalLogs`, `processingPlan`, `preprocessingReport`, and agent interaction state (`isPaused`, `pendingQuestion`).
    - **Flow**: React Flow nodes and edges state.
    - **Training**: `trainingMetrics`, `trainingLogs`, and active `runId`.

### `src/lib/api.ts`
The API client library.

- **`connectPipelineWebSocket(jobId, onMessage)`**: Manages the agent communication socket.
- **`connectTrainingWebSocket(runId, onMessage)`**: Manages the training metric socket.
- **`startTrainingRun(nodes, edges, taskType, datasetPath)`**: Triggers the backend compilation and training launch.

---

## 6. Data Models (`backend/models/schemas.py`)

- **`PipelineMessage`**: The standard envelope for WebSocket communication.
    - `type`: `thought`, `tool_call`, `tool_result`, `log`, `script_log`, `done`, `error`.
- **`JobStatus`**: State of a curation job (`pending`, `running`, `completed`, `failed`).
- **`TrainingMetricPoint`**: Structure for loss/accuracy/mAP updates.

---

## 7. Integration Details

### NVIDIA NIM Integration
The `NIMClient` (in `backend/agent/nim_client.py`) uses an async `httpx` client to talk to NVIDIA's OpenAI-compatible endpoint. It supports streaming and standard chat completions.

### Communication Flow (Dataset Pipeline)
1. Frontend POSTs search query.
2. Backend returns `job_id`.
3. Frontend opens WebSocket to `/ws/pipeline/{job_id}`.
4. Agent sends `thought` messages and `tool_call` updates.
5. If a tool needs user input (e.g., `present_dataset_options`), the backend sends a `done` message with `paused: true`.
6. Frontend displays options; user clicks; frontend POSTs reply.
7. Agent resumes via the same WebSocket logic.

---

## 8. Development Commands
- **Frontend**: `npm run dev` (Vite on port 5173).
- **Backend**: `python -m backend.main` (FastAPI on port 8000).
- **Environment**: Requires `.env` with `NVIDIA_API_KEY`, and optionally `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`.
