# VisCurator Project Status

## Summary
VisCurator is now a credible demo-grade AI-assisted CV engineering workflow prototype. The core dataset -> builder -> training -> analytics path is integrated end to end, but a few surfaces are still mock-backed or intentionally stubbed for demo speed.

## Fully Working Features
- Dataset search kickoff via `POST /api/dataset/search`
- Dataset agent websocket flow via `/ws/pipeline/{job_id}`
- Agent-driven dataset analysis using dataset metadata, blur estimation, duplicate estimation, class balance, and deterministic preprocessing planning
- Deterministic preprocessing toolchain:
  - blur filtering
  - dHash deduplication
  - resize normalization
  - Albumentations-based augmentation
  - preprocessing report export
- Library-to-Builder cloning for `YOLOv8n`, `ResNet50`, `EfficientDet`, and `DETR`
- Builder graph rendering with existing node/edge schema
- Builder compile to PyTorch via `/api/builder/compile`
- Training run creation via `POST /api/builder/train`
- Training websocket streaming via `/ws/train/{run_id}`
- SQLite-backed training metrics via `backend/data/training_metrics.db`
- Analytics page backed by real training run and metrics APIs
- Live builder terminal/training terminal log updates

## Partially Implemented Features
- Dataset pipeline depends on NVIDIA NIM for the agent reasoning layer; when NIM is unavailable, dataset orchestration cannot complete
- Copilot analysis panel is real when backend/NIM are available, but now degrades to a plain unavailable state instead of fake output
- Dashboard combines real recent activity and dataset stats with a still-static infrastructure summary card
- Preprocessing works best for HuggingFace-backed datasets; broader source normalization is not fully generalized

## Mock-Backed or Stubbed Features
- Login/auth flow is local mock auth only
- Setup modal NIM connection test is simulated
- `Settings` route is a placeholder stub
- Some dashboard infrastructure numbers are presentational rather than measured from the machine

## WebSocket-Connected Flows
- Dataset pipeline:
  - frontend: `src/components/dataset/DatasetControls.tsx`
  - transport: `src/lib/api.ts`
  - backend: `backend/main.py` `/ws/pipeline/{job_id}`
- Training pipeline:
  - frontend: `src/components/builder/CopilotPanel.tsx`, `src/components/builder/TrainingTerminal.tsx`
  - transport: `src/lib/api.ts`
  - backend: `backend/main.py` `/ws/train/{run_id}`

## Real Analytics Flows
- Run list: `GET /api/builder/train/runs`
- Metrics: `GET /api/builder/train/{run_id}/metrics`
- Persistence: `backend/training/db.py`
- UI: `src/pages/AnalyticsPage.tsx`

## Training Pipeline Status
Status: working

What is real:
- per-run `runs/<run_id>/model.py`
- generated `train.py`
- subprocess launch
- streamed epoch metrics
- SQLite metric persistence
- analytics consumption of stored metrics

Current caveats:
- training is lightweight/demo-oriented, not benchmark-oriented
- runtime fidelity is intentionally simplified for stable demos

## Dataset Preprocessing Pipeline Status
Status: working

What is real:
- deterministic planning tool
- preprocessing subprocess generation/execution
- blur, duplicate, class balance, and augmentation reporting
- frontend recommendation panel and blur chart updates from backend tool results

Current caveats:
- most robust with HuggingFace datasets
- external dataset availability still depends on network and source health

## Builder Compilation Status
Status: working

What is real:
- graph templates clone into Builder
- Builder renders nodes/edges
- compile endpoint produces deterministic PyTorch code
- compile modal shows real generated code

## Missing Integrations
- No real auth/backend identity integration
- No real machine telemetry feeding dashboard hardware cards
- No unified run history across dataset jobs, builder compile artifacts, and training runs
- No explicit preprocessing artifact browser in UI beyond report-backed summaries

## Dead Routes / Stub Routes
- `/settings` is intentionally a stub placeholder in `src/App.tsx`

## Likely Unused Assets / Low-Value Files
- `src/assets/hero.png`
- `src/assets/vite.svg`
- `src/assets/typescript.svg`
- `run_app.py` appears legacy compared with `start.py` and the new `run_demo.py`

## Known Runtime Risks
- Dataset agent flow hard-stops when NIM is unavailable
- Dataset discovery/download quality depends on external source availability
- Python environment/package setup may still vary by machine
- Large or unusual architecture graphs may compile but still produce awkward demo-time training behavior

## Demo-Breaking Risks Addressed in This Pass
- Removed fake dataset terminal fallback behavior
- Removed fake copilot analysis fallback behavior
- Added websocket close/error handling on dataset and training connections
- Added analytics loading/error handling instead of silent empty-state masking
- Added backend training subprocess launch failure handling
- Removed randomized status-bar timers that looked like live telemetry

## Build / Runtime Consistency
- Frontend build status: passes via `npm run build`
- Backend syntax validation: passes via `python -m compileall backend`
- Backend health route and websocket routes follow the same existing API style

## End-to-End Demo Flow Status
1. Search dataset: wired and real
2. Dataset analysis: wired and real
3. Preprocessing recommendation: wired and real
4. Clone architecture from Library: wired and real
5. Builder graph renders: wired and real
6. Compile to PyTorch: wired and real
7. Train model: wired and real
8. Live metrics stream: wired and real
9. Analytics update: wired and real

## Recommended Demo Framing
- Present the dataset/builder/training/analytics path as the primary workflow
- Treat auth, setup, and infrastructure telemetry as non-core demo shell surfaces
- Keep NIM configured before live demos if the dataset agent/copilot reasoning path is part of the walkthrough
