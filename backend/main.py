"""
VisCurator / CVAgent — FastAPI Backend
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import site
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - environment fallback
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.dataset_agent import DatasetAgent
from backend.agent.nim_client import NIMClient
from backend.agent.tools import search_datasets
from backend.auth import get_current_user, router as auth_router
from backend.models.schemas import (
    BuilderCompileRequest,
    BuilderTrainRequest,
    BuilderTrainResponse,
    CopilotAnalyzeRequest,
    DatasetSearchRequest,
    DatasetSearchResponse,
    JobState,
    JobStatus,
    MessageType,
    PipelineMessage,
    TrainingMetricPoint,
)
from backend.training.db import get_training_metrics, init_training_db, insert_training_metric, list_training_runs
from backend.training.runtime import write_training_runtime

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("viscurator")

# ── In-memory stores ─────────────────────────────────────────

jobs: dict[UUID, dict[str, Any]] = {}
training_runs: dict[str, dict[str, Any]] = {}

# Agent sessions persist between WS connections so conversations continue
# after the user answers a question
agent_sessions: dict[UUID, DatasetAgent] = {}

nim_client: NIMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nim_client
    logger.info("Starting VisCurator backend …")
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key and api_key not in ("your_key_here", ""):
        try:
            nim_client = NIMClient(api_key=api_key)
            logger.info("NIM client initialised.")
        except Exception as exc:
            logger.warning("NIM client init failed: %s", exc)
            nim_client = None
    else:
        logger.warning("NVIDIA_API_KEY not set — agent features disabled.")

    # Ensure output dir exists and is servable
    output_dir = Path("./cvagent_output")
    output_dir.mkdir(exist_ok=True)
    Path("./runs").mkdir(exist_ok=True)
    init_training_db()

    yield

    if nim_client is not None:
        await nim_client.close()
    logger.info("VisCurator backend shut down.")


app = FastAPI(title="VisCurator API", version="0.2.0", lifespan=lifespan)

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve downloaded datasets statically
output_dir = Path("./cvagent_output")
output_dir.mkdir(exist_ok=True)
app.mount("/data", StaticFiles(directory=str(output_dir), html=False), name="data")


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "viscurator-backend",
        "version": app.version,
        "nim_connected": nim_client is not None,
        "active_jobs": len(jobs),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── System Telemetry ──────────────────────────────────────────

def _gpu_stats() -> dict[str, Any]:
    """Return GPU stats via pynvml, or null fields if unavailable."""
    null = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_utilization_percent": None,
    }
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "gpu_available": True,
            "gpu_name": name,
            "gpu_memory_used_mb": mem.used // (1024 * 1024),
            "gpu_memory_total_mb": mem.total // (1024 * 1024),
            "gpu_utilization_percent": util.gpu,
        }
    except Exception:
        return null


@app.get("/api/system/telemetry", tags=["system"])
async def system_telemetry() -> dict[str, Any]:
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path(".").resolve()))
    total_runs = len(list_training_runs())
    return {
        "cpu_percent": round(cpu, 1),
        "ram_used_gb": round(ram.used / 1024 ** 3, 2),
        "ram_total_gb": round(ram.total / 1024 ** 3, 2),
        **_gpu_stats(),
        "disk_used_gb": round(disk.used / 1024 ** 3, 2),
        "disk_total_gb": round(disk.total / 1024 ** 3, 2),
        "active_jobs": len(jobs),
        "total_training_runs": total_runs,
    }


# ── Dataset Browse (no agent, no job) ────────────────────────

@app.get("/api/dataset/browse", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def browse_datasets(
    q: str = "",
    sources: str = "huggingface,paperswithcode",
    max_results: int = 8,
) -> dict[str, Any]:
    """Directly call search_datasets without creating a job or starting the agent."""
    if not q.strip():
        return {"status": "success", "query": q, "total_found": 0, "datasets": [], "source_status": {}}
    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
    return await search_datasets(query=q, sources=source_list, max_results_per_source=max_results)


# ── Dataset Jobs ──────────────────────────────────────────────

@app.get("/api/dataset/jobs", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def list_jobs() -> list[dict[str, Any]]:
    """List all pipeline jobs, newest first."""
    result = []
    for job in sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True):
        result.append({
            "job_id": str(job["job_id"]),
            "status": job["status"],
            "query": job["request"].get("query", ""),
            "target_size": job["request"].get("target_size", 0),
            "created_at": job["created_at"].isoformat(),
            "updated_at": job["updated_at"].isoformat(),
            "summary": (job.get("result") or {}).get("summary", ""),
            "paused": (job.get("result") or {}).get("paused", False),
        })
    return result


@app.post("/api/dataset/search", response_model=DatasetSearchResponse, tags=["dataset"], dependencies=[Depends(get_current_user)])
async def create_dataset_search(request: DatasetSearchRequest) -> DatasetSearchResponse:
    job_id = uuid4()
    now = datetime.utcnow()
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobState.PENDING,
        "progress": 0.0,
        "message": "Job created.",
        "result": {},
        "request": request.model_dump(),
        "created_at": now,
        "updated_at": now,
    }
    logger.info("Job %s created — query='%s'", job_id, request.query)
    return DatasetSearchResponse(job_id=job_id, status=JobState.PENDING, created_at=now)


@app.get("/api/dataset/status/{job_id}", response_model=JobStatus, tags=["dataset"], dependencies=[Depends(get_current_user)])
async def get_job_status(job_id: UUID) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatus(**{k: job[k] for k in JobStatus.model_fields if k in job})


# ── User reply to a paused job ────────────────────────────────

@app.post("/api/dataset/reply/{job_id}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def reply_to_job(job_id: UUID, body: dict[str, Any]) -> dict[str, Any]:
    """Send a user reply to a paused/waiting pipeline job.
    
    The reply is stored and the next WS connection will resume the agent.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    job["pending_reply"] = body.get("reply", "")
    job["updated_at"] = datetime.utcnow()
    return {"status": "ok", "job_id": str(job_id), "reply_received": job["pending_reply"]}


# ── Copilot SSE ───────────────────────────────────────────────

COPILOT_SYSTEM_PROMPT = (
    "You are an expert computer vision model architecture advisor inside VisCurator. "
    "Analyze the provided React Flow graph and give: architecture summary, parameter count estimate, "
    "FLOPs estimate, warnings about dimension mismatches, concrete recommendations, compatibility notes. "
    "Be concise and technical. Use markdown."
)


async def _sse_stream_nim(system_prompt: str, user_content: str) -> AsyncIterator[str]:
    if nim_client is None:
        yield f"data: {json.dumps({'error': 'NIM not configured. Set NVIDIA_API_KEY in .env.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    try:
        async for token in nim_client.chat_stream(messages=messages, temperature=0.25, max_tokens=4096):
            yield f"data: {json.dumps({'chunk': token})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/copilot/analyze", tags=["copilot"])
async def copilot_analyze(request: CopilotAnalyzeRequest) -> StreamingResponse:
    graph_json = json.dumps({"nodes": request.nodes, "edges": request.edges}, indent=2, default=str)
    user_content = f"Here is the current model graph:\n```json\n{graph_json}\n```\n\nAnalyse this architecture."
    return StreamingResponse(
        _sse_stream_nim(COPILOT_SYSTEM_PROMPT, user_content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/builder/compile", tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_compile(request: BuilderCompileRequest) -> dict[str, Any]:
    from backend.builder.code_generator import PyTorchCodeGenerator
    generator = PyTorchCodeGenerator()
    return generator.generate(request.nodes, request.edges)


def _record_training_message(run_id: str, message: PipelineMessage) -> None:
    state = training_runs.setdefault(run_id, {"recent_messages": []})
    history = state.setdefault("recent_messages", [])
    history.append(message.ws_dict())
    if len(history) > 200:
        del history[:-200]


async def _broadcast_training_message(run_id: str, message: PipelineMessage) -> None:
    state = training_runs.setdefault(run_id, {"clients": set(), "recent_messages": []})
    _record_training_message(run_id, message)
    clients = list(state.get("clients", set()))
    stale: list[WebSocket] = []
    for client in clients:
        try:
            await client.send_json(message.ws_dict())
        except Exception:
            stale.append(client)
    for client in stale:
        state.get("clients", set()).discard(client)


async def _emit_training_log(run_id: str, message_type: MessageType, message: str, data: dict[str, Any] | None = None) -> None:
    await _broadcast_training_message(
        run_id,
        PipelineMessage(type=message_type, message=message, data=data or {}),
    )


async def _run_training_process(run_id: str, run_dir: Path) -> None:
    state = training_runs[run_id]
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(run_dir / "train.py"),
            cwd=str(run_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as exc:
        state["status"] = "failed"
        logger.exception("Training subprocess launch failed for run %s", run_id)
        await _emit_training_log(
            run_id,
            MessageType.ERROR,
            f"Failed to launch training subprocess: {exc}",
            {"run_id": run_id, "run_dir": str(run_dir)},
        )
        return

    state["process"] = proc
    state["status"] = "running"

    try:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            if line.startswith("METRIC:"):
                payload = json.loads(line[len("METRIC:"):])
                insert_training_metric(
                    run_id=run_id,
                    epoch=int(payload["epoch"]),
                    loss=float(payload["loss"]),
                    accuracy=float(payload["accuracy"]),
                    precision=float(payload["precision"]),
                    recall=float(payload["recall"]),
                    map_score=float(payload["map"]),
                    timestamp=str(payload["timestamp"]),
                )
                state["latest_metric"] = payload
                await _emit_training_log(
                    run_id,
                    MessageType.TOOL_RESULT,
                    f"Epoch {payload['epoch']}/{payload.get('total_epochs', '?')}",
                    payload,
                )
                continue

            if line.startswith("EVENT:"):
                payload = json.loads(line[len("EVENT:"):])
                event = payload.get("event")
                if event == "completed":
                    state["status"] = "completed"
                    await _emit_training_log(run_id, MessageType.DONE, payload.get("message", "Training completed."), payload)
                elif event == "error":
                    state["status"] = "failed"
                    await _emit_training_log(run_id, MessageType.ERROR, payload.get("message", "Training failed."), payload)
                else:
                    await _emit_training_log(run_id, MessageType.LOG, payload.get("message", "Training event"), payload)
                continue

            await _emit_training_log(run_id, MessageType.SCRIPT_LOG, line, {"run_id": run_id})

        exit_code = await proc.wait()
        state["exit_code"] = exit_code
        if exit_code != 0 and state.get("status") != "failed":
            state["status"] = "failed"
            await _emit_training_log(
                run_id,
                MessageType.ERROR,
                f"Training process exited with code {exit_code}.",
                {"run_id": run_id, "exit_code": exit_code},
            )
        elif state.get("status") == "running":
            state["status"] = "completed"
            await _emit_training_log(
                run_id,
                MessageType.DONE,
                "Training completed.",
                {"run_id": run_id, "exit_code": exit_code},
            )
    except Exception as exc:
        state["status"] = "failed"
        logger.exception("Training runtime failed for run %s", run_id)
        await _emit_training_log(
            run_id,
            MessageType.ERROR,
            f"Training runtime failed: {exc}",
            {"run_id": run_id},
        )


@app.post("/api/builder/train", response_model=BuilderTrainResponse, tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_train(request: BuilderTrainRequest) -> BuilderTrainResponse:
    from backend.builder.code_generator import PyTorchCodeGenerator

    run_id = uuid4().hex[:12]
    
    try:
        generator = PyTorchCodeGenerator()
        compile_result = generator.generate(request.nodes, request.edges)
        
        if request.compare_mode and request.raw_dataset_path:
            run_id_curated = f"{run_id}_curated"
            run_dir_curated = Path("./runs") / run_id_curated
            config_curated = write_training_runtime(
                run_dir=run_dir_curated,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.dataset_path,
            )
            training_runs[run_id_curated] = {
                "run_id": run_id_curated,
                "run_dir": str(run_dir_curated),
                "status": "queued",
                "task_type": request.task_type,
                "config": config_curated,
                "recent_messages": [],
                "clients": set(),
                "created_at": datetime.utcnow().isoformat(),
            }
            
            run_id_raw = f"{run_id}_raw"
            run_dir_raw = Path("./runs") / run_id_raw
            config_raw = write_training_runtime(
                run_dir=run_dir_raw,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.raw_dataset_path,
            )
            training_runs[run_id_raw] = {
                "run_id": run_id_raw,
                "run_dir": str(run_dir_raw),
                "status": "queued",
                "task_type": request.task_type,
                "config": config_raw,
                "recent_messages": [],
                "clients": set(),
                "created_at": datetime.utcnow().isoformat(),
            }
            
            async def _run_both():
                await asyncio.gather(
                    _run_training_process(run_id_curated, run_dir_curated),
                    _run_training_process(run_id_raw, run_dir_raw)
                )
                metrics_curated = get_training_metrics(run_id_curated)
                metrics_raw = get_training_metrics(run_id_raw)
                if metrics_curated and metrics_raw:
                    final_curated = metrics_curated[-1]
                    final_raw = metrics_raw[-1]
                    report = {
                        "run_ids": {"curated": run_id_curated, "raw": run_id_raw},
                        "architecture": compile_result.get("architecture", "Unknown"),
                        "epoch_count": config_curated["epochs"],
                        "final_metrics_curated": final_curated,
                        "final_metrics_raw": final_raw,
                        "accuracy_delta": final_curated["accuracy"] - final_raw["accuracy"],
                        "map_delta": final_curated["map"] - final_raw["map"]
                    }
                    (Path("./runs") / f"{run_id}_comparison_report.json").write_text(json.dumps(report, indent=2))
            asyncio.create_task(_run_both())
            return BuilderTrainResponse(run_id=run_id_curated, task_type=request.task_type)
            
        else:
            run_dir = Path("./runs") / run_id
            config = write_training_runtime(
                run_dir=run_dir,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.dataset_path,
            )
            training_runs[run_id] = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "status": "queued",
                "task_type": request.task_type,
                "config": config,
                "recent_messages": [],
                "clients": set(),
                "created_at": datetime.utcnow().isoformat(),
            }
            asyncio.create_task(_run_training_process(run_id, run_dir))
            return BuilderTrainResponse(run_id=run_id, task_type=request.task_type)
            
    except Exception as exc:
        logger.exception("Failed to initialize training run %s", run_id)
        raise HTTPException(status_code=500, detail=f"Failed to initialize training run: {exc}") from exc


@app.get("/api/builder/train/runs", tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_train_runs() -> list[dict[str, Any]]:
    rows_by_id = {row["run_id"]: row for row in list_training_runs()}
    for run_id, state in training_runs.items():
        row = rows_by_id.setdefault(
            run_id,
            {
                "run_id": run_id,
                "epochs_recorded": 0,
                "started_at": state.get("created_at"),
                "updated_at": state.get("created_at"),
                "best_loss": None,
                "best_accuracy": None,
                "best_map": None,
            },
        )
        row["status"] = state.get("status", "queued")
        row["task_type"] = state.get("task_type", "unknown")
        latest_metric = state.get("latest_metric")
        if latest_metric is not None:
            row["updated_at"] = latest_metric.get("timestamp", row["updated_at"])
            row["epochs_recorded"] = max(int(row.get("epochs_recorded") or 0), int(latest_metric.get("epoch", 0)))
    for row in rows_by_id.values():
        state = training_runs.get(row["run_id"], {})
        row.setdefault("status", state.get("status", "completed"))
        row.setdefault("task_type", state.get("task_type", "unknown"))
    return sorted(rows_by_id.values(), key=lambda item: item.get("updated_at") or "", reverse=True)


@app.get("/api/builder/train/{run_id}/download", tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_train_download(run_id: str) -> StreamingResponse:
    import io
    import zipfile
    run_dir = Path("./runs") / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in ("model.py", "config.json", "train.py"):
            fp = run_dir / filename
            if fp.exists():
                zf.write(fp, arcname=filename)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.zip"'},
    )


@app.get("/api/dataset/jobs/{job_id}/report", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def dataset_job_report(job_id: UUID) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    result = job.get("result") or {}
    # Try the preprocessing_report embedded in the job result first
    report = result.get("preprocessing_report")
    if report:
        return report
    # Fall back to reading the JSON file from disk
    output_dir = Path("./cvagent_output")
    for report_file in output_dir.rglob("preprocessing_report.json"):
        try:
            return json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Preprocessing report not found for this job.")


@app.get("/api/builder/train/{run_id}/metrics", response_model=list[TrainingMetricPoint], tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_train_metrics(run_id: str) -> list[TrainingMetricPoint]:
    return [TrainingMetricPoint(**row) for row in get_training_metrics(run_id)]


# ── WebSocket Pipeline ────────────────────────────────────────

@app.websocket("/ws/pipeline/{job_id}")
async def websocket_pipeline(websocket: WebSocket, job_id: UUID) -> None:
    await websocket.accept()
    logger.info("WebSocket connected — job %s", job_id)

    job = jobs.get(job_id)
    if job is None:
        await websocket.send_json(PipelineMessage(type=MessageType.ERROR, message=f"Job {job_id} not found.").ws_dict())
        await websocket.close(code=4004)
        return

    if nim_client is None:
        await websocket.send_json(PipelineMessage(type=MessageType.ERROR, message="NIM not configured. Set NVIDIA_API_KEY in .env.").ws_dict())
        await websocket.close(code=4003)
        return

    async def emit(msg: PipelineMessage) -> None:
        try:
            await websocket.send_json(msg.ws_dict())
        except Exception:
            pass

    job["status"] = JobState.RUNNING
    job["updated_at"] = datetime.utcnow()
    request_data = job["request"]

    try:
        # Reuse existing agent session if one exists (for conversation continuity)
        agent = agent_sessions.get(job_id)
        if agent is None:
            agent = DatasetAgent(nim_client=nim_client, job_id=job_id, emit=emit)
            agent_sessions[job_id] = agent
        else:
            # Update emit callback for the new WS connection
            agent._emit = emit

        # Check if there's a pending user reply (answering a question)
        user_reply = job.pop("pending_reply", None)

        result = await agent.run(
            query=request_data["query"],
            source=request_data.get("source", "all"),
            target_size=request_data["target_size"],
            user_reply=user_reply,
        )

        if result.get("paused"):
            job["status"] = JobState.PENDING  # paused = still pending user input
        else:
            job["status"] = JobState.COMPLETED
            # Clean up session when done
            agent_sessions.pop(job_id, None)

        job["result"] = result
        job["progress"] = 100.0
        job["updated_at"] = datetime.utcnow()

    except WebSocketDisconnect:
        logger.info("Client disconnected — job %s", job_id)
        job["status"] = JobState.PENDING  # keep it resumable
        job["updated_at"] = datetime.utcnow()

    except Exception as exc:
        logger.exception("Pipeline failed — job %s", job_id)
        job["status"] = JobState.FAILED
        job["updated_at"] = datetime.utcnow()
        try:
            await websocket.send_json(PipelineMessage(type=MessageType.ERROR, message=str(exc)).ws_dict())
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass

    logger.info("WS session ended — job %s status=%s", job_id, job["status"])


@app.websocket("/ws/train/{run_id}")
async def websocket_train(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    state = training_runs.get(run_id)
    if state is None and not get_training_metrics(run_id):
        await websocket.send_json(PipelineMessage(type=MessageType.ERROR, message=f"Run {run_id} not found.").ws_dict())
        await websocket.close(code=4004)
        return

    state = training_runs.setdefault(
        run_id,
        {"run_id": run_id, "status": "completed", "recent_messages": [], "clients": set()},
    )
    state.setdefault("clients", set()).add(websocket)

    try:
        for item in state.get("recent_messages", []):
            await websocket.send_json(item)

        latest_metrics = get_training_metrics(run_id)
        if latest_metrics and not state.get("recent_messages"):
            latest = latest_metrics[-1]
            await websocket.send_json(
                PipelineMessage(
                    type=MessageType.TOOL_RESULT,
                    message=f"Epoch {latest['epoch']}",
                    data=latest,
                ).ws_dict()
            )

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.get("clients", set()).discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import pathlib
    import uvicorn
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("backend.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
