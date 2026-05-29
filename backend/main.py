"""
VisCurator / CVAgent — FastAPI Backend
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import site
import sys
import tempfile
import zipfile
from collections import deque

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
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
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, File, UploadFile
from starlette.background import BackgroundTask
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.dataset_agent import DatasetAgent
from backend.agent.nim_client import NIMClient
from backend.agent.tools import search_huggingface
from backend.auth import get_current_user, router as auth_router
from backend.models.schemas import (
    AugmentationAgentRequest,
    BuilderCompileRequest,
    BuilderTrainRequest,
    BuilderTrainResponse,
    CopilotAnalyzeRequest,
    DatasetAnnotateRequest,
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

load_dotenv(Path(__file__).resolve().parent / ".env")

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
job_logs: dict[UUID, deque] = {}   # deque(maxlen=500) per job — O(1) append/trim
training_runs: dict[str, dict[str, Any]] = {}
active_jobs: dict[str, dict[str, Any]] = {}

# Agent sessions persist between WS connections so conversations continue
# after the user answers a question
agent_sessions: dict[UUID, DatasetAgent] = {}

nim_client: NIMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nim_client
    logger.info("Starting VisCurator backend …")
    
    # ── Diagnostic Banner ──────────────────────────────────────
    print("\n" + "="*50)
    print(" VISCURATOR BACKEND STARTUP DIAGNOSTICS")
    print("="*50)
    
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key and api_key not in ("your_key_here", ""):
        try:
            nim_client = NIMClient(api_key=api_key)
            print(f"[OK] NIM Connected (API Key: {api_key[:8]}...)")
        except Exception as exc:
            logger.warning("NIM client init failed: %s", exc)
            nim_client = None
            print(f"[FAIL] NIM Connection Failed: {exc}")
    else:
        logger.warning("NVIDIA_API_KEY not set — agent features disabled.")
        print("[FAIL] NVIDIA_API_KEY MISSING")

    print(f"\nEvent Loop: {type(asyncio.get_event_loop()).__name__}")
    print("\nPackages:")
    for pkg in ["datasets", "torch", "albumentations", "imagehash", "cv2", "PIL", "aiohttp"]:
        status = "[OK]" if _check_pkg(pkg) else "[FAIL]"
        print(f"  {status} {pkg}")
    
    print("\nEnv Vars:")
    for var in ["HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "ROBOFLOW_API_KEY"]:
        status = "set" if os.getenv(var) else "not set"
        print(f"  {var}: {status}")
    
    print("="*50 + "\n")
    # ──────────────────────────────────────────────────────────

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "null", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve downloaded datasets statically
output_dir = Path("./cvagent_output")
output_dir.mkdir(exist_ok=True)
app.mount("/data", StaticFiles(directory=str(output_dir), html=False), name="data")

hf_datasets_dir = Path("./cvagent_datasets")
hf_datasets_dir.mkdir(exist_ok=True)
app.mount("/hfdata", StaticFiles(directory=str(hf_datasets_dir), html=False), name="hfdata")


# ── Health ───────────────────────────────────────────────────

def _check_pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    return {
        "backend": "online",
        "nim_connected": nim_client is not None,
        "nim_model": "meta/llama-3.1-70b-instruct",
        "nim_error": None if nim_client else ("NVIDIA_API_KEY not set" if not os.getenv("NVIDIA_API_KEY") else "NIM connection failed"),
        "python_packages": {
            "datasets": _check_pkg("datasets"),
            "torch": _check_pkg("torch"),
            "albumentations": _check_pkg("albumentations"),
            "imagehash": _check_pkg("imagehash"),
            "cv2": _check_pkg("cv2"),
            "PIL": _check_pkg("PIL"),
            "aiohttp": _check_pkg("aiohttp"),
        },
        "env_vars": {
            "NVIDIA_API_KEY": "set" if os.getenv("NVIDIA_API_KEY") else "MISSING",
            "HF_TOKEN": "set" if os.getenv("HF_TOKEN") else "not set (optional)",
            "KAGGLE_USERNAME": "set" if os.getenv("KAGGLE_USERNAME") else "not set",
            "KAGGLE_KEY": "set" if os.getenv("KAGGLE_KEY") else "not set",
            "ROBOFLOW_API_KEY": "set" if os.getenv("ROBOFLOW_API_KEY") else "not set",
        },
        "output_dirs": {
            "cvagent_output": str(Path("./cvagent_output").exists()),
            "runs": str(Path("./runs").exists()),
        },
        "active_jobs": len(jobs),
        "websocket_path": "/ws/pipeline/{job_id}",
    }

@app.get("/api/dataset/test-run", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def dataset_test_run() -> dict[str, Any]:
    """Minimal end-to-end sanity check WITHOUT the agent."""
    logger.info("Starting minimal pipeline test run...")
    return await search_huggingface("cats dogs", max_results=3)


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

@app.get("/api/dataset/browse", tags=["dataset"])
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


@app.get("/api/dataset/list", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def list_datasets():
    datasets = []
    base_dir = Path("./cvagent_output")
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir():
                processed_dir = d / "processed"
                if processed_dir.exists():
                    # Try to get metadata from report
                    report_file = processed_dir / "preprocessing_report.json"
                    metadata = {}
                    if report_file.exists():
                        try:
                            report = json.loads(report_file.read_text(encoding="utf-8"))
                            metadata["resolution"] = report.get("plan", {}).get("augmentations", ["224"])[0] # e.g. "resize_224"
                            if "resize_" in str(metadata["resolution"]):
                                metadata["resolution"] = metadata["resolution"].replace("resize_", "")
                            else:
                                metadata["resolution"] = "224"
                            
                            metadata["num_classes"] = len(report.get("class_distribution", {}))
                            metadata["image_count"] = report.get("after_stats", {}).get("images", 0)
                        except Exception:
                            pass
                    
                    if not metadata.get("num_classes"):
                        classes = [cls.name for cls in processed_dir.iterdir() if cls.is_dir()]
                        metadata["num_classes"] = len(classes)
                        metadata["image_count"] = sum(len(list(cls.glob("*"))) for cls in processed_dir.iterdir() if cls.is_dir())
                    
                    datasets.append({
                        "id": d.name,
                        "path": str(processed_dir.absolute()),
                        "name": d.name.replace("_", " ").title(),
                        "metadata": metadata
                    })
    return datasets


@app.get("/api/dataset/inspect", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def inspect_dataset(path: str):
    """Sample images from a dataset directory to detect real image dimensions and class info."""
    import random as _random
    from PIL import Image as _Image
    ds_path = Path(path)
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path not found.")
    
    classes = [d for d in ds_path.iterdir() if d.is_dir()]
    if not classes:
        raise HTTPException(status_code=400, detail="Dataset has no class subdirectories.")
    
    # Sample up to 5 images across classes to detect resolution
    sampled_sizes: list[tuple[int, int]] = []
    for cls_dir in classes[:5]:
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if images:
            try:
                sample = _random.choice(images)
                with _Image.open(sample) as im:
                    sampled_sizes.append((im.width, im.height))
            except Exception:
                pass
    
    if sampled_sizes:
        avg_w = int(sum(w for w, _ in sampled_sizes) / len(sampled_sizes))
        avg_h = int(sum(h for _, h in sampled_sizes) / len(sampled_sizes))
    else:
        avg_w, avg_h = 224, 224

    total_images = sum(len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))) for cls_dir in classes)
    return {
        "path": path,
        "num_classes": len(classes),
        "class_names": [c.name for c in classes],
        "total_images": total_images,
        "sample_width": avg_w,
        "sample_height": avg_h,
        "resolution": f"{avg_w}x{avg_h}",
    }



@app.post("/api/dataset/search", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def create_dataset_search(request: DatasetSearchRequest) -> dict[str, Any]:
    job_id = str(uuid4())
    queue: asyncio.Queue = asyncio.Queue()

    active_jobs[job_id] = {
        "status": "running",
        "queue": queue,
        "created_at": datetime.utcnow().isoformat(),
    }

    class QueueWebSocket:
        async def send_json(self, data):
            await queue.put(data)

    async def run_pipeline():
        try:
            agent = DatasetAgent(
                websocket=QueueWebSocket(),
                job_id=job_id,
                nim_client=nim_client,
            )
            await agent.run(
                query=request.query,
                source=str(request.source),
                target_size=request.target_size,
            )
        except Exception as e:
            logger.exception("Queue-based dataset pipeline failed")
            await queue.put({"id": uuid4().hex, "type": "error", "message": str(e), "data": {}, "timestamp": datetime.utcnow().isoformat()})
        finally:
            active_jobs[job_id]["status"] = "completed"

    asyncio.create_task(run_pipeline())
    return {"job_id": job_id, "status": "running"}


import zipfile
import tempfile
from collections import deque

@app.post("/api/dataset/upload", response_model=DatasetSearchResponse, tags=["dataset"], dependencies=[Depends(get_current_user)])
async def upload_dataset(file: UploadFile = File(...)) -> DatasetSearchResponse:
    """Handle raw dataset upload via ZIP file."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    job_id = uuid4()
    now = datetime.utcnow()
    slug = f"local_{job_id.hex[:8]}"
    
    # Create directories
    base_dir = Path("./cvagent_output") / slug
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded ZIP to a temporary file, then extract safely (Zip Slip prevention)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        raw_dir_abs = raw_dir.resolve()
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                member_path = (raw_dir_abs / member).resolve()
                if not str(member_path).startswith(str(raw_dir_abs)):
                    raise HTTPException(status_code=400, detail=f"Malicious ZIP entry rejected: {member}")
            zip_ref.extractall(raw_dir)

        os.remove(tmp_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    extracted_files = [f.name for f in raw_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    # Register the job as if it were a completed search, ready for the next phase
    jobs[job_id] = {
        "job_id": str(job_id),
        "request": {
            "query": f"Local Upload: {file.filename}",
            "source": "local",
            "target_size": len(extracted_files), 
        },
        "status": JobState.COMPLETED,
        "result": {
            "paused": True,
            "type": "local_upload",
            "dataset_id": slug,
            "local_path": str(raw_dir),
            "files": extracted_files,
            "message": f"Extracted {len(extracted_files)} images for curation."
        },
        "created_at": now,
        "updated_at": now,
    }
    
    logger.info("Local upload job %s created — file='%s'", job_id, file.filename)
    return DatasetSearchResponse(job_id=job_id, status=JobState.COMPLETED, created_at=now)

@app.post("/api/dataset/seed/{job_id}/{class_name}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def upload_seeds(job_id: str, class_name: str, files: list[UploadFile] = File(...)):
    """Upload seed images (or ZIPs) for a specific class."""
    slug = f"local_{UUID(job_id).hex[:8]}"
    seed_dir = Path("./cvagent_output") / slug / "seeds" / class_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename.endswith('.zip'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                seed_dir_abs = seed_dir.resolve()
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        member_path = (seed_dir_abs / member).resolve()
                        if not str(member_path).startswith(str(seed_dir_abs)):
                            logger.warning("Zip Slip attempt in seed upload: %s", member)
                            continue
                    zip_ref.extractall(seed_dir)
                os.remove(tmp_path)
                saved_files.append(f"Extracted {file.filename}")
            except Exception as e:
                logger.error("Failed to extract seed ZIP %s: %s", file.filename, e)
        else:
            file_path = seed_dir / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            saved_files.append(file.filename)
            
    return {"status": "success", "class_name": class_name, "saved": saved_files}

@app.post("/api/dataset/annotate", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def annotate_dataset(req: DatasetAnnotateRequest) -> dict[str, Any]:
    """Trigger the heavyweight SAM/CLIP auto-annotation process."""
    job_id_str = req.job_id
    slug = f"local_{UUID(job_id_str).hex[:8]}"
    base_dir = Path("./cvagent_output") / slug
    raw_dir = base_dir / "raw"
    
    logger.info("annotate_dataset check: job_id_str=%s, slug=%s, raw_dir=%s, exists=%s", job_id_str, slug, raw_dir.absolute(), raw_dir.exists())

    if not raw_dir.exists():
        raise HTTPException(status_code=400, detail=f"Raw dataset not found at {raw_dir}. Upload first.")

    # We broadcast status via the pipeline WS room
    async def _emit(msg: PipelineMessage):
        job_uuid = UUID(job_id_str)
        history = job_logs.setdefault(job_uuid, [])
        history.append(msg)
        if len(history) > 500:
            history.pop(0)
        await manager.broadcast(job_id_str, msg.ws_dict())

    async def run_annotator():
        await _emit(PipelineMessage(type=MessageType.LOG, message="Starting Foundation Models (SAM + CLIP) for Auto-Annotation..."))
        
        script_path = Path(__file__).resolve().parent / "agent" / "annotator.py"
        
        import threading
        import subprocess

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable, str(script_path),
                        "--job-id",         job_id_str,
                        "--raw-dir",        str(raw_dir),
                        "--out-dir",        str(base_dir),
                        "--min-confidence", str(req.min_confidence),
                        "--blur-threshold", str(req.blur_threshold),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace").rstrip()
                        if line:
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                proc.wait()
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", proc.returncode))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", 1))

        threading.Thread(target=_worker, daemon=True).start()

        exit_code = 0
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                exit_code = item[1]
                break
            
            line = item
            if line.startswith("EVENT:"):
                try:
                    payload = json.loads(line[len("EVENT:"):])
                    evt = payload.get("event")
                    msg = payload.get("message", "")
                    if evt == "started":
                        await _emit(PipelineMessage(type=MessageType.LOG, message=f"Annotator: {msg}"))
                    elif evt == "progress":
                        await _emit(PipelineMessage(type=MessageType.LOG, message=f"Progress: {msg}", data=payload))
                    elif evt == "error":
                        await _emit(PipelineMessage(type=MessageType.ERROR, message=f"Annotation Error: {msg}"))
                    elif evt == "completed":
                        await _emit(PipelineMessage(type=MessageType.DONE, message=msg, data=payload))
                    elif evt == "report":
                        await _emit(PipelineMessage(type=MessageType.TOOL_RESULT, message=msg, data={"preprocessing_report": payload}))
                    else:
                        await _emit(PipelineMessage(type=MessageType.SCRIPT_LOG, message=msg))
                except Exception:
                    await _emit(PipelineMessage(type=MessageType.SCRIPT_LOG, message=line))
            else:
                await _emit(PipelineMessage(type=MessageType.SCRIPT_LOG, message=line))

        if exit_code != 0:
            await _emit(PipelineMessage(type=MessageType.ERROR, message=f"Annotator exited with code {exit_code}"))

    # Spawn in background so API returns immediately
    asyncio.create_task(run_annotator())
    return {"status": "started", "job_id": job_id_str}


@app.get("/api/dataset/annotation-report/{job_id}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def get_annotation_report(job_id: str) -> dict[str, Any]:
    """Return the annotation_report.json for a completed annotation job."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")

    slug     = f"local_{job_uuid.hex[:8]}"
    report_path = Path("./cvagent_output") / slug / "annotation_report.json"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Annotation report not found for job {job_id}. Run annotation first."
        )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {exc}")


# ── Anti-Blur + Augmentation endpoint ────────────────────────────────────────

@app.post("/api/dataset/augment", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def augment_blurry(req: dict) -> dict[str, Any]:
    """
    Trigger the anti-blur recovery + augmentation pipeline for a job.
    Scans the blur-rejected images (from the annotator's samples/filtered/ dir)
    and produces sharpened + augmented variants in <base_dir>/augmented/.

    Body JSON:
      {
        "job_id": "<uuid>",
        "n_aug":  4,           // optional, default 4
        "target_size": 224     // optional, default 224
      }
    """
    job_id_str = req.get("job_id", "")
    if not job_id_str:
        raise HTTPException(status_code=400, detail="job_id is required")

    n_aug       = int(req.get("n_aug", 4))
    target_size = int(req.get("target_size", 224))

    try:
        job_uuid = UUID(job_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")

    slug     = f"local_{job_uuid.hex[:8]}"
    base_dir = Path("./cvagent_output") / slug
    out_dir  = base_dir / "augmented"

    if not base_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Job directory not found: {base_dir}. Run annotation first."
        )

    async def _emit_aug(msg: PipelineMessage):
        history = job_logs.setdefault(job_uuid, [])
        history.append(msg)
        if len(history) > 500:
            history.pop(0)
        await manager.broadcast(job_id_str, msg.ws_dict())

    async def run_augmenter():
        await _emit_aug(PipelineMessage(
            type=MessageType.LOG,
            message=f"Starting Anti-Blur + Augmentation pipeline (n_aug={n_aug}, size={target_size})…"
        ))

        script_path = Path(__file__).resolve().parent / "agent" / "augmenter.py"

        import threading
        import subprocess

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable, str(script_path),
                        "--job-id",      job_id_str,
                        "--base-dir",    str(base_dir),
                        "--out-dir",     str(out_dir),
                        "--n-aug",       str(n_aug),
                        "--target-size", str(target_size),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace").rstrip()
                        if line:
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                proc.wait()
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", proc.returncode))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", 1))
                logger.error("Augmenter subprocess failed: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

        exit_code = 0
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                exit_code = item[1]
                break

            line = str(item)
            if line.startswith("AUG_EVENT:"):
                try:
                    payload = json.loads(line[len("AUG_EVENT:"):])
                    evt = payload.get("event", "log")
                    msg_text = payload.get("message", "")
                    if evt == "started":
                        await _emit_aug(PipelineMessage(type=MessageType.LOG, message=f"Augmenter: {msg_text}"))
                    elif evt == "progress":
                        await _emit_aug(PipelineMessage(type=MessageType.LOG, message=msg_text, data=payload))
                    elif evt == "completed":
                        await _emit_aug(PipelineMessage(type=MessageType.DONE, message=msg_text, data={"augmentation_report": payload}))
                    elif evt == "error":
                        await _emit_aug(PipelineMessage(type=MessageType.ERROR, message=msg_text))
                    elif evt == "warning":
                        await _emit_aug(PipelineMessage(type=MessageType.LOG, message=f"⚠ {msg_text}"))
                    else:
                        await _emit_aug(PipelineMessage(type=MessageType.SCRIPT_LOG, message=msg_text))
                except Exception:
                    await _emit_aug(PipelineMessage(type=MessageType.SCRIPT_LOG, message=line))
            else:
                await _emit_aug(PipelineMessage(type=MessageType.SCRIPT_LOG, message=line))

        if exit_code != 0:
            await _emit_aug(PipelineMessage(type=MessageType.ERROR, message=f"Augmenter exited with code {exit_code}"))
        else:
            # Attempt to read the summary report and broadcast it
            report_path = out_dir / "augmentation_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    await _emit_aug(PipelineMessage(
                        type=MessageType.TOOL_RESULT,
                        message="Augmentation report ready.",
                        data={"augmentation_report": report}
                    ))
                except Exception:
                    pass

    asyncio.create_task(run_augmenter())
    return {"status": "started", "job_id": job_id_str, "out_dir": str(out_dir)}


# ── Augmentation Agent (standalone, any dataset) ─────────────────────────────

@app.post("/api/dataset/augment-agent", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def run_augmentation_agent(req: AugmentationAgentRequest) -> dict[str, Any]:
    """
    Run the standalone Augmentation Agent on any ImageFolder-style dataset.

    Accepts a job_id (from an upload job) or a direct input_dir path.
    Streams progress events over the job's WebSocket channel.
    """
    job_id_str = req.job_id

    # Resolve input directory
    if req.input_dir:
        input_dir = Path(req.input_dir)
    else:
        try:
            job_uuid_inner = UUID(job_id_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job_id UUID")
        slug = f"local_{job_uuid_inner.hex[:8]}"
        # Prefer the curated/annotated output; fall back to raw upload
        base = Path("./cvagent_output") / slug
        for candidate in [base / "processed", base / "curated", base / "raw"]:
            if candidate.exists() and any(candidate.iterdir()):
                input_dir = candidate
                break
        else:
            input_dir = base  # last resort

    if not input_dir.exists():
        raise HTTPException(status_code=400,
                            detail=f"Input directory not found: {input_dir}")

    try:
        job_uuid = UUID(job_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")

    out_dir = Path("./cvagent_output") / f"local_{job_uuid.hex[:8]}" / "aug_agent_output"

    async def _emit_aug_agent(msg: PipelineMessage):
        history = job_logs.setdefault(job_uuid, [])
        history.append(msg)
        if len(history) > 500:
            history.pop(0)
        await manager.broadcast(job_id_str, msg.ws_dict())

    async def _run():
        await _emit_aug_agent(PipelineMessage(
            type=MessageType.LOG,
            message=(
                f"Augmentation Agent started — strategy={req.strategy.value}, "
                f"multiplier={req.multiplier}×, target_px={req.target_px}, "
                f"balance={req.balance}"
            )
        ))

        script_path = Path(__file__).resolve().parent / "agent" / "augmentation_agent.py"

        import threading, subprocess
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                cmd = [
                    sys.executable, str(script_path),
                    "--job-id",      job_id_str,
                    "--input-dir",   str(input_dir),
                    "--out-dir",     str(out_dir),
                    "--strategy",    req.strategy.value,
                    "--multiplier",  str(req.multiplier),
                    "--target-size", str(req.target_size),
                    "--target-px",   str(req.target_px),
                    "--balance",     str(req.balance).lower(),
                    "--max-workers", str(req.max_workers),
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace").rstrip()
                        if line:
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                proc.wait()
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", proc.returncode))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", 1))
                logger.error("Augmentation agent subprocess failed: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

        exit_code = 0
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                exit_code = item[1]
                break

            line = str(item)
            if line.startswith("AUGAGENT_EVENT:"):
                try:
                    payload = json.loads(line[len("AUGAGENT_EVENT:"):])
                    evt      = payload.get("event", "log")
                    msg_text = payload.get("message", "")
                    if evt == "started":
                        await _emit_aug_agent(PipelineMessage(
                            type=MessageType.LOG, message=f"Agent: {msg_text}", data=payload))
                    elif evt == "progress":
                        await _emit_aug_agent(PipelineMessage(
                            type=MessageType.LOG, message=msg_text, data=payload))
                    elif evt == "completed":
                        await _emit_aug_agent(PipelineMessage(
                            type=MessageType.DONE, message=msg_text,
                            data={"augmentation_report": payload}))
                    elif evt == "error":
                        await _emit_aug_agent(PipelineMessage(
                            type=MessageType.ERROR, message=msg_text))
                    else:
                        await _emit_aug_agent(PipelineMessage(
                            type=MessageType.SCRIPT_LOG, message=msg_text))
                except Exception:
                    await _emit_aug_agent(PipelineMessage(
                        type=MessageType.SCRIPT_LOG, message=line))
            else:
                await _emit_aug_agent(PipelineMessage(
                    type=MessageType.SCRIPT_LOG, message=line))

        if exit_code != 0:
            await _emit_aug_agent(PipelineMessage(
                type=MessageType.ERROR,
                message=f"Augmentation agent exited with code {exit_code}"))
        else:
            report_path = out_dir / "augmentation_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    await _emit_aug_agent(PipelineMessage(
                        type=MessageType.TOOL_RESULT,
                        message="Augmentation complete — report ready.",
                        data={"augmentation_report": report}
                    ))
                except Exception:
                    pass

    asyncio.create_task(_run())
    return {
        "status": "started",
        "job_id": job_id_str,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
    }


@app.get("/api/dataset/augment-agent/report/{job_id}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def get_augmentation_agent_report(job_id: str) -> dict[str, Any]:
    """Return the augmentation_report.json for a completed augmentation agent run."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")
    slug = f"local_{job_uuid.hex[:8]}"
    report_path = Path("./cvagent_output") / slug / "aug_agent_output" / "augmentation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No augmentation report found. Run augmentation first.")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {exc}")


@app.get("/api/dataset/download-augmented-agent/{job_id}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def download_augmented_agent(job_id: str):
    """Stream a ZIP of the augmented ImageFolder output from the augmentation agent."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")
    slug = f"local_{job_uuid.hex[:8]}"
    aug_dir = Path("./cvagent_output") / slug / "aug_agent_output"
    if not aug_dir.exists():
        raise HTTPException(status_code=404, detail="Augmented dataset not found. Run augmentation first.")

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip",
                                          dir=str(Path("./cvagent_output") / slug))
    tmp_zip_path = Path(tmp_zip.name)
    tmp_zip.close()
    try:
        with zipfile.ZipFile(tmp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(aug_dir.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"}:
                    zf.write(fp, arcname=str(fp.relative_to(aug_dir)))

        file_size = tmp_zip_path.stat().st_size

        def _stream():
            try:
                with open(tmp_zip_path, "rb") as f:
                    while chunk := f.read(65536):
                        yield chunk
            finally:
                try:
                    tmp_zip_path.unlink()
                except Exception:
                    pass

        return StreamingResponse(
            _stream(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="augmented_{slug}.zip"',
                "Content-Length": str(file_size),
            },
        )
    except Exception as exc:
        tmp_zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {exc}")


# ── Download Augmented Dataset ────────────────────────────────────────────────

@app.get("/api/dataset/download-augmented/{job_id}", tags=["dataset"], dependencies=[Depends(get_current_user)])
async def download_augmented(job_id: str):
    """
    Streams a ZIP archive containing:
      augmented/   — albumentations-generated variants
      recovered/   — anti-blurred originals
    Falls back to packaging blur_rejected/ if augmentation hasn't run yet.
    """
    from fastapi.responses import StreamingResponse

    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id UUID")

    slug     = f"local_{job_uuid.hex[:8]}"
    base_dir = Path("./cvagent_output") / slug

    # Folders to include in the ZIP (only ones that exist)
    candidates = [
        base_dir / "augmented" / "augmented",
        base_dir / "augmented" / "recovered",
        base_dir / "blur_rejected",   # fallback if augmenter hasn't run
    ]
    source_dirs = [d for d in candidates if d.exists() and any(d.iterdir())]

    if not source_dirs:
        raise HTTPException(
            status_code=404,
            detail="No augmented or blur-rejected images found. Run augmentation first."
        )

    # Write ZIP to a temp file on disk to avoid loading entire dataset into RAM (OOM fix)
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=str(base_dir))
    tmp_zip_path = Path(tmp_zip.name)
    tmp_zip.close()
    try:
        with zipfile.ZipFile(tmp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src_dir in source_dirs:
                arc_root = src_dir.name
                for file_path in sorted(src_dir.rglob("*")):
                    if file_path.is_file():
                        arc_name = arc_root + "/" + str(file_path.relative_to(src_dir))
                        zf.write(file_path, arcname=arc_name)
    except Exception as exc:
        tmp_zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {exc}")

    filename = f"augmented_{slug}.zip"
    return FileResponse(
        path=tmp_zip_path,
        filename=filename,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/dataset/status/{job_id}", response_model=JobStatus, tags=["dataset"], dependencies=[Depends(get_current_user)])
async def get_job_status(job_id: UUID) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatus(**{k: job[k] for k in JobStatus.model_fields if k in job})


import shutil
from fastapi.responses import FileResponse
from backend.agent.export_utils import export_to_yolo_classification, export_to_coco_classification, export_to_yolo_detection


@app.get("/api/dataset/processed", tags=["dataset"])
async def list_processed_datasets() -> list[dict[str, Any]]:
    """List all locally processed datasets available for download."""
    # Scan all directories where the LLM has historically written output
    OUTPUT_ROOTS = [
        Path("./cvagent_output"),
        Path("./curated_dataset"),
        Path("./output"),
    ]
    results = []
    seen_slugs: set[str] = set()

    for output_root in OUTPUT_ROOTS:
        if not output_root.exists():
            continue
        for subdir in sorted(output_root.iterdir()):
            if not subdir.is_dir():
                continue
            processed = subdir / "processed"
            if not processed.exists():
                continue
            classes = [d.name for d in processed.iterdir() if d.is_dir()]
            if not classes:
                continue
            slug = subdir.name
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            image_count = sum(
                len(list((processed / cls).glob("*.jpg"))) + len(list((processed / cls).glob("*.png")))
                for cls in classes
            )
            report_file = processed / "preprocessing_report.json"
            report = None
            if report_file.exists():
                try:
                    report = json.loads(report_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            results.append({
                "slug": slug,
                "dataset_id": report.get("dataset_id", slug) if report else slug,
                "classes": classes,
                "class_count": len(classes),
                "image_count": image_count,
                "has_report": report_file.exists(),
                "output_root": str(output_root),
                "formats": ["zip", "coco", "yolo"],
            })
    return results

@app.get("/api/dataset/download/{job_id}", tags=["dataset"])
async def download_dataset(job_id: str, format: str = "zip"):
    import io as _io
    dataset_dir = None
    for candidate in [
        Path("./cvagent_datasets") / job_id,
        Path("./cvagent_output") / job_id,
        Path("./curated_dataset") / job_id,
        Path("./output") / job_id,
    ]:
        if candidate.exists() and candidate.is_dir():
            dataset_dir = candidate
            break
    if dataset_dir is None:
        zips = list(Path("./cvagent_datasets").glob(f"*_{job_id[:8]}.zip"))
        if zips:
            return FileResponse(path=str(zips[0]), filename=zips[0].name, media_type="application/zip")
        raise HTTPException(status_code=404, detail="Dataset not found")

    buf = _io.BytesIO()
    labels_dir = dataset_dir / "labels"
    images_dir = dataset_dir / "images"
    anno_dir   = dataset_dir / "annotations"
    processed_dir = dataset_dir / "processed"
    data_yaml  = dataset_dir / "data.yaml"

    if format == "yolo":
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if data_yaml.exists():
                zf.write(data_yaml, "data.yaml")
            if labels_dir.exists():
                for fp in sorted(labels_dir.rglob("*.txt")):
                    zf.write(fp, str(fp.relative_to(dataset_dir)))
            if images_dir.exists():
                for fp in sorted(images_dir.rglob("*")):
                    if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        zf.write(fp, str(fp.relative_to(dataset_dir)))
            elif processed_dir.exists():
                classes = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])
                zf.writestr("data.yaml", f"path: .\ntrain: images/train\nval: images/val\nnc: {len(classes)}\nnames: {classes}\n")
                for ci, cls in enumerate(classes):
                    imgs = sorted((processed_dir / cls).glob("*.jpg")) + sorted((processed_dir / cls).glob("*.png"))
                    for i, img in enumerate(imgs):
                        split = "train" if i % 5 != 4 else "val"
                        zf.write(img, f"images/{split}/{cls}_{img.name}")
                        zf.writestr(f"labels/{split}/{cls}_{img.stem}.txt", f"{ci} 0.5 0.5 1.0 1.0\n")
        filename = f"{job_id}_yolo.zip"

    elif format == "coco":
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if anno_dir.exists():
                for fp in sorted(anno_dir.glob("*.json")):
                    zf.write(fp, f"annotations/{fp.name}")
                if images_dir.exists():
                    for fp in sorted(images_dir.rglob("*")):
                        if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            zf.write(fp, str(fp.relative_to(dataset_dir)))
            elif processed_dir.exists():
                from backend.agent.export_utils import export_to_coco_classification
                import tempfile as _tmp
                tmp = Path(_tmp.mktemp(suffix=".zip"))
                export_to_coco_classification(processed_dir, tmp)
                buf = _io.BytesIO(tmp.read_bytes())
                tmp.unlink(missing_ok=True)
                buf.seek(0)
                return StreamingResponse(buf, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{job_id}_coco.zip"'})
        filename = f"{job_id}_coco.zip"

    else:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(dataset_dir.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".json", ".txt", ".yaml"}:
                    zf.write(fp, str(fp.relative_to(dataset_dir)))
        filename = f"{job_id}.zip"

    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

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
    state = training_runs.setdefault(run_id, {"recent_messages": deque(maxlen=200)})
    # Ensure legacy list entries are promoted to deque
    if not isinstance(state.get("recent_messages"), deque):
        state["recent_messages"] = deque(state.get("recent_messages", []), maxlen=200)
    state["recent_messages"].append(message.ws_dict())


async def _broadcast_training_message(run_id: str, message: PipelineMessage) -> None:
    _record_training_message(run_id, message)
    await manager.broadcast(run_id, message.ws_dict())


async def _emit_training_log(run_id: str, message_type: MessageType, message: str, data: dict[str, Any] | None = None) -> None:
    await _broadcast_training_message(
        run_id,
        PipelineMessage(type=message_type, message=message, data=data or {}),
    )


async def _run_training_process(run_id: str, run_dir: Path) -> None:
    state = training_runs[run_id]
    try:
        import threading
        import subprocess

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                abs_run_dir = run_dir.absolute()
                proc = subprocess.Popen(
                    [sys.executable, str(abs_run_dir / "train.py")],
                    cwd=str(abs_run_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace").rstrip()
                        if line:
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                proc.wait()
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", proc.returncode))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("ERROR", str(e)))

        threading.Thread(target=_worker, daemon=True).start()
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

    state["status"] = "running"

    try:
        exit_code = 0
        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "DONE":
                exit_code = item[1]
                break
            elif isinstance(item, tuple) and item[0] == "ERROR":
                state["status"] = "failed"
                await _emit_training_log(run_id, MessageType.ERROR, f"Process failed: {item[1]}", {})
                return
            
            line = item

            # Robust parsing of METRIC and EVENT logs
            if line.startswith("METRIC:"):
                try:
                    payload = json.loads(line[len("METRIC:"):])
                    insert_training_metric(
                        run_id=run_id,
                        epoch=int(payload.get("epoch", 0)),
                        loss=float(payload.get("loss", 0.0)),
                        accuracy=float(payload.get("accuracy", 0.0)),
                        precision=float(payload.get("precision", 0.0)),
                        recall=float(payload.get("recall", 0.0)),
                        map_score=float(payload.get("map", 0.0)),
                        timestamp=str(payload.get("timestamp", datetime.utcnow().isoformat())),
                    )
                    state["latest_metric"] = payload
                    await _emit_training_log(
                        run_id,
                        MessageType.TOOL_RESULT,
                        f"Epoch {payload.get('epoch', '?')}/{payload.get('total_epochs', '?')}",
                        payload,
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning("Malformed metric line: %s", e)
                continue

            if line.startswith("EVENT:"):
                try:
                    payload = json.loads(line[len("EVENT:"):])
                    event_kind = payload.get("event")
                    if event_kind == "completed":
                        state["status"] = "completed"
                        await _emit_training_log(run_id, MessageType.DONE, payload.get("message", "Training completed."), payload)
                    elif event_kind == "error":
                        state["status"] = "failed"
                        await _emit_training_log(run_id, MessageType.ERROR, payload.get("message", "Training failed."), payload)
                    else:
                        await _emit_training_log(run_id, MessageType.LOG, payload.get("message", "Training event"), payload)
                except json.JSONDecodeError:
                    pass
                continue

            await _emit_training_log(run_id, MessageType.SCRIPT_LOG, line, {"run_id": run_id})

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
    logger.info("Initializing training run %s (task_type=%s)", run_id, request.task_type)
    
    try:
        generator = PyTorchCodeGenerator()
        logger.info("Compiling graph for run %s", run_id)
        compile_result = generator.generate(request.nodes, request.edges)
        
        if request.compare_mode and request.raw_dataset_path:
            logger.info("Compare mode enabled for run %s", run_id)
            run_id_curated = f"{run_id}_curated"
            run_dir_curated = Path("./runs") / run_id_curated
            run_dir_curated.mkdir(parents=True, exist_ok=True)
            config_curated = write_training_runtime(
                run_dir=run_dir_curated,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.dataset_path,
                epochs=request.epochs,
                num_images=request.num_images,
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
            run_dir_raw.mkdir(parents=True, exist_ok=True)
            config_raw = write_training_runtime(
                run_dir=run_dir_raw,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.raw_dataset_path,
                epochs=request.epochs,
                num_images=request.num_images,
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
            logger.info("Writing training runtime to %s", run_dir)
            config = write_training_runtime(
                run_dir=run_dir,
                model_code=compile_result["code"],
                nodes=request.nodes,
                task_type=request.task_type,
                dataset_path=request.dataset_path,
                epochs=request.epochs,
                num_images=request.num_images,
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
            logger.info("Spawning training process for run %s", run_id)
            asyncio.create_task(_run_training_process(run_id, run_dir))
            return BuilderTrainResponse(run_id=run_id, task_type=request.task_type)
            
    except Exception as exc:
        logger.exception("CRITICAL: Failed to initialize training run %s", run_id)
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


@app.get("/api/library", tags=["system"])
async def get_library() -> dict[str, Any]:
    """Return all processed datasets and training runs for the Library page."""
    # 1. Find processed datasets
    datasets = []
    output_dir = Path("./cvagent_output")
    if output_dir.exists():
        for ds_dir in output_dir.iterdir():
            if not ds_dir.is_dir(): continue
            processed_dir = ds_dir / "processed"
            report_file = processed_dir / "preprocessing_report.json"
            
            if report_file.exists():
                try:
                    report = json.loads(report_file.read_text(encoding="utf-8"))
                    datasets.append({
                        "id": ds_dir.name,
                        "name": report.get("dataset_id", ds_dir.name),
                        "images": report.get("after_stats", {}).get("images", 0),
                        "classes": list(report.get("class_distribution", {}).keys()),
                        "created_at": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat(),
                        "type": "dataset"
                    })
                except Exception:
                    continue

    # 2. Find training runs
    runs = list_training_runs()
    formatted_runs = []
    for run in runs:
        formatted_runs.append({
            "id": run["run_id"],
            "name": f"Model: {run['run_id']}",
            "accuracy": run.get("best_accuracy"),
            "loss": run.get("best_loss"),
            "epochs": run.get("epochs_recorded"),
            "created_at": run.get("started_at"),
            "type": "model"
        })

    return {
        "datasets": sorted(datasets, key=lambda x: x["created_at"], reverse=True),
        "models": formatted_runs
    }

@app.get("/api/builder/train/{run_id}/metrics", response_model=list[TrainingMetricPoint], tags=["builder"], dependencies=[Depends(get_current_user)])
async def builder_train_metrics(run_id: str) -> list[TrainingMetricPoint]:
    return [TrainingMetricPoint(**row) for row in get_training_metrics(run_id)]


# ── WebSocket Management ──────────────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for pipeline and training rooms."""

    def __init__(self):
        self.rooms: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        # WebSocket.accept() should be called by the handler before this if needed, 
        # but we'll do it here for consistency if not already accepted.
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.rooms:
            self.rooms[room_id].discard(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def broadcast(self, room_id: str, message: dict):
        if room_id not in self.rooms:
            return
        dead = []
        for connection in self.rooms[room_id]:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d, room_id)

manager = ConnectionManager()
active_agent_tasks: dict[UUID, asyncio.Task] = {}


# ── WebSocket Pipeline ────────────────────────────────────────

@app.websocket("/ws/pipeline/{job_id}")
async def websocket_pipeline(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        job = active_jobs.get(job_id)
        if not job:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            await websocket.close()
            return

        queue = job.get("queue")
        if queue is None:
            await websocket.send_json({"type": "error", "message": "No queue for job"})
            return

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=300.0)
                await websocket.send_json(msg)
                if msg.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"id": uuid4().hex, "type": "log", "message": "> still processing...", "data": {}, "timestamp": datetime.utcnow().isoformat()})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/train/{run_id}")
async def websocket_train(
    websocket: WebSocket, 
    run_id: str,
    token: str | None = Query(default=None),
) -> None:
    from backend.auth import get_ws_user
    await websocket.accept()

    # Auth check
    user = await get_ws_user(token)
    if not user:
        await websocket.send_json({
            "type": MessageType.ERROR,
            "message": "Authentication required. Token missing or invalid.",
            "timestamp": datetime.utcnow().isoformat(),
            "id": str(uuid4())
        })
        await websocket.close(code=4001)
        return

    state = training_runs.get(run_id)
    if state is None and not get_training_metrics(run_id):
        await websocket.send_json(PipelineMessage(type=MessageType.ERROR, message=f"Run {run_id} not found.").ws_dict())
        await websocket.close(code=4004)
        return

    # Ensure run state exists for manager to use
    state = training_runs.setdefault(
        run_id,
        {"run_id": run_id, "status": "completed", "recent_messages": []},
    )

    # Add to manager
    await manager.connect(websocket, run_id)

    try:
        # Catch up with metrics/logs
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

        # Handle pings and keep connection open
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text("pong")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
    finally:
        manager.disconnect(websocket, run_id)


if __name__ == "__main__":
    import pathlib
    import uvicorn
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("backend.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
