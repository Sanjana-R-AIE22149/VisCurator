"""
VisCurator / CVAgent — FastAPI Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production-grade API server with REST endpoints and WebSocket streaming
for the agentic dataset curation pipeline.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agent.dataset_agent import DatasetAgent
from backend.agent.nim_client import NIMClient
from backend.models.schemas import (
    DatasetSearchRequest,
    DatasetSearchResponse,
    JobState,
    JobStatus,
    PipelineMessage,
    MessageType,
    CopilotAnalyzeRequest,
    BuilderCompileRequest,
)

# ── Configuration ────────────────────────────────────────────

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("viscurator")

# ── In-memory job store ──────────────────────────────────────

jobs: dict[UUID, dict[str, Any]] = {}

# ── NIM client singleton ─────────────────────────────────────

nim_client: NIMClient | None = None


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    global nim_client

    logger.info("Starting VisCurator backend …")

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key and api_key != "your_key_here":
        try:
            nim_client = NIMClient(api_key=api_key)
            logger.info("NIM client initialised successfully.")
        except Exception as exc:
            logger.warning("NIM client init failed — agent features disabled: %s", exc)
            nim_client = None
    else:
        logger.warning(
            "NVIDIA_API_KEY not configured. "
            "Agent features are disabled; set the key in .env."
        )

    yield  # ← app runs here

    if nim_client is not None:
        await nim_client.close()
    logger.info("VisCurator backend shut down.")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="VisCurator API",
    description="AI-powered computer-vision dataset curation platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    """Quick liveness / readiness probe."""
    return {
        "status": "healthy",
        "service": "viscurator-backend",
        "version": app.version,
        "nim_connected": nim_client is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Dataset Search ───────────────────────────────────────────

@app.post(
    "/api/dataset/search",
    response_model=DatasetSearchResponse,
    tags=["dataset"],
)
async def create_dataset_search(request: DatasetSearchRequest) -> DatasetSearchResponse:
    """Create a new dataset curation job.

    Returns a ``job_id`` the client can use to connect via WebSocket
    at ``/ws/pipeline/{job_id}`` to receive real-time agent updates.
    """
    job_id = uuid4()
    now = datetime.utcnow()

    jobs[job_id] = {
        "job_id": job_id,
        "status": JobState.PENDING,
        "progress": 0.0,
        "message": "Job created, awaiting WebSocket connection.",
        "result": {},
        "request": request.model_dump(),
        "created_at": now,
        "updated_at": now,
    }

    logger.info(
        "Job %s created — query='%s' source=%s target=%d",
        job_id, request.query, request.source, request.target_size,
    )

    return DatasetSearchResponse(
        job_id=job_id,
        status=JobState.PENDING,
        created_at=now,
    )


# ── Job Status ───────────────────────────────────────────────

@app.get(
    "/api/dataset/status/{job_id}",
    response_model=JobStatus,
    tags=["dataset"],
)
async def get_job_status(job_id: UUID) -> JobStatus:
    """Return the current status of a pipeline job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatus(**{k: job[k] for k in JobStatus.model_fields if k in job})


# ── Copilot – Streaming SSE ──────────────────────────────────

COPILOT_SYSTEM_PROMPT = (
    "You are an expert computer vision model architecture advisor built into "
    "VisCurator. The user is building a CV model by connecting nodes on a canvas. "
    "You receive the graph as JSON. Analyze it and provide:\n"
    "1. A summary of the current architecture\n"
    "2. Parameter count estimate\n"
    "3. FLOPs estimate\n"
    "4. Specific warnings about architectural issues (dimension mismatches, inefficiencies)\n"
    "5. Concrete recommendations with specific layer names and parameters\n"
    "6. Compatibility notes (ONNX export, quantization support)\n"
    "Be concise, technical, and specific. Use markdown formatting."
)

BUILDER_COMPILE_PROMPT = (
    "You are a PyTorch code generator built into VisCurator. "
    "The user provides a visual model graph as JSON (nodes and edges). "
    "Generate a complete, runnable PyTorch nn.Module class that implements "
    "this architecture. Include:\n"
    "- Proper imports\n"
    "- An __init__ method defining all layers\n"
    "- A forward method with correct tensor flow\n"
    "- Shape comments on each layer\n"
    "- A brief __main__ block that creates the model and runs a dummy forward pass\n"
    "Output ONLY the Python code, no explanation."
)


async def _sse_stream_nim(
    system_prompt: str,
    user_content: str,
) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from a NIM streaming response."""
    if nim_client is None:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA NIM is not configured. Set NVIDIA_API_KEY in .env.",
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        async for token in nim_client.chat_stream(
            messages=messages,
            temperature=0.25,
            max_tokens=4096,
        ):
            import json as _json
            yield f"data: {_json.dumps({'chunk': token})}\n\n"
    except Exception as exc:
        import json as _json
        logger.exception("NIM streaming error in copilot/compile")
        yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/copilot/analyze", tags=["copilot"])
async def copilot_analyze(request: CopilotAnalyzeRequest) -> StreamingResponse:
    """Analyse a React Flow graph via NVIDIA NIM and stream the response
    back as Server-Sent Events.

    Each SSE event:  ``data: {"chunk": "text here"}``
    Final event:     ``data: [DONE]``
    """
    import json as _json

    graph_json = _json.dumps(
        {"nodes": request.nodes, "edges": request.edges},
        indent=2,
        default=str,
    )
    user_content = (
        f"Here is the current model graph:\n```json\n{graph_json}\n```\n\n"
        f"Analyse this architecture and provide your expert assessment."
    )
    logger.info(
        "Copilot analyze — %d nodes, %d edges",
        len(request.nodes), len(request.edges),
    )
    return StreamingResponse(
        _sse_stream_nim(COPILOT_SYSTEM_PROMPT, user_content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/builder/compile", tags=["builder"])
async def builder_compile(request: BuilderCompileRequest) -> dict[str, Any]:
    """Deterministically compile a React Flow graph into runnable PyTorch code.

    Returns ``{"code": str, "model_summary": str, "warnings": list[str]}``.
    No LLM is needed — this uses the PyTorchCodeGenerator.
    """
    from backend.builder.code_generator import PyTorchCodeGenerator

    logger.info(
        "Builder compile — %d nodes, %d edges",
        len(request.nodes), len(request.edges),
    )
    generator = PyTorchCodeGenerator()
    result = generator.generate(request.nodes, request.edges)
    return result


# ── WebSocket Pipeline ──────────────────────────────────────

@app.websocket("/ws/pipeline/{job_id}")
async def websocket_pipeline(websocket: WebSocket, job_id: UUID) -> None:
    """Stream agent pipeline messages to the frontend in real time.

    Message format (JSON):
        {
            "type": "thought" | "tool_call" | "tool_result" | "log" | "done" | "error",
            "message": "...",
            "data": {...},
            "timestamp": "ISO-8601"
        }
    """
    await websocket.accept()
    logger.info("WebSocket connected — job %s", job_id)

    job = jobs.get(job_id)
    if job is None:
        await websocket.send_json(
            PipelineMessage(
                type=MessageType.ERROR,
                message=f"Job {job_id} not found.",
            ).ws_dict()
        )
        await websocket.close(code=4004)
        return

    if nim_client is None:
        await websocket.send_json(
            PipelineMessage(
                type=MessageType.ERROR,
                message="NVIDIA NIM is not configured. Set NVIDIA_API_KEY in .env.",
            ).ws_dict()
        )
        await websocket.close(code=4003)
        return

    # ── Emit callback: sends each PipelineMessage over the WS ──

    async def emit(msg: PipelineMessage) -> None:
        try:
            await websocket.send_json(msg.ws_dict())
        except Exception:
            logger.warning("Failed to send WS message for job %s", job_id)

    # ── Run the agent ──

    job["status"] = JobState.RUNNING
    job["updated_at"] = datetime.utcnow()

    request_data = job["request"]

    try:
        agent = DatasetAgent(
            nim_client=nim_client,
            job_id=job_id,
            emit=emit,
        )
        result = await agent.run(
            query=request_data["query"],
            source=request_data["source"],
            target_size=request_data["target_size"],
        )
        job["status"] = JobState.COMPLETED
        job["result"] = result
        job["progress"] = 100.0
        job["message"] = "Pipeline completed successfully."

    except WebSocketDisconnect:
        logger.info("Client disconnected during job %s", job_id)
        job["status"] = JobState.CANCELLED
        job["message"] = "Client disconnected."

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        job["status"] = JobState.FAILED
        job["message"] = f"Pipeline error: {exc}"
        try:
            await websocket.send_json(
                PipelineMessage(
                    type=MessageType.ERROR,
                    message=str(exc),
                ).ws_dict()
            )
        except Exception:
            pass

    finally:
        job["updated_at"] = datetime.utcnow()
        try:
            await websocket.close()
        except Exception:
            pass

    logger.info("Job %s finished — status=%s", job_id, job["status"])


# ── CLI entry point ──────────────────────────────────────────
# Usage:
#   cd VisCurator
#   python backend/main.py          # or
#   python -m backend.main

if __name__ == "__main__":
    import pathlib
    import uvicorn

    # Ensure the project root is on sys.path so `backend.*` imports resolve
    # no matter whether the user runs `python backend/main.py` from the
    # project root or from inside the backend/ directory.
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"\n  VisCurator Backend v{app.version}")
    print(f"  → http://{host}:{port}")
    print(f"  → http://{host}:{port}/docs  (Swagger UI)")
    print(f"  → WebSocket ws://{host}:{port}/ws/pipeline/{{job_id}}\n")

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
