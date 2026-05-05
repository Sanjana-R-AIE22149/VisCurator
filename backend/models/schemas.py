"""
VisCurator / CVAgent — Pydantic Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Canonical data models shared across the API and agent layers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────

class DatasetSource(str, Enum):
    """Supported dataset search sources."""

    HUGGINGFACE = "huggingface"
    OPEN_IMAGES = "openimages"
    ROBOFLOW = "roboflow"
    KAGGLE = "kaggle"
    WEB = "web"


class MessageType(str, Enum):
    """WebSocket message categories streamed to the frontend."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOG = "log"
    SCRIPT_LOG = "script_log"   # live stdout/stderr from script execution
    DONE = "done"
    ERROR = "error"


class JobState(str, Enum):
    """Lifecycle states for a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Request / Response ───────────────────────────────────────

class DatasetSearchRequest(BaseModel):
    """Body for POST /api/dataset/search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language description of the desired dataset.",
    )
    source: DatasetSource = Field(
        default=DatasetSource.HUGGINGFACE,
        description="Platform to search for datasets.",
    )
    target_size: int = Field(
        default=1000,
        ge=10,
        le=100_000,
        description="Desired number of images in the curated dataset.",
    )


class DatasetSearchResponse(BaseModel):
    """Returned by POST /api/dataset/search upon job creation."""

    job_id: UUID
    status: JobState = JobState.PENDING
    message: str = "Pipeline job created. Connect via WebSocket to stream progress."
    created_at: datetime


# ── WebSocket Messages ───────────────────────────────────────

class PipelineMessage(BaseModel):
    """Single message streamed over the WebSocket pipeline channel."""

    type: MessageType
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def ws_dict(self) -> dict[str, Any]:
        """Serialize for WebSocket transmission (ISO timestamps)."""
        return {
            "type": self.type.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Job Status ───────────────────────────────────────────────

# ── Copilot / Builder ────────────────────────────────────────

class CopilotAnalyzeRequest(BaseModel):
    """Body for POST /api/copilot/analyze (SSE streaming)."""

    nodes: list[dict[str, Any]] = Field(
        ...,
        description="Serialised React Flow nodes (id, type, data, position).",
    )
    edges: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialised React Flow edges (source, target, etc.).",
    )


class BuilderCompileRequest(BaseModel):
    """Body for POST /api/builder/compile."""

    nodes: list[dict[str, Any]] = Field(...)
    edges: list[dict[str, Any]] = Field(default_factory=list)


# ── Job Status ───────────────────────────────────────────────

class JobStatus(BaseModel):
    """Snapshot of a running / completed pipeline job."""

    job_id: UUID
    status: JobState
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0–100).",
    )
    message: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
