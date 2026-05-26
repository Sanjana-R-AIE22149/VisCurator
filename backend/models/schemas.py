"""
VisCurator / CVAgent — Pydantic Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Canonical data models shared across the API and agent layers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

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

class DatasetAnnotateRequest(BaseModel):
    """Body for POST /api/dataset/annotate."""

    job_id: str = Field(..., description="The ID of the local upload job.")
    min_confidence: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="CLIP cosine similarity threshold below which images go to low_confidence/ instead of train/.",
    )
    blur_threshold: float = Field(
        default=80.0,
        ge=1.0,
        le=500.0,
        description="Laplacian variance threshold for blur rejection. Lower = stricter (rejects more).",
    )


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

    id: UUID = Field(default_factory=uuid4)
    type: MessageType
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def ws_dict(self) -> dict[str, Any]:
        """Serialize for WebSocket transmission (ISO timestamps)."""
        return {
            "id": str(self.id),
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


class BuilderTrainRequest(BaseModel):
    """Body for POST /api/builder/train."""

    nodes: list[dict[str, Any]] = Field(...)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    task_type: str = Field(default="mnist_classification", pattern="^(mnist_classification|object_detection|custom_curated)$")
    dataset_path: str | None = Field(default=None, description="Path to an ImageFolder-format dataset directory.")
    compare_mode: bool = Field(default=False, description="Whether to run a parallel training run for comparison.")
    raw_dataset_path: str | None = Field(default=None, description="Path to the raw dataset for comparison.")
    epochs: int | None = Field(default=None, description="Number of epochs to train.")
    num_images: int | None = Field(default=None, description="Number of images to use.")


class BuilderTrainResponse(BaseModel):
    """Returned when a training run is created."""

    run_id: str
    status: str = "queued"
    task_type: str


class TrainingMetricPoint(BaseModel):
    run_id: str
    epoch: int
    loss: float
    accuracy: float
    precision: float
    recall: float
    map: float
    timestamp: str


# ── Augmentation Agent ───────────────────────────────────────

class AugmentationStrategy(str, Enum):
    """Available augmentation intensity profiles."""
    LIGHT       = "light"
    MEDIUM      = "medium"
    HEAVY       = "heavy"
    MEDICAL     = "medical"
    ADVERSARIAL = "adversarial"


class AugmentationAgentRequest(BaseModel):
    """Body for POST /api/dataset/augment-agent."""

    job_id: str = Field(
        ...,
        description="UUID of an existing upload job whose dataset will be augmented.",
    )
    strategy: AugmentationStrategy = Field(
        default=AugmentationStrategy.MEDIUM,
        description="Augmentation intensity / profile.",
    )
    multiplier: float = Field(
        default=3.0,
        ge=1.1,
        le=20.0,
        description="Target dataset size = original × multiplier (ignored if target_size > 0).",
    )
    target_size: int = Field(
        default=0,
        ge=0,
        description="Exact target images per class. Overrides multiplier when > 0.",
    )
    target_px: int = Field(
        default=224,
        ge=32,
        le=1024,
        description="All output images will be resized to target_px × target_px.",
    )
    balance: bool = Field(
        default=True,
        description="If True, minority classes receive extra augmentations to match the majority.",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel worker threads.",
    )
    # Optional: directly supply a filesystem path instead of deriving from job_id
    input_dir: str | None = Field(
        default=None,
        description="Absolute path to ImageFolder root. If set, overrides job_id-based path.",
    )


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
