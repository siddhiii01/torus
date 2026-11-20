"""Request/response and record models for the service."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class JobRecord(BaseModel):
    """One queued unit of work: a single input file. Metadata only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    batch_id: str
    source_path: str
    content_hash: str
    template_id: str
    status: JobStatus
    stage: str | None = None
    attempts: int = 0
    error_code: str | None = None
    created_at: float
    updated_at: float
    next_run_at: float = 0.0
    lease_until: float | None = None
    worker_id: str | None = None
    result_path: str | None = None
    summary: dict[str, int] = Field(default_factory=dict)


class SubmitPathRequest(BaseModel):
    """Submit a file or folder already under the data directory."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Relative to TORUS_DATA_DIR")
    template_id: str | None = None


class SubmitResponse(BaseModel):
    batch_id: str
    job_ids: list[str]


class JobView(BaseModel):
    """Public job status. Never includes server paths or document content."""

    id: str
    batch_id: str
    status: JobStatus
    stage: str | None
    attempts: int
    error_code: str | None
    template_id: str
    created_at: float
    updated_at: float
    result_ready: bool
    summary: dict[str, int]

    @classmethod
    def from_record(cls, rec: JobRecord) -> JobView:
        return cls(
            id=rec.id,
            batch_id=rec.batch_id,
            status=rec.status,
            stage=rec.stage,
            attempts=rec.attempts,
            error_code=rec.error_code,
            template_id=rec.template_id,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
            result_ready=rec.status is JobStatus.SUCCEEDED and bool(rec.result_path),
            summary=rec.summary,
        )


class BatchView(BaseModel):
    batch_id: str
    total: int
    counts: dict[str, int]
    jobs: list[JobView]
