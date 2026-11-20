"""Interface between the service and the extraction pipeline.

The service depends only on this protocol, never on pipeline internals. The
real pipeline (Phases 6-8) provides an implementation; tests use a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class JobContext:
    """Everything a stage needs. Paths only; no document content."""

    job_id: str
    source_path: Path
    template_id: str
    work_dir: Path
    # Counts only (rows, confirmed, needs_review ...), filled by stages.
    summary: dict[str, int] = field(default_factory=dict)
    # Audit metadata a stage may set: document_id, config_hash (str),
    # model_versions (dict[str, str]), timings_ms (dict[str, float]).
    meta: dict[str, Any] = field(default_factory=dict)


class PipelineRunner(Protocol):
    """Runs one input file through the pipeline, stage by stage.

    A stage must be safe to re-run after a failure (idempotent for the same
    ``work_dir``), because the worker retries stages independently.
    """

    def stages(self) -> list[str]: ...

    def run_stage(self, stage: str, ctx: JobContext) -> None: ...

    def result_path(self, ctx: JobContext) -> Path:
        """Path of the finished Excel file inside ``ctx.work_dir``."""
        ...
