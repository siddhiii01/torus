"""Retention and deletion of job data (P11.5).

Deletes a job's work folder (extraction, Excel) and, when the input was an
upload, the stored copy. Files in the inbox are never touched: they belong to
the operator. The job row keeps counts only; audit events are retained.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from torus.logging import get_logger
from torus.service.audit import AuditLog
from torus.service.schemas import JobRecord
from torus.service.settings import ServiceSettings
from torus.service.store import JobStore

_log = get_logger(__name__)


def purge_job(
    job: JobRecord,
    settings: ServiceSettings,
    store: JobStore,
    audit: AuditLog | None,
    *,
    actor: str,
) -> None:
    work = (settings.jobs_dir / job.id).resolve()
    if work.is_relative_to(settings.jobs_dir.resolve()) and work.is_dir():
        shutil.rmtree(work)
    source = Path(job.source_path).resolve() if job.source_path else None
    if source and source.is_relative_to(settings.uploads_dir.resolve()):
        source.unlink(missing_ok=True)
    store.mark_purged(job.id)
    if audit is not None:
        audit.record(job.id, "data_purged", actor=actor)


def purge_expired(
    settings: ServiceSettings,
    store: JobStore,
    audit: AuditLog | None,
    *,
    now: float | None = None,
) -> int:
    now = time.time() if now is None else now
    cutoff = now - settings.config.retention.job_days * 86400.0
    expired = store.finished_before(cutoff)
    for job in expired:
        purge_job(job, settings, store, audit, actor="retention")
    _log.info("retention_purged", job_count=len(expired))
    return len(expired)
