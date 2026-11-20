"""Operational metrics and silent-error spot checks (P11.4). Counts only."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Final

from pydantic import BaseModel, ConfigDict

from torus.service.audit import AuditLog
from torus.service.schemas import JobRecord, JobStatus
from torus.service.settings import ModelServerConfig, MonitoringConfig
from torus.service.store import JobStore
from torus.service.supervisor import http_health

_FIELD_STATUSES: Final[tuple[str, ...]] = (
    "confirmed",
    "likely",
    "needs_review",
    "not_found",
)


class SpotCheckStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    checked: int
    incorrect: int
    # Share of spot-checked confirmed values that were wrong; None until checked.
    silent_error_rate: float | None


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_s: float
    queue_length: int
    running: int
    status_counts: dict[str, int]
    succeeded_in_window: int
    failed_in_window: int
    throughput_per_hour: float
    median_job_seconds: float | None
    review_rate: float | None
    fields_in_window: dict[str, int]
    spot_checks: SpotCheckStats
    model_health: dict[str, bool]


def model_health(
    servers: list[ModelServerConfig],
    probe: Callable[[str], bool] = http_health,
) -> dict[str, bool]:
    return {s.name: probe(s.health_url) for s in servers}


def is_spot_check_sample(job_id: str, rate: float) -> bool:
    """Deterministic sampling: the same job is always in or out of the sample."""
    bucket = int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:8], "big")
    return bucket / 2**64 < rate


def _confirmed(rec: JobRecord) -> int:
    return rec.summary.get("confirmed", 0)


def pending_spot_checks(
    store: JobStore, audit: AuditLog, cfg: MonitoringConfig, *, since: float = 0.0
) -> list[str]:
    """Succeeded jobs in the sample, with confirmed values, not yet checked."""
    done = {e.job_id for e in audit.events() if e.event == "spot_check"}
    return [
        r.id
        for r in store.updated_since(since)
        if r.status is JobStatus.SUCCEEDED
        and _confirmed(r) > 0
        and r.id not in done
        and is_spot_check_sample(r.id, cfg.spot_check_rate)
    ]


def spot_check_stats(audit: AuditLog) -> SpotCheckStats:
    checks = [e for e in audit.events() if e.event == "spot_check"]
    wrong = sum(1 for e in checks if e.data.get("correct") is False)
    return SpotCheckStats(
        checked=len(checks),
        incorrect=wrong,
        silent_error_rate=(wrong / len(checks)) if checks else None,
    )


def collect_metrics(
    store: JobStore,
    audit: AuditLog,
    cfg: MonitoringConfig,
    health: dict[str, bool],
    *,
    now: float | None = None,
) -> Metrics:
    now = time.time() if now is None else now
    recent = store.updated_since(now - cfg.window_s)
    ok = [r for r in recent if r.status is JobStatus.SUCCEEDED]
    failed = [r for r in recent if r.status is JobStatus.FAILED]
    fields = {k: sum(r.summary.get(k, 0) for r in ok) for k in _FIELD_STATUSES}
    total = sum(fields.values())
    durations = sorted(r.updated_at - r.created_at for r in ok)
    counts = store.counts_by_status()
    return Metrics(
        window_s=cfg.window_s,
        queue_length=counts.get("queued", 0),
        running=counts.get("running", 0),
        status_counts=counts,
        succeeded_in_window=len(ok),
        failed_in_window=len(failed),
        throughput_per_hour=len(ok) * 3600.0 / cfg.window_s,
        median_job_seconds=durations[len(durations) // 2] if durations else None,
        review_rate=(fields["needs_review"] / total) if total else None,
        fields_in_window=fields,
        spot_checks=spot_check_stats(audit),
        model_health=health,
    )
