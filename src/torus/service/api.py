"""FastAPI application: submit, status, result download. Metadata only."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from torus.ingest.formats import SUPPORTED_SUFFIXES
from torus.logging import get_logger
from torus.service.audit import AuditEvent, AuditLog
from torus.service.audit_review import ReviewAction, ReviewSource
from torus.service.auth import make_auth_dependency
from torus.service.errors import ServiceError, UnsafePathError
from torus.service.intake import enqueue_path, resolve_under
from torus.service.monitoring import Metrics, collect_metrics, pending_spot_checks
from torus.service.retention import purge_job
from torus.service.schemas import (
    BatchView,
    JobStatus,
    JobView,
    SubmitPathRequest,
    SubmitResponse,
)
from torus.service.settings import ServiceSettings
from torus.service.store import JobStore, new_id

_log = get_logger(__name__)
_TEMPLATE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _template(settings: ServiceSettings, requested: str | None) -> str:
    template_id = requested or settings.config.default_template_id
    if not _TEMPLATE_ID.match(template_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "bad template_id")
    return template_id


class SpotCheckBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_key: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    correct: bool
    reviewer: str = Field(pattern=r"^[A-Za-z0-9_.@-]{1,64}$")


class AuditReport(BaseModel):
    job_id: str
    events: list[AuditEvent]
    review_actions: list[ReviewAction]


def create_app(
    settings: ServiceSettings,
    store: JobStore,
    token: str,
    *,
    audit: AuditLog | None = None,
    review_source: ReviewSource | None = None,
    health: Callable[[], dict[str, bool]] | None = None,
) -> FastAPI:
    audit = audit or AuditLog(settings.db_path)
    app = FastAPI(title="Torus service", version="0.1.0")
    auth = [Depends(make_auth_dependency(token))]
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/healthz")
    def healthz() -> dict[str, int | str]:
        return {"status": "ok", "queued": store.queue_length()}

    @app.post("/v1/jobs", response_model=SubmitResponse, dependencies=auth)
    def submit_path(body: SubmitPathRequest) -> SubmitResponse:
        """Queue a file or folder located under the data directory."""
        template_id = _template(settings, body.template_id)
        try:
            target = resolve_under(settings.data_dir, body.path)
            batch_id, jobs = enqueue_path(
                store, target, template_id=template_id, audit=audit
            )
        except UnsafePathError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except ServiceError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        _log.info("jobs_submitted", job_count=len(jobs), template_id=template_id)
        return SubmitResponse(batch_id=batch_id, job_ids=[j.id for j in jobs])

    @app.put(
        "/v1/jobs/upload",
        response_model=SubmitResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=auth,
    )
    async def upload(
        request: Request,
        suffix: str = Query(description="File type, for example .pdf"),
        template_id: str | None = None,
    ) -> SubmitResponse:
        """Upload one file as the raw request body (no multipart dependency)."""
        suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported")
        template = _template(settings, template_id)
        limit = settings.config.api.max_upload_bytes
        dest = settings.uploads_dir / f"{new_id()}{suffix}"
        received = 0
        try:
            with dest.open("wb") as handle:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > limit:
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "too large"
                        )
                    handle.write(chunk)
            if received == 0:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty body")
            batch_id, jobs = enqueue_path(
                store, dest, template_id=template, audit=audit
            )
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return SubmitResponse(batch_id=batch_id, job_ids=[j.id for j in jobs])

    @app.get("/v1/jobs/{job_id}", response_model=JobView, dependencies=auth)
    def job_status(job_id: str) -> JobView:
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        return JobView.from_record(rec)

    @app.get("/v1/batches/{batch_id}", response_model=BatchView, dependencies=auth)
    def batch_status(batch_id: str) -> BatchView:
        records = store.list_batch(batch_id)
        if not records:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown batch")
        counts = Counter(r.status.value for r in records)
        return BatchView(
            batch_id=batch_id,
            total=len(records),
            counts=dict(counts),
            jobs=[JobView.from_record(r) for r in records],
        )

    @app.get("/v1/jobs/{job_id}/result", dependencies=auth)
    def job_result(job_id: str) -> FileResponse:
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        if rec.status is not JobStatus.SUCCEEDED or not rec.result_path:
            raise HTTPException(status.HTTP_409_CONFLICT, "result not ready")
        path = Path(rec.result_path).resolve()
        if not path.is_relative_to(settings.jobs_dir.resolve()) or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "result missing")
        return FileResponse(path, media_type=_XLSX, filename=f"{job_id}.xlsx")

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=auth)
    def cancel(job_id: str) -> dict[str, bool]:
        if store.get(job_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        return {"cancelled": store.cancel(job_id)}

    @app.get("/v1/jobs/{job_id}/audit", response_model=AuditReport, dependencies=auth)
    def job_audit(job_id: str) -> AuditReport:
        """Models, config hash, timings, statuses and reviewer actions. No values."""
        events = audit.events(job_id)
        if not events:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        actions: list[ReviewAction] = []
        doc_ids = {
            str(e.data["document_id"]) for e in events if "document_id" in e.data
        }
        if review_source is not None:
            for doc_id in sorted(doc_ids):
                actions.extend(review_source(doc_id))
        return AuditReport(job_id=job_id, events=events, review_actions=actions)

    @app.delete("/v1/jobs/{job_id}", dependencies=auth)
    def delete_job_data(job_id: str) -> dict[str, bool]:
        """Delete a finished job's files now (retention override)."""
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        if rec.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(status.HTTP_409_CONFLICT, "job is not finished")
        purge_job(rec, settings, store, audit, actor="api")
        return {"deleted": True}

    @app.get("/v1/metrics", response_model=Metrics, dependencies=auth)
    def metrics() -> Metrics:
        return collect_metrics(
            store,
            audit,
            settings.config.monitoring,
            health() if health else {},
        )

    @app.get("/v1/spot-checks/pending", dependencies=auth)
    def spot_checks_pending() -> dict[str, list[str]]:
        return {
            "job_ids": pending_spot_checks(store, audit, settings.config.monitoring)
        }

    @app.post("/v1/jobs/{job_id}/spot-check", dependencies=auth)
    def record_spot_check(job_id: str, body: SpotCheckBody) -> dict[str, bool]:
        """A person re-checked one confirmed value: was it right?"""
        rec = store.get(job_id)
        if rec is None or rec.status is not JobStatus.SUCCEEDED:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no finished job")
        audit.record(
            job_id,
            "spot_check",
            actor=f"reviewer:{body.reviewer}",
            data={"field_key": body.field_key, "correct": body.correct},
        )
        return {"recorded": True}

    return app
