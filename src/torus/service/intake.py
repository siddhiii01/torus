"""Turn files/folders into queued jobs. Shared by the API and the watcher."""

from __future__ import annotations

from pathlib import Path

from torus.ingest.formats import SUPPORTED_SUFFIXES
from torus.ingest.hashing import file_sha256
from torus.service.audit import AuditLog
from torus.service.errors import ServiceError, UnsafePathError
from torus.service.schemas import JobRecord
from torus.service.store import JobStore, new_id


def resolve_under(base: Path, relative: str) -> Path:
    """Resolve ``relative`` inside ``base``; reject anything that escapes it."""
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise UnsafePathError("path escapes the data directory")
    return candidate


def list_supported(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def enqueue_path(
    store: JobStore,
    path: Path,
    *,
    template_id: str,
    skip_known: bool = False,
    audit: AuditLog | None = None,
    actor: str = "api",
) -> tuple[str, list[JobRecord]]:
    """Queue one file, or every supported file in a folder, as one batch."""
    batch_id = new_id()
    if path.is_dir():
        files = list_supported(path)
    elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
        files = [path]
    else:
        raise ServiceError("path is not a supported file or a folder")
    if not files:
        raise ServiceError("folder contains no supported files")
    jobs: list[JobRecord] = []
    for file in files:
        digest = file_sha256(file)
        if skip_known and store.exists_hash(digest, template_id):
            continue
        job = store.enqueue(
            source_path=str(file),
            content_hash=digest,
            template_id=template_id,
            batch_id=batch_id,
        )
        jobs.append(job)
        if audit is not None:
            audit.record(
                job.id,
                "job_submitted",
                actor=actor,
                data={"template_id": template_id, "content_hash": digest},
            )
    return batch_id, jobs
