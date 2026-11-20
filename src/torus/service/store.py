"""SQLite-backed job store and queue.

SQLite (stdlib) is shared safely between the API process and worker processes.
Claiming a job is a single atomic UPDATE, so two workers never take the same
job. The store holds metadata only: paths, hashes, statuses, counts.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from torus.service.schemas import JobRecord, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    template_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    next_run_at REAL NOT NULL DEFAULT 0,
    lease_until REAL,
    worker_id TEXT,
    result_path TEXT,
    summary TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, next_run_at);
CREATE INDEX IF NOT EXISTS jobs_batch ON jobs (batch_id);
CREATE INDEX IF NOT EXISTS jobs_hash ON jobs (content_hash, template_id);
"""

_COLUMNS = (
    "id, batch_id, source_path, content_hash, template_id, status, stage, "
    "attempts, error_code, created_at, updated_at, next_run_at, lease_until, "
    "worker_id, result_path, summary"
)


def new_id() -> str:
    return uuid.uuid4().hex


class JobStore:
    def __init__(
        self, db_path: Path, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._path = db_path
        self._clock = clock
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(row: tuple[object, ...]) -> JobRecord:
        import json

        (
            id_, batch_id, source_path, content_hash, template_id, status, stage,
            attempts, error_code, created_at, updated_at, next_run_at,
            lease_until, worker_id, result_path, summary,
        ) = row  # fmt: skip
        return JobRecord(
            id=str(id_),
            batch_id=str(batch_id),
            source_path=str(source_path),
            content_hash=str(content_hash),
            template_id=str(template_id),
            status=JobStatus(str(status)),
            stage=None if stage is None else str(stage),
            attempts=int(attempts),  # type: ignore[call-overload]
            error_code=None if error_code is None else str(error_code),
            created_at=float(created_at),  # type: ignore[arg-type]
            updated_at=float(updated_at),  # type: ignore[arg-type]
            next_run_at=float(next_run_at),  # type: ignore[arg-type]
            lease_until=None if lease_until is None else float(lease_until),  # type: ignore[arg-type]
            worker_id=None if worker_id is None else str(worker_id),
            result_path=None if result_path is None else str(result_path),
            summary=json.loads(str(summary)),
        )

    # -- submit / read ------------------------------------------------------

    def enqueue(
        self,
        *,
        source_path: str,
        content_hash: str,
        template_id: str,
        batch_id: str | None = None,
    ) -> JobRecord:
        now = self._clock()
        job_id = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, batch_id, source_path, content_hash, "
                "template_id, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    batch_id or job_id,
                    source_path,
                    content_hash,
                    template_id,
                    JobStatus.QUEUED.value,
                    now,
                    now,
                ),
            )
        return self.get_required(job_id)

    def get(self, job_id: str) -> JobRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return None if row is None else self._row(row)

    def get_required(self, job_id: str) -> JobRecord:
        rec = self.get(job_id)
        if rec is None:
            raise KeyError(job_id)
        return rec

    def list_batch(self, batch_id: str) -> list[JobRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE batch_id=? ORDER BY created_at, id",
                (batch_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def exists_hash(self, content_hash: str, template_id: str) -> bool:
        """True if a non-failed, non-cancelled job already covers this content."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE content_hash=? AND template_id=? "
                "AND status NOT IN (?,?) LIMIT 1",
                (
                    content_hash,
                    template_id,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            ).fetchone()
        return row is not None

    def queue_length(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status=?", (JobStatus.QUEUED.value,)
            ).fetchone()
        return int(row[0])

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Running jobs are not interrupted."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=? AND status=?",
                (
                    JobStatus.CANCELLED.value,
                    self._clock(),
                    job_id,
                    JobStatus.QUEUED.value,
                ),
            )
        return cur.rowcount == 1

    # -- monitoring and retention (P11.4, P11.5) ------------------------------

    def counts_by_status(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall()
        return {str(s): int(n) for s, n in rows}

    def updated_since(self, since: float) -> list[JobRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE updated_at>=? ORDER BY updated_at",
                (since,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def finished_before(self, cutoff: float) -> list[JobRecord]:
        """Terminal jobs not touched since ``cutoff`` whose data is still held."""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE status IN (?,?,?) "
                "AND updated_at<? AND source_path!=''",
                (
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    cutoff,
                ),
            ).fetchall()
        return [self._row(r) for r in rows]

    def mark_purged(self, job_id: str) -> None:
        """Forget file locations after the data was deleted; keep counts."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET source_path='', result_path=NULL WHERE id=? "
                "AND status IN (?,?,?)",
                (
                    job_id,
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            )

    # -- worker side (P11.2) ------------------------------------------------

    def claim_next(self, worker_id: str, lease_s: float) -> JobRecord | None:
        """Atomically move the oldest runnable queued job to running."""
        now = self._clock()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM jobs WHERE status=? AND next_run_at<=? "
                "ORDER BY created_at, id LIMIT 1",
                (JobStatus.QUEUED.value, now),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status=?, worker_id=?, lease_until=?, "
                "attempts=attempts+1, updated_at=?, stage=NULL, error_code=NULL "
                "WHERE id=?",
                (JobStatus.RUNNING.value, worker_id, now + lease_s, now, row[0]),
            )
            conn.execute("COMMIT")
        return self.get_required(str(row[0]))

    def heartbeat(
        self, job_id: str, worker_id: str, lease_s: float, stage: str | None
    ) -> bool:
        """Extend the lease; False means the job was taken away (lease lost)."""
        now = self._clock()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET lease_until=?, stage=?, updated_at=? "
                "WHERE id=? AND worker_id=? AND status=?",
                (now + lease_s, stage, now, job_id, worker_id, JobStatus.RUNNING.value),
            )
        return cur.rowcount == 1

    def complete(
        self,
        job_id: str,
        worker_id: str,
        *,
        result_path: str,
        summary: dict[str, int],
    ) -> bool:
        import json

        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, result_path=?, summary=?, stage=NULL, "
                "lease_until=NULL, updated_at=? "
                "WHERE id=? AND worker_id=? AND status=?",
                (
                    JobStatus.SUCCEEDED.value,
                    result_path,
                    json.dumps(summary),
                    self._clock(),
                    job_id,
                    worker_id,
                    JobStatus.RUNNING.value,
                ),
            )
        return cur.rowcount == 1

    def fail(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str,
        retry_at: float | None,
    ) -> bool:
        """Mark failed, or re-queue for ``retry_at`` when a retry is allowed."""
        status = JobStatus.FAILED if retry_at is None else JobStatus.QUEUED
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, error_code=?, next_run_at=?, "
                "lease_until=NULL, worker_id=NULL, updated_at=? "
                "WHERE id=? AND worker_id=? AND status=?",
                (
                    status.value,
                    error_code,
                    retry_at or 0.0,
                    self._clock(),
                    job_id,
                    worker_id,
                    JobStatus.RUNNING.value,
                ),
            )
        return cur.rowcount == 1

    def requeue_expired(self, max_attempts: int) -> list[str]:
        """Recover jobs whose worker died (lease expired).

        Jobs with attempts left go back to the queue; the rest are failed so a
        poison job cannot loop forever.
        """
        now = self._clock()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, attempts FROM jobs WHERE status=? AND lease_until<?",
                (JobStatus.RUNNING.value, now),
            ).fetchall()
            for job_id, attempts in rows:
                dead = attempts >= max_attempts
                conn.execute(
                    "UPDATE jobs SET status=?, error_code=?, lease_until=NULL, "
                    "worker_id=NULL, updated_at=? WHERE id=?",
                    (
                        (JobStatus.FAILED if dead else JobStatus.QUEUED).value,
                        "lease_expired",
                        now,
                        job_id,
                    ),
                )
            conn.execute("COMMIT")
        return [str(r[0]) for r in rows]
