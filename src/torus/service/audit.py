"""Append-only, tamper-evident audit log (P11.3). Metadata only.

Each event stores the SHA-256 of the previous event, so editing or deleting a
row breaks the chain (``verify`` finds it). Only allow-listed keys with scalar
(or flat scalar-map) values are accepted; anything else raises instead of being
silently stored, so document content cannot enter the log by accident.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from torus.service.errors import ServiceError

ALLOWED_DATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action",
        "attempt",
        "config_hash",
        "content_hash",
        "correct",
        "document_id",
        "duration_ms",
        "error_code",
        "field_key",
        "model_versions",
        "status",
        "status_before",
        "status_counts",
        "stage",
        "template_id",
        "timings_ms",
        "worker_id",
    }
)
_MAX_STR: Final[int] = 256
_Scalar = (str, int, float, bool, type(None))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    job_id TEXT NOT NULL,
    event TEXT NOT NULL,
    actor TEXT NOT NULL,
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_job ON audit_events (job_id, seq);
"""
_GENESIS = "0" * 64


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    ts: float
    job_id: str
    event: str
    actor: str
    data: dict[str, Any]
    hash: str


def _check_scalar(key: str, value: object) -> None:
    if not isinstance(value, _Scalar):
        raise ServiceError(f"audit field {key!r} must be a scalar")
    if isinstance(value, str) and len(value) > _MAX_STR:
        raise ServiceError(f"audit field {key!r} is too long for metadata")


def validate_data(data: Mapping[str, object]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in ALLOWED_DATA_KEYS:
            raise ServiceError(f"audit field {key!r} is not allowed")
        if isinstance(value, Mapping):
            for k, v in value.items():
                _check_scalar(f"{key}.{k}", v)
            out[key] = dict(value)
        else:
            _check_scalar(key, value)
            out[key] = value
    return out


def _digest(
    prev: str, ts: float, job_id: str, event: str, actor: str, data: str
) -> str:
    payload = json.dumps([prev, ts, job_id, event, actor, data], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLog:
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
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    def record(
        self,
        job_id: str,
        event: str,
        *,
        actor: str,
        data: Mapping[str, object] | None = None,
    ) -> None:
        clean = json.dumps(validate_data(data or {}), sort_keys=True)
        ts = self._clock()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev = row[0] if row else _GENESIS
            conn.execute(
                "INSERT INTO audit_events "
                "(ts, job_id, event, actor, data, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    ts,
                    job_id,
                    event,
                    actor,
                    clean,
                    prev,
                    _digest(prev, ts, job_id, event, actor, clean),
                ),
            )
            conn.execute("COMMIT")

    def events(self, job_id: str | None = None) -> list[AuditEvent]:
        query = "SELECT seq, ts, job_id, event, actor, data, hash FROM audit_events"
        args: tuple[str, ...] = ()
        if job_id is not None:
            query += " WHERE job_id=?"
            args = (job_id,)
        with self._conn() as conn:
            rows = conn.execute(query + " ORDER BY seq", args).fetchall()
        return [
            AuditEvent(
                seq=r[0],
                ts=r[1],
                job_id=r[2],
                event=r[3],
                actor=r[4],
                data=json.loads(r[5]),
                hash=r[6],
            )
            for r in rows
        ]

    def verify(self) -> int | None:
        """Return the first broken ``seq`` or None when the chain is intact."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT seq, ts, job_id, event, actor, data, prev_hash, hash "
                "FROM audit_events ORDER BY seq"
            ).fetchall()
        prev = _GENESIS
        for seq, ts, job_id, event, actor, data, prev_hash, digest in rows:
            if prev_hash != prev or digest != _digest(
                prev, ts, job_id, event, actor, data
            ):
                return int(seq)
            prev = digest
        return None
