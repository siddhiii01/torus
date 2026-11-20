"""Correction store (P9.2): every review decision, persisted under TORUS_DATA_DIR.

One append-only JSON-lines file per document. Appending (never rewriting) keeps
the full audit trail; the latest record for an item is the effective decision.
Records hold document content (values), so they live under ``TORUS_DATA_DIR``
and are never logged or committed. Errors name files and line numbers only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.core.models import Evidence
from torus.core.types import FieldStatus
from torus.review.queue import ReviewError, ReviewItem
from torus.review.server import Decision

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CORRECTIONS_SUBDIR = "corrections"


class CorrectionRecord(BaseModel):
    """One decision: which field, which crop(s), final value, who, when."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    document_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    column_key: str = Field(min_length=1)
    action: Literal["accept", "correct", "skip"]
    value: str | None = None
    suggestion: str | None = None
    status_before: FieldStatus
    crop_refs: list[Evidence] = Field(default_factory=list)
    reviewer: str = Field(min_length=1)
    decided_at: datetime


def data_dir() -> Path:
    raw = os.environ.get("TORUS_DATA_DIR", "").strip()
    if not raw:
        raise ReviewError("TORUS_DATA_DIR must be set to store corrections")
    path = Path(raw)
    if not path.is_dir():
        raise ReviewError("TORUS_DATA_DIR is not a directory")
    return path


def default_reviewer() -> str:
    """``TORUS_REVIEWER``, else the OS user name."""
    import getpass

    name = os.environ.get("TORUS_REVIEWER", "").strip()
    return name or getpass.getuser()


class CorrectionStore:
    """Persistent ``DecisionSink``; also reads decisions back for re-export."""

    def __init__(
        self,
        root: Path,
        reviewer: str,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not reviewer.strip():
            raise ReviewError("a reviewer name is required")
        self._root = root
        self._reviewer = reviewer.strip()
        self._clock = clock
        self._lock = Lock()

    @classmethod
    def from_env(cls, reviewer: str | None = None) -> CorrectionStore:
        return cls(
            data_dir() / CORRECTIONS_SUBDIR,
            reviewer if reviewer else default_reviewer(),
        )

    def _path(self, document_id: str) -> Path:
        # The id becomes a file name: allow-list it so it cannot escape the root.
        if not _SAFE_ID.match(document_id):
            raise ReviewError("document id is not a safe file name")
        return self._root / f"{document_id}.jsonl"

    def record(self, item: ReviewItem, decision: Decision) -> None:
        rec = CorrectionRecord(
            document_id=item.document_id,
            item_id=item.item_id,
            row=item.row,
            column_key=item.column_key,
            action=decision.action,
            value=decision.value,
            suggestion=item.suggestion,
            status_before=item.status,
            crop_refs=list(item.evidence),
            reviewer=self._reviewer,
            decided_at=self._clock(),
        )
        path = self._path(item.document_id)
        line = rec.model_dump_json() + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

    def load(self, document_id: str) -> list[CorrectionRecord]:
        """All records for a document, oldest first. Empty when none exist."""
        path = self._path(document_id)
        if not path.is_file():
            return []
        records: list[CorrectionRecord] = []
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(CorrectionRecord.model_validate_json(line))
                except (ValidationError, ValueError) as exc:
                    raise ReviewError(
                        f"corrupt correction record: {path.name} line {number}"
                    ) from exc
        return records

    def latest(self, document_id: str) -> dict[str, CorrectionRecord]:
        """Effective decision per item id: the last one recorded wins."""
        out: dict[str, CorrectionRecord] = {}
        for rec in self.load(document_id):
            out[rec.item_id] = rec
        return out
