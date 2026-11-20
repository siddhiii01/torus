"""Reviewer actions for the audit report, without field values.

Corrections hold the corrected text, which is document content. The audit
report only gets who did what to which column, never the value.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from torus.core.types import FieldStatus
from torus.review.store import CorrectionStore


class ReviewAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer: str
    action: str
    column_key: str
    status_before: FieldStatus
    decided_at: datetime


# document_id -> actions
ReviewSource = Callable[[str], list[ReviewAction]]


def review_source_from_store(store: CorrectionStore) -> ReviewSource:
    """Project a ``CorrectionStore`` onto value-free :class:`ReviewAction`s."""

    def source(document_id: str) -> list[ReviewAction]:
        return [
            ReviewAction(
                reviewer=r.reviewer,
                action=r.action,
                column_key=r.column_key,
                status_before=r.status_before,
                decided_at=r.decided_at,
            )
            for r in store.load(document_id)
        ]

    return source
