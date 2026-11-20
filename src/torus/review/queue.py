"""Review queue (P9.1): which fields a person must look at, and what to show.

Pure data. An item bundles everything the reviewer needs so they never open the
PDF separately: the suggested value, every reader's reading of the evidence
regions, validator messages and the evidence boxes. Reading text is document
content: it is returned to the local UI only and never logged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from torus.core.errors import TorusError
from torus.core.models import (
    Evidence,
    ExtractionResult,
    Reading,
    ValidationResult,
)
from torus.core.types import FieldStatus


class ReviewError(TorusError):
    """Bad review bundle, unknown item, or invalid decision."""


# needs_review before likely; confirmed / not_found never enter the queue.
_QUEUE_ORDER: dict[FieldStatus, int] = {
    FieldStatus.NEEDS_REVIEW: 0,
    FieldStatus.LIKELY: 1,
}


class ReadingView(BaseModel):
    """One reader's reading of one evidence region."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region_id: str
    reader_name: str
    text: str
    raw_score: float | None = None
    error: str | None = None


class ReviewItem(BaseModel):
    """One field awaiting a human decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    document_id: str
    row: int = Field(ge=0)
    column_key: str
    status: FieldStatus
    confidence: float = Field(ge=0.0, le=1.0)
    suggestion: str | None = None
    reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    readings: list[ReadingView] = Field(default_factory=list)
    validators: list[ValidationResult] = Field(default_factory=list)


def make_item_id(row: int, column_key: str) -> str:
    return f"{row}:{column_key}"


def parse_item_id(item_id: str) -> tuple[int, str]:
    """Strict inverse of :func:`make_item_id`; raises ``ReviewError``."""
    head, sep, column_key = item_id.partition(":")
    if not sep or not head.isdigit() or not column_key:
        raise ReviewError("malformed item id")
    return int(head), column_key


def build_queue(
    result: ExtractionResult,
    readings: Sequence[Reading] = (),
    *,
    statuses: Iterable[FieldStatus] = (
        FieldStatus.NEEDS_REVIEW,
        FieldStatus.LIKELY,
    ),
) -> list[ReviewItem]:
    """Items for every field whose status is in ``statuses``.

    Order: ``needs_review`` first, then ``likely``; inside each, lowest
    confidence first, ties broken by row then column key so the order is
    deterministic. Readings are matched to an item by its evidence region ids.
    """
    wanted = {s for s in statuses if s in _QUEUE_ORDER}
    by_region: dict[str, list[Reading]] = {}
    for reading in readings:
        by_region.setdefault(reading.region_id, []).append(reading)

    items: list[ReviewItem] = []
    for row_index, row in enumerate(result.rows):
        for column_key, field in row.items():
            if field.status not in wanted:
                continue
            views: list[ReadingView] = []
            seen: set[str] = set()
            for ev in field.evidence:
                if ev.region_id in seen:
                    continue
                seen.add(ev.region_id)
                views.extend(
                    ReadingView(
                        region_id=r.region_id,
                        reader_name=r.reader_name,
                        text=r.text,
                        raw_score=r.raw_score,
                        error=r.error,
                    )
                    for r in by_region.get(ev.region_id, [])
                )
            items.append(
                ReviewItem(
                    item_id=make_item_id(row_index, column_key),
                    document_id=result.document_id,
                    row=row_index,
                    column_key=column_key,
                    status=field.status,
                    confidence=field.confidence,
                    suggestion=field.value,
                    reason=field.reason,
                    evidence=list(field.evidence),
                    readings=views,
                    validators=list(field.validator_results),
                )
            )
    items.sort(
        key=lambda i: (_QUEUE_ORDER[i.status], i.confidence, i.row, i.column_key)
    )
    return items
