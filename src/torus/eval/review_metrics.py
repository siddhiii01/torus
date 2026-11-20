"""Review metrics (P9.5): how much human review a run needed, and where.

Metrics only: column keys, counts and durations. No field values, no document
text. Inputs are plain event tuples so this module does not depend on the
review package (``torus.review.metrics`` adapts stored decisions to them).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from torus.core.models import ExtractionResult
from torus.core.types import FieldStatus

__all__ = [
    "ColumnReviewMetrics",
    "DecisionEvent",
    "PackReviewInput",
    "ReviewMetrics",
    "aggregate_review_metrics",
]

_REVIEW_STATUSES = frozenset({FieldStatus.NEEDS_REVIEW, FieldStatus.LIKELY})


@dataclass(frozen=True)
class DecisionEvent:
    column_key: str
    action: str  # accept | correct | skip
    decided_at: datetime


@dataclass(frozen=True)
class PackReviewInput:
    result: ExtractionResult
    # Latest decision per item: what was finally decided.
    final: Sequence[DecisionEvent]
    # Every decision, for timing (re-decisions still cost reviewer time).
    history: Sequence[DecisionEvent]


class ColumnReviewMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str
    decided: int = Field(ge=0)
    corrected: int = Field(ge=0)
    correction_rate: float = Field(ge=0.0, le=1.0)


class ReviewMetrics(BaseModel):
    """Aggregate over reviewed packs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    packs: int = Field(ge=0)
    fields_needing_review: int = Field(ge=0)
    fields_needing_review_per_pack: float = Field(ge=0.0)
    accepted: int = Field(ge=0)
    corrected: int = Field(ge=0)
    skipped: int = Field(ge=0)
    undecided: int = Field(ge=0)
    correction_rate: float = Field(ge=0.0, le=1.0)
    timed_fields: int = Field(ge=0)
    seconds_per_field: float | None = Field(default=None, ge=0.0)
    max_gap_s: float = Field(gt=0.0)
    most_corrected: list[ColumnReviewMetrics] = Field(default_factory=list)


def _gaps(history: Sequence[DecisionEvent], max_gap_s: float) -> list[float]:
    """Seconds between consecutive decisions; long pauses are breaks, not work.

    The first decision of a pack has no known start, so it contributes no gap.
    """
    ordered = sorted(h.decided_at for h in history)
    deltas = [
        (b - a).total_seconds() for a, b in zip(ordered, ordered[1:], strict=False)
    ]
    return [d for d in deltas if 0.0 <= d <= max_gap_s]


def aggregate_review_metrics(
    packs: Sequence[PackReviewInput], *, max_gap_s: float
) -> ReviewMetrics:
    needing = 0
    undecided = 0
    counts = {"accept": 0, "correct": 0, "skip": 0}
    per_col: dict[str, list[int]] = {}  # key -> [decided, corrected]
    gaps: list[float] = []

    for pack in packs:
        queue_size = sum(
            1
            for row in pack.result.rows
            for fv in row.values()
            if fv.status in _REVIEW_STATUSES
        )
        needing += queue_size
        undecided += max(queue_size - len(pack.final), 0)
        for event in pack.final:
            counts[event.action] = counts.get(event.action, 0) + 1
            if event.action == "skip":
                continue
            col = per_col.setdefault(event.column_key, [0, 0])
            col[0] += 1
            col[1] += event.action == "correct"
        gaps.extend(_gaps(pack.history, max_gap_s))

    decided_total = counts["accept"] + counts["correct"]
    columns = sorted(
        (
            ColumnReviewMetrics(
                column_key=key,
                decided=d,
                corrected=c,
                correction_rate=c / d if d else 0.0,
            )
            for key, (d, c) in per_col.items()
        ),
        key=lambda m: (-m.corrected, -m.correction_rate, m.column_key),
    )
    return ReviewMetrics(
        packs=len(packs),
        fields_needing_review=needing,
        fields_needing_review_per_pack=needing / len(packs) if packs else 0.0,
        accepted=counts["accept"],
        corrected=counts["correct"],
        skipped=counts["skip"],
        undecided=undecided,
        correction_rate=counts["correct"] / decided_total if decided_total else 0.0,
        timed_fields=len(gaps),
        seconds_per_field=mean(gaps) if gaps else None,
        max_gap_s=max_gap_s,
        most_corrected=columns,
    )
