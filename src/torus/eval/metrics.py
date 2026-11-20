"""Evaluation metric result models (no document text)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from torus.core.types import FieldStatus


class RowAlignmentMetrics(BaseModel):
    """How well predicted rows matched gold rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gold: int = Field(ge=0)
    pred: int = Field(ge=0)
    aligned: int = Field(ge=0)
    unmatched_gold: int = Field(ge=0)
    unmatched_pred: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class FieldMetrics(BaseModel):
    """Per-column scores on aligned rows only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    compared: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    silent_errors: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)


class MicroMetrics(BaseModel):
    """Micro-averaged field metrics across scored columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    compared: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class StatusRates(BaseModel):
    """Share of scored cells by ``FieldStatus`` (aligned rows with a status)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    counts: dict[str, int] = Field(default_factory=dict)
    rates: dict[str, float] = Field(default_factory=dict)
    automatic_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of status-bearing cells marked confirmed (no human).",
    )
    review_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of status-bearing cells marked needs_review.",
    )


class CaseScore(BaseModel):
    """Full score for one expected vs predicted document.

    Field-level detail lists omit gold/pred text — only keys and booleans.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str | None = None
    template_id: str
    rows: RowAlignmentMetrics
    fields_micro: MicroMetrics
    per_field: dict[str, FieldMetrics] = Field(default_factory=dict)
    silent_errors: int = Field(ge=0)
    status_rates: StatusRates
    mismatches: list[dict[str, str | bool]] = Field(
        default_factory=list,
        description="Metadata-only mismatch records (field key, match flag).",
    )


def empty_status_counts() -> dict[str, int]:
    return {status.value: 0 for status in FieldStatus}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
