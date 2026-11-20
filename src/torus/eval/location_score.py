"""Field-location scoring: did layout find the right box before reading? (P4.8).

Matches gold and predicted regions by ``(column_key, row_index)``, then
requires the same ``page_index`` and IoU ≥ threshold. Metrics carry column
keys and counts only — never OCR text or crops.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from torus.core.models import BBox
from torus.eval.metrics import f1_score, safe_div
from torus.eval.schema import FieldLocation, LocationsDocument

DEFAULT_LOCATE_IOU = 0.3


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union (kept local so eval does not import layout)."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class ColumnLocationMetrics(BaseModel):
    """Per-column location recall (column key and counts only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str = Field(min_length=1)
    expected: int = Field(ge=0, description="Gold locations for this column.")
    located: int = Field(ge=0, description="Gold locations matched by a prediction.")
    unmatched_pred: int = Field(
        ge=0,
        description="Predictions for this column with no gold match.",
    )
    recall: float = Field(
        ge=0.0,
        le=1.0,
        description="located / expected (Phase 4 acceptance uses this).",
    )
    precision: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    mean_iou: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean IoU over matched pairs (0 when none matched).",
    )

    @field_validator("column_key")
    @classmethod
    def _strip_column_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "column_key must be non-empty"
            raise ValueError(msg)
        return stripped


class LocationMetrics(BaseModel):
    """Aggregate field-location scores for a run or case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: int = Field(ge=0, description="Gold field locations scored.")
    located: int = Field(ge=0, description="Gold locations matched (IoU + page).")
    unmatched_pred: int = Field(
        ge=0,
        description="Predicted locations with no matching gold field.",
    )
    located_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="located / expected — primary Phase 4 gate (≥0.97 on dev).",
    )
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(
        ge=0.0,
        le=1.0,
        description="Same as located_rate; alias for report consistency.",
    )
    f1: float = Field(ge=0.0, le=1.0)
    mean_iou: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    per_column: list[ColumnLocationMetrics] = Field(default_factory=list)
    missed_columns: list[str] = Field(
        default_factory=list,
        description="Column keys with at least one unmatched gold location.",
    )


def score_locations(
    gold: LocationsDocument | Sequence[FieldLocation],
    pred: LocationsDocument | Sequence[FieldLocation],
    *,
    iou_threshold: float = DEFAULT_LOCATE_IOU,
) -> LocationMetrics:
    """Score predicted field boxes against gold locations."""
    gold_fields = list(gold.fields if isinstance(gold, LocationsDocument) else gold)
    pred_fields = list(pred.fields if isinstance(pred, LocationsDocument) else pred)

    if not gold_fields and not pred_fields:
        return empty_location_metrics(iou_threshold=iou_threshold)

    # Group predictions by (column_key, row_index) for greedy IoU matching.
    pred_by_key: dict[tuple[str, int], list[FieldLocation]] = defaultdict(list)
    for item in pred_fields:
        pred_by_key[(item.column_key, item.row_index)].append(item)

    located = 0
    iou_sum = 0.0
    used_pred_ids: set[int] = set()
    # Per-column tallies.
    col_expected: dict[str, int] = defaultdict(int)
    col_located: dict[str, int] = defaultdict(int)
    col_iou_sum: dict[str, float] = defaultdict(float)
    missed: set[str] = set()

    for gt in gold_fields:
        col_expected[gt.column_key] += 1
        candidates = pred_by_key.get((gt.column_key, gt.row_index), [])
        best_i = -1
        best_iou = 0.0
        for i, cand in enumerate(candidates):
            if id(cand) in used_pred_ids:
                continue
            if cand.page_index != gt.page_index:
                continue
            score = _bbox_iou(gt.bbox, cand.bbox)
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i >= 0 and best_iou >= iou_threshold:
            located += 1
            iou_sum += best_iou
            col_located[gt.column_key] += 1
            col_iou_sum[gt.column_key] += best_iou
            used_pred_ids.add(id(candidates[best_i]))
        else:
            missed.add(gt.column_key)

    unmatched_pred = 0
    col_unmatched: dict[str, int] = defaultdict(int)
    for item in pred_fields:
        if id(item) in used_pred_ids:
            continue
        unmatched_pred += 1
        col_unmatched[item.column_key] += 1

    expected = len(gold_fields)
    recall = safe_div(located, expected)
    precision = safe_div(located, located + unmatched_pred)
    mean_iou = safe_div(iou_sum, located) if located else 0.0

    keys = sorted(set(col_expected) | set(col_unmatched) | set(col_located))
    per_column: list[ColumnLocationMetrics] = []
    for key in keys:
        exp = col_expected.get(key, 0)
        loc = col_located.get(key, 0)
        up = col_unmatched.get(key, 0)
        rec = safe_div(loc, exp)
        prec = safe_div(loc, loc + up)
        per_column.append(
            ColumnLocationMetrics(
                column_key=key,
                expected=exp,
                located=loc,
                unmatched_pred=up,
                recall=round(rec, 4),
                precision=round(prec, 4),
                f1=round(f1_score(prec, rec), 4),
                mean_iou=round(safe_div(col_iou_sum.get(key, 0.0), loc), 4)
                if loc
                else 0.0,
            )
        )

    return LocationMetrics(
        expected=expected,
        located=located,
        unmatched_pred=unmatched_pred,
        located_rate=round(recall, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1_score(precision, recall), 4),
        mean_iou=round(mean_iou, 4),
        iou_threshold=iou_threshold,
        per_column=per_column,
        missed_columns=sorted(missed),
    )


def aggregate_locations(
    scores: Sequence[LocationMetrics],
    *,
    iou_threshold: float | None = None,
) -> LocationMetrics:
    """Sum several case-level location scores into one aggregate."""
    if not scores:
        thr = DEFAULT_LOCATE_IOU if iou_threshold is None else iou_threshold
        return empty_location_metrics(iou_threshold=thr)

    thr = iou_threshold if iou_threshold is not None else scores[0].iou_threshold
    # Rebuild by re-scoring flattened per-column counts is lossy for mean_iou;
    # weight mean_iou by located counts instead.
    expected = located = unmatched_pred = 0
    iou_weight = 0.0
    col_expected: dict[str, int] = defaultdict(int)
    col_located: dict[str, int] = defaultdict(int)
    col_unmatched: dict[str, int] = defaultdict(int)
    col_iou_weight: dict[str, float] = defaultdict(float)
    missed: set[str] = set()

    for score in scores:
        expected += score.expected
        located += score.located
        unmatched_pred += score.unmatched_pred
        iou_weight += score.mean_iou * score.located
        missed.update(score.missed_columns)
        for col in score.per_column:
            col_expected[col.column_key] += col.expected
            col_located[col.column_key] += col.located
            col_unmatched[col.column_key] += col.unmatched_pred
            col_iou_weight[col.column_key] += col.mean_iou * col.located

    recall = safe_div(located, expected)
    precision = safe_div(located, located + unmatched_pred)
    mean_iou = safe_div(iou_weight, located) if located else 0.0

    keys = sorted(set(col_expected) | set(col_unmatched) | set(col_located))
    per_column: list[ColumnLocationMetrics] = []
    for key in keys:
        exp = col_expected.get(key, 0)
        loc = col_located.get(key, 0)
        up = col_unmatched.get(key, 0)
        rec = safe_div(loc, exp)
        prec = safe_div(loc, loc + up)
        per_column.append(
            ColumnLocationMetrics(
                column_key=key,
                expected=exp,
                located=loc,
                unmatched_pred=up,
                recall=round(rec, 4),
                precision=round(prec, 4),
                f1=round(f1_score(prec, rec), 4),
                mean_iou=round(safe_div(col_iou_weight.get(key, 0.0), loc), 4)
                if loc
                else 0.0,
            )
        )

    return LocationMetrics(
        expected=expected,
        located=located,
        unmatched_pred=unmatched_pred,
        located_rate=round(recall, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1_score(precision, recall), 4),
        mean_iou=round(mean_iou, 4),
        iou_threshold=thr,
        per_column=per_column,
        missed_columns=sorted(missed),
    )


def empty_location_metrics(
    *, iou_threshold: float = DEFAULT_LOCATE_IOU
) -> LocationMetrics:
    """Zeroed metrics when no locations were scored."""
    return LocationMetrics(
        expected=0,
        located=0,
        unmatched_pred=0,
        located_rate=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        mean_iou=0.0,
        iou_threshold=iou_threshold,
        per_column=[],
        missed_columns=[],
    )
