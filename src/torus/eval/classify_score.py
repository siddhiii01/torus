"""Page-type classification scoring (P3.5).

Produces per-type precision/recall/F1, a confusion matrix, and a
misclassification tally by gold type. Metrics carry type ids and counts
only — never page images or OCR text.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from torus.eval.metrics import f1_score, safe_div
from torus.eval.schema import PageLabel, PageLabelsDocument

# Sentinel for a gold page with no matching predicted index.
MISSING_PRED = "__missing__"


class PageTypeMetrics(BaseModel):
    """Per-type classification metrics (type id and counts only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_type: str = Field(min_length=1)
    support: int = Field(ge=0, description="Gold pages of this type (tp + fn).")
    correct: int = Field(ge=0, description="True positives (same as tp).")
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="correct / support (equals recall when support > 0).",
    )
    tp: int = Field(default=0, ge=0)
    fp: int = Field(default=0, ge=0)
    fn: int = Field(default=0, ge=0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("page_type")
    @classmethod
    def _strip_page_type(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "page_type must be non-empty"
            raise ValueError(msg)
        return stripped


class ConfusionMatrix(BaseModel):
    """Square count matrix: counts[i][j] = gold labels[i] → pred labels[j]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: list[str] = Field(min_length=1)
    counts: list[list[int]]

    @model_validator(mode="after")
    def _square_matches_labels(self) -> ConfusionMatrix:
        n = len(self.labels)
        if len(self.counts) != n:
            msg = "counts rows must match labels length"
            raise ValueError(msg)
        for row in self.counts:
            if len(row) != n:
                msg = "counts must be a square matrix matching labels"
                raise ValueError(msg)
            if any(cell < 0 for cell in row):
                msg = "counts must be non-negative"
                raise ValueError(msg)
        return self


class MisclassificationByType(BaseModel):
    """How often gold type *gold_type* was predicted as each wrong type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gold_type: str = Field(min_length=1)
    total: int = Field(ge=0, description="Pages with this gold type that were wrong.")
    as_predicted: dict[str, int] = Field(
        default_factory=dict,
        description="pred_type → count (wrong predictions only).",
    )


class ClassificationMetrics(BaseModel):
    """Aggregate page-type classification scores for a run or case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0, description="Gold pages scored (including missing preds).")
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    unmatched_gold: int = Field(
        ge=0,
        description="Gold page indices with no prediction (counted as __missing__).",
    )
    unmatched_pred: int = Field(
        ge=0,
        description="Prediction indices with no gold label (ignored in the matrix).",
    )
    per_type: list[PageTypeMetrics] = Field(default_factory=list)
    confusion: ConfusionMatrix | None = None
    misclassifications: list[MisclassificationByType] = Field(default_factory=list)


def pair_page_labels(
    gold: PageLabelsDocument | Sequence[PageLabel],
    pred: PageLabelsDocument | Sequence[PageLabel],
) -> tuple[list[tuple[str, str]], int, int]:
    """Align gold and predicted labels by page index.

    Returns ``(pairs, unmatched_gold, unmatched_pred)``. Each pair is
    ``(gold_type, pred_type)``. Gold indices without a prediction use
    :data:`MISSING_PRED` as the predicted type.
    """
    gold_pages = gold.pages if isinstance(gold, PageLabelsDocument) else gold
    pred_pages = pred.pages if isinstance(pred, PageLabelsDocument) else pred
    gold_by = {p.index: p.page_type for p in gold_pages}
    pred_by = {p.index: p.page_type for p in pred_pages}
    unmatched_gold = len(set(gold_by) - set(pred_by))
    unmatched_pred = len(set(pred_by) - set(gold_by))
    pairs: list[tuple[str, str]] = []
    for index in sorted(gold_by):
        g = gold_by[index]
        p = pred_by.get(index, MISSING_PRED)
        pairs.append((g, p))
    return pairs, unmatched_gold, unmatched_pred


def score_page_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    label_order: Sequence[str] | None = None,
    unmatched_gold: int = 0,
    unmatched_pred: int = 0,
) -> ClassificationMetrics:
    """Score ``(gold, pred)`` page-type pairs into :class:`ClassificationMetrics`."""
    if not pairs:
        return ClassificationMetrics(
            total=0,
            correct=0,
            accuracy=0.0,
            unmatched_gold=unmatched_gold,
            unmatched_pred=unmatched_pred,
            per_type=[],
            confusion=None,
            misclassifications=[],
        )

    correct = sum(1 for g, p in pairs if g == p)
    total = len(pairs)
    accuracy = safe_div(correct, total)

    labels = _resolve_labels(pairs, label_order)
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for g, p in pairs:
        matrix[index[g]][index[p]] += 1

    per_type = _per_type_metrics(labels, matrix)
    mis = _misclassifications(pairs)
    confusion = ConfusionMatrix(labels=labels, counts=matrix)
    return ClassificationMetrics(
        total=total,
        correct=correct,
        accuracy=round(accuracy, 4),
        unmatched_gold=unmatched_gold,
        unmatched_pred=unmatched_pred,
        per_type=per_type,
        confusion=confusion,
        misclassifications=mis,
    )


def score_page_labels(
    gold: PageLabelsDocument | Sequence[PageLabel],
    pred: PageLabelsDocument | Sequence[PageLabel],
    *,
    label_order: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Align by index then score gold vs predicted page types."""
    pairs, ug, up = pair_page_labels(gold, pred)
    return score_page_pairs(
        pairs,
        label_order=label_order,
        unmatched_gold=ug,
        unmatched_pred=up,
    )


def aggregate_classification(
    scores: Sequence[ClassificationMetrics],
    *,
    label_order: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Sum several case-level classification scores into one aggregate."""
    if not scores:
        return ClassificationMetrics(
            total=0,
            correct=0,
            accuracy=0.0,
            unmatched_gold=0,
            unmatched_pred=0,
        )

    # Rebuild pairs from confusion matrices when present; else skip empty.
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    unmatched_gold = unmatched_pred = 0
    for score in scores:
        unmatched_gold += score.unmatched_gold
        unmatched_pred += score.unmatched_pred
        if score.confusion is None:
            continue
        labs = score.confusion.labels
        for i, gold in enumerate(labs):
            for j, pred in enumerate(labs):
                count = score.confusion.counts[i][j]
                if count:
                    pair_counts[(gold, pred)] += count

    pairs: list[tuple[str, str]] = []
    for (g, p), count in sorted(pair_counts.items()):
        pairs.extend([(g, p)] * count)
    return score_page_pairs(
        pairs,
        label_order=label_order,
        unmatched_gold=unmatched_gold,
        unmatched_pred=unmatched_pred,
    )


def _resolve_labels(
    pairs: Sequence[tuple[str, str]],
    label_order: Sequence[str] | None,
) -> list[str]:
    seen: set[str] = set()
    for g, p in pairs:
        seen.add(g)
        seen.add(p)
    if label_order is None:
        # Prefer catalogue order stability: known types first would need config;
        # without it, sort but keep MISSING_PRED last.
        normal = sorted(t for t in seen if t != MISSING_PRED)
        if MISSING_PRED in seen:
            normal.append(MISSING_PRED)
        return normal
    ordered: list[str] = []
    for lab in label_order:
        if lab in seen and lab not in ordered:
            ordered.append(lab)
    for lab in sorted(seen):
        if lab not in ordered and lab != MISSING_PRED:
            ordered.append(lab)
    if MISSING_PRED in seen and MISSING_PRED not in ordered:
        ordered.append(MISSING_PRED)
    return ordered


def _per_type_metrics(
    labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> list[PageTypeMetrics]:
    out: list[PageTypeMetrics] = []
    n = len(labels)
    for i, page_type in enumerate(labels):
        if page_type == MISSING_PRED:
            # Sentinel is not a real gold class; skip per-type row.
            continue
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n) if r != i)
        fn = sum(matrix[i][c] for c in range(n) if c != i)
        support = tp + fn
        if support == 0 and fp == 0:
            continue
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        out.append(
            PageTypeMetrics(
                page_type=page_type,
                support=support,
                correct=tp,
                accuracy=round(safe_div(tp, support), 4),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1_score(prec, rec), 4),
            )
        )
    return out


def _misclassifications(
    pairs: Sequence[tuple[str, str]],
) -> list[MisclassificationByType]:
    wrong: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in pairs:
        if g == p:
            continue
        wrong[g][p] += 1
    result: list[MisclassificationByType] = []
    for gold_type in sorted(wrong):
        as_pred = dict(sorted(wrong[gold_type].items()))
        result.append(
            MisclassificationByType(
                gold_type=gold_type,
                total=sum(as_pred.values()),
                as_predicted=as_pred,
            )
        )
    return result


def empty_classification() -> ClassificationMetrics:
    """Zeroed metrics when no page labels were scored."""
    return ClassificationMetrics(
        total=0,
        correct=0,
        accuracy=0.0,
        unmatched_gold=0,
        unmatched_pred=0,
    )
