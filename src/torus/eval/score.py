"""Score predicted Excel rows against expected gold (P1.2).

Ports the behaviour of ``scripts/deb_score.py`` onto template config:
row alignment from ``template.row_key``, match rules from each field's
``match`` / ``normalize``, plus silent-error and status-rate metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torus.config.schemas import FieldSpec, TemplateConfig
from torus.core.models import ExtractionResult, FieldValue
from torus.core.types import FieldStatus
from torus.eval.align import AlignmentResult, RowDict, align_rows
from torus.eval.match import values_match
from torus.eval.metrics import (
    CaseScore,
    FieldMetrics,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
    empty_status_counts,
    f1_score,
    safe_div,
)
from torus.eval.normalize import is_empty
from torus.eval.schema import ExpectedCell, ExpectedDocument


def _cell_value(cell: Any) -> ExpectedCell:
    if isinstance(cell, FieldValue):
        return cell.value
    if cell is None or isinstance(cell, (str, int, float, bool)):
        return cell
    if hasattr(cell, "value"):
        return cell.value  # type: ignore[no-any-return]
    return cell  # type: ignore[no-any-return]


def _cell_status(cell: Any) -> FieldStatus | None:
    if isinstance(cell, FieldValue):
        return cell.status
    status = getattr(cell, "status", None)
    if isinstance(status, FieldStatus):
        return status
    if isinstance(status, str):
        try:
            return FieldStatus(status)
        except ValueError:
            return None
    return None


def _row_as_mapping(row: RowDict | Mapping[str, FieldValue]) -> RowDict:
    return row


def _aliases_for_field(
    template: TemplateConfig, field_key: str
) -> dict[str, list[str]] | None:
    """Aliases attached to a row_key part that lists *field_key* as a source."""
    if template.row_key is None:
        return None
    parts = list(template.row_key.primary)
    if template.row_key.disambiguator is not None:
        parts.append(template.row_key.disambiguator)
    for part in parts:
        if field_key in part.sources and part.aliases:
            return dict(part.aliases)
    return None


def _fuzzy_threshold(template: TemplateConfig) -> float:
    if template.row_key is None:
        return 0.8
    return template.row_key.fuzzy_threshold


def _score_aligned_pair(
    *,
    gold: RowDict,
    pred: RowDict,
    scored: Sequence[FieldSpec],
    fuzzy_threshold: float,
    field_stats: dict[str, dict[str, int]],
    status_totals: dict[str, int],
    mismatches: list[dict[str, str | bool]],
    template: TemplateConfig,
) -> int:
    """Update stats for one aligned pair; return silent-error count added."""
    silent = 0
    for spec in scored:
        key = spec.key
        gv = _cell_value(gold.get(key)) if key in gold else None
        pv = _cell_value(pred.get(key)) if key in pred else None
        status = _cell_status(pred.get(key)) if key in pred else None

        g_empty = is_empty(gv)
        p_empty = is_empty(pv)
        if g_empty and p_empty:
            if status is not None:
                status_totals[status.value] = status_totals.get(status.value, 0) + 1
                field_stats[key]["status_" + status.value] = (
                    field_stats[key].get("status_" + status.value, 0) + 1
                )
            continue

        field_stats[key]["compared"] += 1
        ok = values_match(
            spec,
            pv,
            gv,
            fuzzy_threshold=fuzzy_threshold,
            aliases=_aliases_for_field(template, key),
        )
        if ok:
            field_stats[key]["tp"] += 1
            field_stats[key]["correct"] += 1
        elif g_empty and not p_empty:
            field_stats[key]["fp"] += 1
        elif p_empty and not g_empty:
            field_stats[key]["fn"] += 1
        else:
            field_stats[key]["fp"] += 1
            field_stats[key]["fn"] += 1

        if status is not None:
            status_totals[status.value] = status_totals.get(status.value, 0) + 1
            field_stats[key]["status_" + status.value] = (
                field_stats[key].get("status_" + status.value, 0) + 1
            )
            if status is FieldStatus.CONFIRMED and not ok:
                field_stats[key]["silent_errors"] += 1
                silent += 1

        mismatches.append({"field": key, "match": ok})
    return silent


def _score_unmatched_pred(
    *,
    pred: RowDict,
    scored: Sequence[FieldSpec],
    field_stats: dict[str, dict[str, int]],
    status_totals: dict[str, int],
) -> int:
    """Confirmed non-empty values on unmatched pred rows are silent errors."""
    silent = 0
    for spec in scored:
        key = spec.key
        if key not in pred:
            continue
        cell = pred[key]
        pv = _cell_value(cell)
        status = _cell_status(cell)
        if is_empty(pv):
            continue
        field_stats[key]["fp"] += 1
        field_stats[key]["compared"] += 1
        if status is not None:
            status_totals[status.value] = status_totals.get(status.value, 0) + 1
            if status is FieldStatus.CONFIRMED:
                field_stats[key]["silent_errors"] += 1
                silent += 1
    return silent


def _build_per_field(
    scored: Sequence[FieldSpec],
    field_stats: dict[str, dict[str, int]],
) -> dict[str, FieldMetrics]:
    per_field: dict[str, FieldMetrics] = {}
    for spec in scored:
        st = field_stats[spec.key]
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        compared = st["compared"]
        correct = st["correct"]
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        status_counts = {
            status.value: st.get("status_" + status.value, 0) for status in FieldStatus
        }
        per_field[spec.key] = FieldMetrics(
            key=spec.key,
            tp=tp,
            fp=fp,
            fn=fn,
            compared=compared,
            correct=correct,
            accuracy=safe_div(correct, compared),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1_score(prec, rec), 4),
            silent_errors=st["silent_errors"],
            status_counts=status_counts,
        )
    return per_field


def _build_status_rates(status_totals: dict[str, int]) -> StatusRates:
    total = sum(status_totals.values())
    counts = empty_status_counts()
    counts.update({k: v for k, v in status_totals.items() if v})
    # Ensure all statuses present
    for status in FieldStatus:
        counts.setdefault(status.value, 0)
    rates = {name: safe_div(count, total) for name, count in counts.items()}
    automatic = safe_div(counts.get(FieldStatus.CONFIRMED.value, 0), total)
    review = safe_div(counts.get(FieldStatus.NEEDS_REVIEW.value, 0), total)
    return StatusRates(
        total=total,
        counts=counts,
        rates={k: round(v, 4) for k, v in rates.items()},
        automatic_rate=round(automatic, 4),
        review_rate=round(review, 4),
    )


def score_rows(
    gold_rows: Sequence[RowDict],
    pred_rows: Sequence[RowDict],
    template: TemplateConfig,
    *,
    case_id: str | None = None,
) -> CaseScore:
    """Score plain or ``FieldValue`` rows against gold using *template* rules."""
    scored = template.scored_fields()
    fuzzy_threshold = _fuzzy_threshold(template)

    alignment: AlignmentResult = align_rows(gold_rows, pred_rows, template.row_key)
    n_gold, n_pred = len(gold_rows), len(pred_rows)
    n_aligned = len(alignment.pairs)
    row_recall = safe_div(n_aligned, n_gold)
    row_precision = safe_div(n_aligned, n_pred)
    row_metrics = RowAlignmentMetrics(
        gold=n_gold,
        pred=n_pred,
        aligned=n_aligned,
        unmatched_gold=len(alignment.unmatched_gold),
        unmatched_pred=len(alignment.unmatched_pred),
        precision=round(row_precision, 4),
        recall=round(row_recall, 4),
        f1=round(f1_score(row_precision, row_recall), 4),
    )

    field_stats: dict[str, dict[str, int]] = {
        spec.key: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "compared": 0,
            "correct": 0,
            "silent_errors": 0,
        }
        for spec in scored
    }
    status_totals: dict[str, int] = {}
    mismatches: list[dict[str, str | bool]] = []
    silent_errors = 0

    for gold, pred in alignment.pairs:
        silent_errors += _score_aligned_pair(
            gold=_row_as_mapping(gold),
            pred=_row_as_mapping(pred),
            scored=scored,
            fuzzy_threshold=fuzzy_threshold,
            field_stats=field_stats,
            status_totals=status_totals,
            mismatches=mismatches,
            template=template,
        )

    for pred in alignment.unmatched_pred:
        silent_errors += _score_unmatched_pred(
            pred=_row_as_mapping(pred),
            scored=scored,
            field_stats=field_stats,
            status_totals=status_totals,
        )

    per_field = _build_per_field(scored, field_stats)
    tot_tp = sum(m.tp for m in per_field.values())
    tot_fp = sum(m.fp for m in per_field.values())
    tot_fn = sum(m.fn for m in per_field.values())
    tot_compared = sum(m.compared for m in per_field.values())
    tot_correct = sum(m.correct for m in per_field.values())
    micro_p = safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = safe_div(tot_tp, tot_tp + tot_fn)
    micro = MicroMetrics(
        tp=tot_tp,
        fp=tot_fp,
        fn=tot_fn,
        compared=tot_compared,
        correct=tot_correct,
        accuracy=round(safe_div(tot_correct, tot_compared), 4),
        precision=round(micro_p, 4),
        recall=round(micro_r, 4),
        f1=round(f1_score(micro_p, micro_r), 4),
    )

    # Drop match=True entries to keep payloads small; keep failures only.
    mismatch_failures = [m for m in mismatches if m.get("match") is False]

    return CaseScore(
        case_id=case_id,
        template_id=template.id,
        rows=row_metrics,
        fields_micro=micro,
        per_field=per_field,
        silent_errors=silent_errors,
        status_rates=_build_status_rates(status_totals),
        mismatches=mismatch_failures,
    )


def score_expected(
    expected: ExpectedDocument,
    pred_rows: Sequence[RowDict],
    template: TemplateConfig,
    *,
    case_id: str | None = None,
) -> CaseScore:
    """Score predictions against an ``ExpectedDocument``."""
    if expected.template_id != template.id:
        from torus.core.errors import EvalError

        raise EvalError(
            f"template_id mismatch: expected={expected.template_id!r} "
            f"template={template.id!r}"
        )
    resolved_case = case_id if case_id is not None else expected.case_id
    return score_rows(
        expected.rows,
        pred_rows,
        template,
        case_id=resolved_case,
    )


def score_extraction(
    expected: ExpectedDocument,
    result: ExtractionResult,
    template: TemplateConfig,
    *,
    case_id: str | None = None,
) -> CaseScore:
    """Score a pipeline ``ExtractionResult`` against gold."""
    if result.template_id != template.id:
        from torus.core.errors import EvalError

        raise EvalError(
            f"template_id mismatch: extraction={result.template_id!r} "
            f"template={template.id!r}"
        )
    return score_expected(
        expected,
        result.rows,
        template,
        case_id=case_id,
    )


def aggregate_scores(scores: Sequence[CaseScore]) -> CaseScore:
    """Micro-aggregate several case scores into one summary ``CaseScore``.

    Row metrics and per-field counters are summed; rates recomputed.
    """
    if not scores:
        empty_status = StatusRates(
            total=0,
            counts=empty_status_counts(),
            rates={s.value: 0.0 for s in FieldStatus},
            automatic_rate=0.0,
            review_rate=0.0,
        )
        return CaseScore(
            case_id="aggregate",
            template_id="",
            rows=RowAlignmentMetrics(
                gold=0,
                pred=0,
                aligned=0,
                unmatched_gold=0,
                unmatched_pred=0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            ),
            fields_micro=MicroMetrics(
                tp=0,
                fp=0,
                fn=0,
                compared=0,
                correct=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            ),
            silent_errors=0,
            status_rates=empty_status,
        )

    template_id = scores[0].template_id
    gold = pred = aligned = ug = up = 0
    silent = 0
    status_counts = empty_status_counts()
    # Union of field keys
    field_keys: list[str] = []
    seen: set[str] = set()
    for score in scores:
        for key in score.per_field:
            if key not in seen:
                seen.add(key)
                field_keys.append(key)

    accum: dict[str, dict[str, int]] = {
        key: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "compared": 0,
            "correct": 0,
            "silent_errors": 0,
            **{f"status_{s.value}": 0 for s in FieldStatus},
        }
        for key in field_keys
    }

    for score in scores:
        gold += score.rows.gold
        pred += score.rows.pred
        aligned += score.rows.aligned
        ug += score.rows.unmatched_gold
        up += score.rows.unmatched_pred
        silent += score.silent_errors
        for name, count in score.status_rates.counts.items():
            status_counts[name] = status_counts.get(name, 0) + count
        for key, metrics in score.per_field.items():
            bucket = accum[key]
            bucket["tp"] += metrics.tp
            bucket["fp"] += metrics.fp
            bucket["fn"] += metrics.fn
            bucket["compared"] += metrics.compared
            bucket["correct"] += metrics.correct
            bucket["silent_errors"] += metrics.silent_errors
            for status in FieldStatus:
                bucket["status_" + status.value] += metrics.status_counts.get(
                    status.value, 0
                )

    # Rebuild FieldMetrics via a synthetic FieldSpec list is awkward; build directly.
    per_field: dict[str, FieldMetrics] = {}
    for key, st in accum.items():
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        compared, correct = st["compared"], st["correct"]
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        per_field[key] = FieldMetrics(
            key=key,
            tp=tp,
            fp=fp,
            fn=fn,
            compared=compared,
            correct=correct,
            accuracy=round(safe_div(correct, compared), 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1_score(prec, rec), 4),
            silent_errors=st["silent_errors"],
            status_counts={
                status.value: st.get("status_" + status.value, 0)
                for status in FieldStatus
            },
        )

    tot_tp = sum(m.tp for m in per_field.values())
    tot_fp = sum(m.fp for m in per_field.values())
    tot_fn = sum(m.fn for m in per_field.values())
    tot_compared = sum(m.compared for m in per_field.values())
    tot_correct = sum(m.correct for m in per_field.values())
    micro_p = safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = safe_div(tot_tp, tot_tp + tot_fn)
    row_p = safe_div(aligned, pred)
    row_r = safe_div(aligned, gold)

    return CaseScore(
        case_id="aggregate",
        template_id=template_id,
        rows=RowAlignmentMetrics(
            gold=gold,
            pred=pred,
            aligned=aligned,
            unmatched_gold=ug,
            unmatched_pred=up,
            precision=round(row_p, 4),
            recall=round(row_r, 4),
            f1=round(f1_score(row_p, row_r), 4),
        ),
        fields_micro=MicroMetrics(
            tp=tot_tp,
            fp=tot_fp,
            fn=tot_fn,
            compared=tot_compared,
            correct=tot_correct,
            accuracy=round(safe_div(tot_correct, tot_compared), 4),
            precision=round(micro_p, 4),
            recall=round(micro_r, 4),
            f1=round(f1_score(micro_p, micro_r), 4),
        ),
        per_field=per_field,
        silent_errors=silent,
        status_rates=_build_status_rates(status_counts),
        mismatches=[],
    )
