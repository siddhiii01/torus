"""Run the evaluation harness over a labelled test set (P1.6 / S6 / P3.5 / P4.8).

Loads gold from ``TORUS_DATA_DIR/testsets/<set>/``, predictions from a
directory of Excel-shaped JSON (legacy extract outputs), scores with
``torus.eval.score``, optionally scores page-type labels (``pages.json``)
and field locations (``locations.json``), and writes a metrics-only report.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torus.config import default_config_dir, load_config, load_template
from torus.config.schemas import TemplateConfig, TorusConfig
from torus.core.errors import ConfigError, EvalError
from torus.core.types import FieldStatus
from torus.eval.align import RowDict
from torus.eval.classify_score import (
    ClassificationMetrics,
    aggregate_classification,
    empty_classification,
    score_page_labels,
)
from torus.eval.location_score import (
    LocationMetrics,
    aggregate_locations,
    empty_location_metrics,
    score_locations,
)
from torus.eval.locations import load_pred_locations
from torus.eval.metrics import (
    CaseScore,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
    empty_status_counts,
)
from torus.eval.page_labels import load_pred_page_labels
from torus.eval.predictions import load_prediction_rows, resolve_prediction_path
from torus.eval.report import (
    EvalRunReport,
    RunVersions,
    TimingSummary,
    WrittenReportPaths,
    build_run_report,
    merge_timings,
    write_report,
)
from torus.eval.review_metrics import ReviewMetrics
from torus.eval.schema import Split
from torus.eval.score import aggregate_scores, score_expected
from torus.eval.validate import ValidatedCase, validate_testset


@dataclass(frozen=True, slots=True)
class CaseRunResult:
    """Per-case timing and whether a prediction file was found."""

    case_id: str
    scored: bool
    skipped_missing_pred: bool
    elapsed_ms: float
    classification_scored: bool = False
    location_scored: bool = False


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Full harness run: report object, written paths, case outcomes."""

    report: EvalRunReport
    paths: WrittenReportPaths
    cases: tuple[CaseRunResult, ...]
    skipped_missing_pred: tuple[str, ...]


def _template_for_cases(
    cases: Sequence[ValidatedCase],
    config: TorusConfig,
    *,
    template_id: str | None,
) -> TemplateConfig:
    ids = {c.meta.template_id for c in cases}
    if template_id is not None:
        if template_id not in config.templates:
            raise EvalError(f"unknown template_id: {template_id!r}")
        return config.templates[template_id]
    if len(ids) != 1:
        raise EvalError(
            f"test set mixes template ids {sorted(ids)}; pass --template to "
            f"select one, or split the set"
        )
    only = next(iter(ids))
    if only not in config.templates:
        raise EvalError(f"unknown template_id from case meta: {only!r}")
    return config.templates[only]


def _filter_split(
    cases: Sequence[ValidatedCase], split: Split | None
) -> list[ValidatedCase]:
    if split is None:
        return list(cases)
    return [c for c in cases if c.meta.split == split]


def run_evaluation(
    *,
    set_id: str,
    pred_dir: Path,
    report_path: Path,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
    split: Split | None = Split.DEV,
    template_id: str | None = None,
    pred_variant: str | None = None,
    model_versions: Mapping[str, str] | None = None,
    require_predictions: bool = False,
    empty_pred_when_missing: bool = False,
    review: ReviewMetrics | None = None,
) -> EvalRunResult:
    """Score *set_id* against predictions in *pred_dir* and write a report.

    Parameters
    ----------
    require_predictions:
        When True, raise if any selected case lacks a prediction file.
    empty_pred_when_missing:
        When True (and not require_predictions), score missing preds as
        zero rows. When False, skip those cases and record them on the
        result.
    """
    pred_root = Path(pred_dir)
    if not pred_root.is_dir():
        raise EvalError(f"prediction directory not found: {pred_root}")

    cfg_root = Path(config_dir) if config_dir else default_config_dir()
    try:
        config = load_config(cfg_root)
    except ConfigError:
        raise

    cases = validate_testset(set_id, data_dir=data_dir)
    cases = _filter_split(cases, split)
    if not cases:
        raise EvalError(
            f"no cases in set {set_id!r}"
            + (f" for split {split.value}" if split is not None else "")
        )

    template = _template_for_cases(cases, config, template_id=template_id)
    # Re-validate keys against the resolved template when ids match.
    if template_id is not None:
        mismatched = [c.case_id for c in cases if c.meta.template_id != template.id]
        if mismatched:
            raise EvalError(
                f"cases with template_id != {template.id!r}: {', '.join(mismatched)}"
            )

    scores = []
    timings: list[dict[str, float]] = []
    case_results: list[CaseRunResult] = []
    scored_ids: list[str] = []
    skipped: list[str] = []
    class_scores: list[ClassificationMetrics] = []
    location_scores: list[LocationMetrics] = []
    label_order = [spec.id for spec in config.page_types.page_types]
    locate_iou = config.pipeline.layout.locate_iou

    for case in cases:
        t0 = time.perf_counter()
        pred_path = resolve_prediction_path(
            pred_root, case.case_id, variant=pred_variant
        )
        if pred_path is None:
            if require_predictions:
                raise EvalError(
                    f"missing prediction for case {case.case_id} under {pred_root}"
                )
            if not empty_pred_when_missing:
                elapsed = (time.perf_counter() - t0) * 1000.0
                case_results.append(
                    CaseRunResult(
                        case_id=case.case_id,
                        scored=False,
                        skipped_missing_pred=True,
                        elapsed_ms=elapsed,
                    )
                )
                skipped.append(case.case_id)
                continue
            pred_rows: list[RowDict] = []
        else:
            pred_rows = load_prediction_rows(pred_path, case_id=case.case_id)

        score = score_expected(
            case.expected,
            pred_rows,
            template,
            case_id=case.case_id,
        )
        clf_scored = False
        if case.pages is not None:
            pred_pages = load_pred_page_labels(
                pred_root, case.case_id, variant=pred_variant
            )
            if pred_pages is not None:
                class_scores.append(
                    score_page_labels(case.pages, pred_pages, label_order=label_order)
                )
                clf_scored = True
        loc_scored = False
        if case.locations is not None:
            pred_locs = load_pred_locations(
                pred_root, case.case_id, variant=pred_variant
            )
            if pred_locs is not None:
                location_scores.append(
                    score_locations(case.locations, pred_locs, iou_threshold=locate_iou)
                )
                loc_scored = True
        elapsed = (time.perf_counter() - t0) * 1000.0
        scores.append(score)
        timings.append({"score": elapsed})
        scored_ids.append(case.case_id)
        case_results.append(
            CaseRunResult(
                case_id=case.case_id,
                scored=True,
                skipped_missing_pred=False,
                elapsed_ms=elapsed,
                classification_scored=clf_scored,
                location_scored=loc_scored,
            )
        )

    if not scores:
        raise EvalError(
            f"no cases scored in set {set_id!r} "
            f"(skipped missing predictions: {', '.join(skipped) or 'none'})"
        )

    aggregate = aggregate_scores(scores)
    # Preserve template id on the aggregate (aggregate_scores uses first score).
    aggregate = aggregate.model_copy(update={"template_id": template.id})
    timing_summary: TimingSummary = merge_timings(timings)
    versions = RunVersions(
        config_hash=config.config_hash,
        template_id=template.id,
        model_versions=dict(model_versions or {}),
    )
    classification = (
        aggregate_classification(class_scores, label_order=label_order)
        if class_scores
        else None
    )
    location = (
        aggregate_locations(location_scores, iou_threshold=locate_iou)
        if location_scores
        else None
    )
    report = build_run_report(
        set_id=set_id,
        aggregate=aggregate,
        case_ids=scored_ids,
        timings=timing_summary,
        versions=versions,
        classification=classification,
        location=location,
        review=review,
        split=split.value if split is not None else None,
    )
    paths = write_report(report, report_path)
    return EvalRunResult(
        report=report,
        paths=paths,
        cases=tuple(case_results),
        skipped_missing_pred=tuple(skipped),
    )


def _placeholder_field_score(template_id: str) -> CaseScore:
    """Minimal CaseScore when running classification-only evaluation."""
    return CaseScore(
        case_id="aggregate",
        template_id=template_id,
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
        status_rates=StatusRates(
            total=0,
            counts=empty_status_counts(),
            rates={s.value: 0.0 for s in FieldStatus},
            automatic_rate=0.0,
            review_rate=0.0,
        ),
    )


def run_classification_evaluation(
    *,
    set_id: str,
    pred_dir: Path,
    report_path: Path,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
    split: Split | None = Split.DEV,
    template_id: str | None = None,
    pred_variant: str | None = None,
    model_versions: Mapping[str, str] | None = None,
    require_page_labels: bool = True,
) -> EvalRunResult:
    """Score page-type predictions against gold ``pages.json`` and write a report.

    Field/Excel scoring is skipped; the report carries classification metrics,
    confusion matrix, and misclassifications by gold type.
    """
    pred_root = Path(pred_dir)
    if not pred_root.is_dir():
        raise EvalError(f"prediction directory not found: {pred_root}")

    cfg_root = Path(config_dir) if config_dir else default_config_dir()
    try:
        config = load_config(cfg_root)
    except ConfigError:
        raise

    cases = validate_testset(set_id, data_dir=data_dir, require_source=False)
    cases = _filter_split(cases, split)
    if not cases:
        raise EvalError(
            f"no cases in set {set_id!r}"
            + (f" for split {split.value}" if split is not None else "")
        )

    template = _template_for_cases(cases, config, template_id=template_id)
    label_order = [spec.id for spec in config.page_types.page_types]
    class_scores: list[ClassificationMetrics] = []
    timings: list[dict[str, float]] = []
    case_results: list[CaseRunResult] = []
    scored_ids: list[str] = []
    skipped: list[str] = []

    for case in cases:
        t0 = time.perf_counter()
        if case.pages is None:
            if require_page_labels:
                raise EvalError(f"missing gold pages.json for case {case.case_id}")
            elapsed = (time.perf_counter() - t0) * 1000.0
            case_results.append(
                CaseRunResult(
                    case_id=case.case_id,
                    scored=False,
                    skipped_missing_pred=True,
                    elapsed_ms=elapsed,
                )
            )
            skipped.append(case.case_id)
            continue
        pred_pages = load_pred_page_labels(
            pred_root, case.case_id, variant=pred_variant
        )
        if pred_pages is None:
            if require_page_labels:
                raise EvalError(
                    f"missing page-type prediction for case {case.case_id} "
                    f"under {pred_root}"
                )
            elapsed = (time.perf_counter() - t0) * 1000.0
            case_results.append(
                CaseRunResult(
                    case_id=case.case_id,
                    scored=False,
                    skipped_missing_pred=True,
                    elapsed_ms=elapsed,
                )
            )
            skipped.append(case.case_id)
            continue
        metrics = score_page_labels(case.pages, pred_pages, label_order=label_order)
        elapsed = (time.perf_counter() - t0) * 1000.0
        class_scores.append(metrics)
        timings.append({"classify_score": elapsed})
        scored_ids.append(case.case_id)
        case_results.append(
            CaseRunResult(
                case_id=case.case_id,
                scored=True,
                skipped_missing_pred=False,
                elapsed_ms=elapsed,
                classification_scored=True,
            )
        )

    if not class_scores:
        raise EvalError(
            f"no page-type cases scored in set {set_id!r} "
            f"(skipped: {', '.join(skipped) or 'none'})"
        )

    classification = aggregate_classification(class_scores, label_order=label_order)
    if classification.total == 0:
        classification = empty_classification()

    versions = RunVersions(
        config_hash=config.config_hash,
        template_id=template.id,
        model_versions=dict(model_versions or {}),
    )
    report = build_run_report(
        set_id=set_id,
        aggregate=_placeholder_field_score(template.id),
        case_ids=scored_ids,
        timings=merge_timings(timings),
        versions=versions,
        classification=classification,
        split=split.value if split is not None else None,
    )
    paths = write_report(report, report_path)
    return EvalRunResult(
        report=report,
        paths=paths,
        cases=tuple(case_results),
        skipped_missing_pred=tuple(skipped),
    )


def run_location_evaluation(
    *,
    set_id: str,
    pred_dir: Path,
    report_path: Path,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
    split: Split | None = Split.DEV,
    template_id: str | None = None,
    pred_variant: str | None = None,
    model_versions: Mapping[str, str] | None = None,
    require_locations: bool = True,
    iou_threshold: float | None = None,
) -> EvalRunResult:
    """Score field-location predictions against gold ``locations.json``.

    Field/Excel scoring is skipped; the report carries location metrics
    (located rate per column, mean IoU). Used before reading to gate Phase 4.
    """
    pred_root = Path(pred_dir)
    if not pred_root.is_dir():
        raise EvalError(f"prediction directory not found: {pred_root}")

    cfg_root = Path(config_dir) if config_dir else default_config_dir()
    try:
        config = load_config(cfg_root)
    except ConfigError:
        raise

    cases = validate_testset(set_id, data_dir=data_dir, require_source=False)
    cases = _filter_split(cases, split)
    if not cases:
        raise EvalError(
            f"no cases in set {set_id!r}"
            + (f" for split {split.value}" if split is not None else "")
        )

    template = _template_for_cases(cases, config, template_id=template_id)
    locate_iou = (
        float(iou_threshold)
        if iou_threshold is not None
        else config.pipeline.layout.locate_iou
    )
    location_scores: list[LocationMetrics] = []
    timings: list[dict[str, float]] = []
    case_results: list[CaseRunResult] = []
    scored_ids: list[str] = []
    skipped: list[str] = []

    for case in cases:
        t0 = time.perf_counter()
        if case.locations is None:
            if require_locations:
                raise EvalError(f"missing gold locations.json for case {case.case_id}")
            elapsed = (time.perf_counter() - t0) * 1000.0
            case_results.append(
                CaseRunResult(
                    case_id=case.case_id,
                    scored=False,
                    skipped_missing_pred=True,
                    elapsed_ms=elapsed,
                )
            )
            skipped.append(case.case_id)
            continue
        pred_locs = load_pred_locations(pred_root, case.case_id, variant=pred_variant)
        if pred_locs is None:
            if require_locations:
                raise EvalError(
                    f"missing location prediction for case {case.case_id} "
                    f"under {pred_root}"
                )
            elapsed = (time.perf_counter() - t0) * 1000.0
            case_results.append(
                CaseRunResult(
                    case_id=case.case_id,
                    scored=False,
                    skipped_missing_pred=True,
                    elapsed_ms=elapsed,
                )
            )
            skipped.append(case.case_id)
            continue
        metrics = score_locations(case.locations, pred_locs, iou_threshold=locate_iou)
        elapsed = (time.perf_counter() - t0) * 1000.0
        location_scores.append(metrics)
        timings.append({"location_score": elapsed})
        scored_ids.append(case.case_id)
        case_results.append(
            CaseRunResult(
                case_id=case.case_id,
                scored=True,
                skipped_missing_pred=False,
                elapsed_ms=elapsed,
                location_scored=True,
            )
        )

    if not location_scores:
        raise EvalError(
            f"no location cases scored in set {set_id!r} "
            f"(skipped: {', '.join(skipped) or 'none'})"
        )

    location = aggregate_locations(location_scores, iou_threshold=locate_iou)
    if location.expected == 0 and location.unmatched_pred == 0:
        location = empty_location_metrics(iou_threshold=locate_iou)

    versions = RunVersions(
        config_hash=config.config_hash,
        template_id=template.id,
        model_versions=dict(model_versions or {}),
    )
    report = build_run_report(
        set_id=set_id,
        aggregate=_placeholder_field_score(template.id),
        case_ids=scored_ids,
        timings=merge_timings(timings),
        versions=versions,
        location=location,
        split=split.value if split is not None else None,
    )
    paths = write_report(report, report_path)
    return EvalRunResult(
        report=report,
        paths=paths,
        cases=tuple(case_results),
        skipped_missing_pred=tuple(skipped),
    )


def load_template_by_id(
    template_id: str, *, config_dir: Path | None = None
) -> TemplateConfig:
    """Load one template YAML by id (helper for tests and promote)."""
    root = Path(config_dir) if config_dir else default_config_dir()
    path = root / "templates" / f"{template_id}.yaml"
    return load_template(path)
