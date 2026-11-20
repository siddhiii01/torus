"""Promote a reviewed pack into the test-set format (P9.4, format from P1.1).

What becomes gold: only values a person decided (accepted or corrected). A value
the engine alone produced is never gold, or the evaluation would grade the
engine against itself. ``include_unreviewed_confirmed`` opts engine-confirmed
values in for packs where the reviewer cannot re-check everything; it biases the
set toward the engine, so it is off by default and recorded in ``notes``.

Skipped, undecided and ``not_found`` fields get no key (no gold). Decisions that
no longer match the result abort the promotion rather than being guessed at.
"""

from __future__ import annotations

from pathlib import Path

from torus.config.schemas import TemplateConfig
from torus.core.errors import EvalError
from torus.core.models import ExtractionResult
from torus.core.types import FieldStatus
from torus.eval.label import LabelCaseRequest, LabelledCaseResult, write_labelled_case
from torus.eval.schema import (
    ExpectedCell,
    HandwritingLevel,
    ScanType,
    Split,
)
from torus.review.queue import ReviewError
from torus.review.reexport import apply_corrections
from torus.review.store import CorrectionStore


def gold_rows(
    result: ExtractionResult,
    store: CorrectionStore,
    *,
    include_unreviewed_confirmed: bool = False,
) -> tuple[list[dict[str, ExpectedCell]], int]:
    """Gold rows for ``result`` and the count of reviewed values among them."""
    applied = apply_corrections(result, store.latest(result.document_id))
    if applied.unapplied:
        raise ReviewError(
            f"{len(applied.unapplied)} decision(s) no longer match the result; "
            "re-run review before promoting"
        )
    rows: list[dict[str, ExpectedCell]] = []
    reviewed_count = 0
    for row_index, row in enumerate(applied.result.rows):
        out: dict[str, ExpectedCell] = {}
        for key, fv in row.items():
            if not fv.value:
                continue
            is_reviewed = (result.document_id, row_index, key) in applied.reviewed
            if is_reviewed:
                reviewed_count += 1
            elif not (
                include_unreviewed_confirmed and fv.status == FieldStatus.CONFIRMED
            ):
                continue
            out[key] = fv.value
        rows.append(out)
    return rows, reviewed_count


def promote_reviewed(
    result: ExtractionResult,
    source_path: Path,
    store: CorrectionStore,
    template: TemplateConfig,
    *,
    set_id: str,
    case_id: str,
    split: Split = Split.DEV,
    year: int | None = None,
    scan_type: ScanType | None = None,
    handwriting_level: HandwritingLevel | None = None,
    include_unreviewed_confirmed: bool = False,
    force: bool = False,
    data_dir: Path | None = None,
) -> LabelledCaseResult:
    """Write ``testsets/<set>/<case>/`` from a reviewed pack."""
    if result.template_id != template.id:
        raise EvalError(
            f"result is for template {result.template_id!r}, not {template.id!r}"
        )
    rows, reviewed_count = gold_rows(
        result, store, include_unreviewed_confirmed=include_unreviewed_confirmed
    )
    if reviewed_count == 0:
        raise EvalError("nothing was reviewed in this pack; nothing to promote")
    if not any(rows):
        raise EvalError("no gold values to write")
    notes = "promoted from review"
    if include_unreviewed_confirmed:
        notes += "; includes engine-confirmed values that were not reviewed"
    return write_labelled_case(
        LabelCaseRequest(
            pdf_path=source_path,
            set_id=set_id,
            case_id=case_id,
            template=template,
            split=split,
            rows=rows,
            year=year,
            scan_type=scan_type,
            handwriting_level=handwriting_level,
            notes=notes,
            force=force,
        ),
        data_dir=data_dir,
    )
