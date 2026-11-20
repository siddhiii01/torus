"""Re-export (P9.3): apply stored review decisions, then write the Excel again.

Decisions are applied to a copy of the ``ExtractionResult``; the original is
never mutated. A decision is applied only when it still matches the field it was
made on (same row, column and suggested value). Anything else is reported as
``unapplied`` rather than silently dropped or forced onto a different value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torus.config.schemas import OutputConfig, TemplateConfig
from torus.core.models import ExtractionResult, FieldValue
from torus.core.types import FieldStatus
from torus.output.excel import WriteSummary, write_workbook
from torus.review.store import CorrectionRecord, CorrectionStore

Reviewed = frozenset[tuple[str, int, str]]


@dataclass(frozen=True)
class AppliedResult:
    result: ExtractionResult
    reviewed: Reviewed
    corrected: int
    accepted: int
    skipped: int
    unapplied: tuple[str, ...]  # item ids; identifiers only, never values


@dataclass(frozen=True)
class ReexportSummary:
    write: WriteSummary
    corrected: int
    accepted: int
    skipped: int
    unapplied: tuple[str, ...]


def apply_corrections(
    result: ExtractionResult, decisions: dict[str, CorrectionRecord]
) -> AppliedResult:
    """Return ``result`` with each effective decision applied."""
    rows = [dict(row) for row in result.rows]
    reviewed: set[tuple[str, int, str]] = set()
    corrected = accepted = skipped = 0
    unapplied: list[str] = []

    for item_id, rec in decisions.items():
        field = _target(rows, rec, result.document_id)
        if field is None:
            unapplied.append(item_id)
            continue
        if rec.action == "skip":
            skipped += 1
            continue
        if rec.action == "accept":
            if not field.value:
                unapplied.append(item_id)
                continue
            new = field.model_copy(
                update={
                    "status": FieldStatus.CONFIRMED,
                    "reason": "reviewed: accepted",
                }
            )
            accepted += 1
        else:
            # A value needs evidence; a field that had none cannot take one.
            if not rec.value or not field.evidence:
                unapplied.append(item_id)
                continue
            # The reviewer's value was never validated, so the old results go.
            new = field.model_copy(
                update={
                    "value": rec.value,
                    "status": FieldStatus.CONFIRMED,
                    "validator_results": [],
                    "reason": "reviewed: corrected",
                }
            )
            corrected += 1
        rows[rec.row][rec.column_key] = new
        reviewed.add((result.document_id, rec.row, rec.column_key))

    return AppliedResult(
        result=result.model_copy(update={"rows": rows}),
        reviewed=frozenset(reviewed),
        corrected=corrected,
        accepted=accepted,
        skipped=skipped,
        unapplied=tuple(sorted(unapplied)),
    )


def _target(
    rows: list[dict[str, FieldValue]], rec: CorrectionRecord, document_id: str
) -> FieldValue | None:
    if rec.document_id != document_id or not 0 <= rec.row < len(rows):
        return None
    field = rows[rec.row].get(rec.column_key)
    if field is None or field.value != rec.suggestion:
        return None
    return field


def reexport(
    template_xlsx: Path,
    out_xlsx: Path,
    template: TemplateConfig,
    results: Sequence[ExtractionResult],
    store: CorrectionStore,
    output: OutputConfig | None = None,
) -> ReexportSummary:
    """Write ``results`` with stored decisions applied and marked reviewed."""
    applied = [apply_corrections(r, store.latest(r.document_id)) for r in results]
    reviewed: set[tuple[str, int, str]] = set()
    for a in applied:
        reviewed |= a.reviewed
    write = write_workbook(
        template_xlsx,
        out_xlsx,
        template,
        [a.result for a in applied],
        output,
        reviewed=reviewed,
    )
    return ReexportSummary(
        write=write,
        corrected=sum(a.corrected for a in applied),
        accepted=sum(a.accepted for a in applied),
        skipped=sum(a.skipped for a in applied),
        unapplied=tuple(i for a in applied for i in a.unapplied),
    )
