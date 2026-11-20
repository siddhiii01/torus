"""Excel writer (P8.5): fill a copy of the client template, add an evidence sheet.

The client workbook is never modified: it is loaded, filled and saved under a
new path, so the client's formatting survives. Filled cells are coloured by
status; a second sheet lists status, confidence and source page + box for every
engine-owned cell, whether or not its value was written.

Safety rules (set in ``config/pipeline.yaml`` under ``output``):

* a cell that already holds anything is protected unless ``overwrite_existing``;
* passthrough columns are never written, whatever the policy;
* values are always stored as text, so ``=...`` can never become a formula and
  leading zeros survive.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torus.config.schemas import FieldSpec, OutputConfig, TemplateConfig
from torus.core.errors import ConfigError, OutputError
from torus.core.models import Evidence, ExtractionResult, FieldValue
from torus.core.types import FieldStatus
from torus.output.template import read_headers

_EVIDENCE_HEADERS = (
    "document_id",
    "row",
    "column_key",
    "column",
    "cell",
    "value",
    "status",
    "confidence",
    "written",
    "note",
    "evidence",
    "reviewed",
)

# Written-state vocabulary for the evidence sheet's ``written`` column.
WRITTEN = "written"
SKIPPED_PROTECTED = "skipped_protected"
SKIPPED_STATUS = "skipped_status"
SKIPPED_EMPTY = "skipped_empty"


@dataclass(frozen=True)
class WriteSummary:
    """Counts only — never any cell content."""

    path: Path
    sheet: str
    rows: int
    written: int
    skipped_protected: int
    skipped_status: int
    skipped_empty: int


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def _set_text(cell: Any, value: str) -> None:
    cell.value = value
    # openpyxl turns a leading "=" into a formula; force plain text.
    cell.data_type = "s"


def format_evidence(evidence: Evidence) -> str:
    """``p<page, 1-based> [x0,y0,x1,y1] <region id>`` — locations, no content."""
    b = evidence.bbox
    return (
        f"p{evidence.page_index + 1} "
        f"[{b.x0:.3f},{b.y0:.3f},{b.x1:.3f},{b.y1:.3f}] {evidence.region_id}"
    )


def _engine_fields(template: TemplateConfig) -> list[FieldSpec]:
    fields = template.scored_fields()
    for spec in fields:
        if spec.excel_col is None:
            raise OutputError(
                f"template {template.id!r} field {spec.key!r} has no excel_col; "
                "re-run the template import (P8.1)"
            )
    return fields


def _check_headers(
    ws: Any, header_row: int, fields: Sequence[FieldSpec], template_id: str
) -> None:
    """Fail loudly when the workbook no longer matches the template config."""
    for spec in fields:
        assert spec.excel_col is not None
        actual = ws.cell(row=header_row, column=spec.excel_col).value
        if actual is None or _norm(actual) != _norm(spec.excel_header):
            raise OutputError(
                f"workbook header at column {spec.excel_col} does not match "
                f"template {template_id!r} field {spec.key!r}; "
                "re-sync the template (P8.1)"
            )


def write_workbook(
    template_xlsx: Path,
    out_xlsx: Path,
    template: TemplateConfig,
    results: Sequence[ExtractionResult],
    output: OutputConfig | None = None,
    *,
    sheet: str | None = None,
    header_row: int | None = None,
    reviewed: Collection[tuple[str, int, str]] = frozenset(),
) -> WriteSummary:
    """Write ``results`` into a copy of ``template_xlsx`` saved at ``out_xlsx``.

    ``reviewed`` holds ``(document_id, row index within that result, column_key)``
    for fields a human decided (P9.3): they get the reviewed colour and a
    ``yes`` in the evidence sheet's ``reviewed`` column.
    """
    from openpyxl import load_workbook  # type: ignore[import-untyped]
    from openpyxl.styles import Font, PatternFill  # type: ignore[import-untyped]

    cfg = output if output is not None else OutputConfig()
    if out_xlsx.resolve() == template_xlsx.resolve():
        raise OutputError("output path must differ from the client template path")
    for result in results:
        if result.template_id != template.id:
            raise OutputError(
                f"result for template {result.template_id!r} cannot be written "
                f"with template {template.id!r}"
            )

    fields = _engine_fields(template)
    headers = read_headers(template_xlsx, sheet=sheet, header_row=header_row)
    try:
        wb = load_workbook(template_xlsx, keep_vba=template_xlsx.suffix == ".xlsm")
    except Exception as exc:  # openpyxl raises many types on a bad file
        raise ConfigError(
            f"cannot open workbook {template_xlsx.name}: {type(exc).__name__}"
        ) from exc
    if cfg.evidence_sheet in wb.sheetnames:
        raise OutputError(
            f"template already has a sheet named {cfg.evidence_sheet!r}; "
            "change output.evidence_sheet"
        )
    ws = wb[headers.sheet]
    _check_headers(ws, headers.header_row, fields, template.id)

    fills = {
        status: PatternFill("solid", start_color=code, end_color=code)
        for status, code in cfg.colors.items()
    }
    reviewed_fill = PatternFill(
        "solid", start_color=cfg.reviewed_color, end_color=cfg.reviewed_color
    )
    allowed = set(cfg.write_statuses)
    ev = wb.create_sheet(cfg.evidence_sheet)
    for c, name in enumerate(_EVIDENCE_HEADERS, start=1):
        ev.cell(row=1, column=c, value=name).font = Font(bold=True)

    next_row = cfg.data_start_row or headers.header_row + 1
    counts = {WRITTEN: 0, SKIPPED_PROTECTED: 0, SKIPPED_STATUS: 0, SKIPPED_EMPTY: 0}
    ev_row = 2
    n_rows = 0
    for result in results:
        for row_index, row in enumerate(result.rows):
            for spec in fields:
                assert spec.excel_col is not None
                fv = row.get(spec.key) or FieldValue(
                    column_key=spec.key,
                    status=FieldStatus.NOT_FOUND,
                    confidence=0.0,
                    reason="column absent from result",
                )
                cell = ws.cell(row=next_row, column=spec.excel_col)
                value = fv.value if fv.value not in (None, "") else None
                if value is None:
                    state = SKIPPED_EMPTY
                elif fv.status not in allowed:
                    state = SKIPPED_STATUS
                elif cell.value not in (None, "") and not cfg.overwrite_existing:
                    state = SKIPPED_PROTECTED
                else:
                    state = WRITTEN
                # Colour only cells the engine actually owns this run: an empty
                # cell for not_found, or a cell it wrote.
                colour_ok = state == WRITTEN or (
                    state in (SKIPPED_EMPTY, SKIPPED_STATUS)
                    and cell.value in (None, "")
                )
                if state == WRITTEN:
                    assert value is not None
                    _set_text(cell, value)
                is_reviewed = (result.document_id, row_index, spec.key) in reviewed
                if colour_ok:
                    cell.fill = reviewed_fill if is_reviewed else fills[fv.status]
                counts[state] += 1

                values: list[Any] = [
                    result.document_id,
                    next_row,
                    spec.key,
                    spec.excel_header,
                    cell.coordinate,
                    value,
                    fv.status.value,
                    fv.confidence,
                    state,
                    fv.reason,
                    "\n".join(format_evidence(e) for e in fv.evidence) or None,
                    "yes" if is_reviewed else None,
                ]
                for c, item in enumerate(values, start=1):
                    target = ev.cell(row=ev_row, column=c)
                    if isinstance(item, str):
                        _set_text(target, item)
                    else:
                        target.value = item
                ev_row += 1
            next_row += 1
            n_rows += 1

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)
    return WriteSummary(
        path=out_xlsx,
        sheet=headers.sheet,
        rows=n_rows,
        written=counts[WRITTEN],
        skipped_protected=counts[SKIPPED_PROTECTED],
        skipped_status=counts[SKIPPED_STATUS],
        skipped_empty=counts[SKIPPED_EMPTY],
    )


def write_blank_template(template: TemplateConfig, path: Path) -> Path:
    """Header-only workbook from the template config.

    Used when no client workbook is supplied (for example before the client's
    Excel arrives). Column positions come from ``excel_col``.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = template.id[:31]
    for spec in template.fields:
        if spec.excel_col is None:
            raise OutputError(
                f"template {template.id!r} field {spec.key!r} has no excel_col"
            )
        cell = ws.cell(row=1, column=spec.excel_col)
        _set_text(cell, spec.excel_header)
        cell.font = Font(bold=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
