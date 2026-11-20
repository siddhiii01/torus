"""P8.5: Excel writer (synthetic workbooks and synthetic values only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from torus.config.schemas import FieldSpec, OutputConfig, TemplateConfig
from torus.core.errors import OutputError
from torus.core.models import BBox, Evidence, ExtractionResult, FieldValue
from torus.core.types import FieldStatus, FieldType
from torus.output.excel import (
    SKIPPED_PROTECTED,
    SKIPPED_STATUS,
    WRITTEN,
    format_evidence,
    write_workbook,
)

_EV = Evidence(region_id="r1", page_index=2, bbox=BBox(x0=0.1, y0=0.2, x1=0.4, y1=0.25))


def _fv(
    key: str,
    value: str | None,
    status: FieldStatus = FieldStatus.CONFIRMED,
    conf: float = 0.995,
    reason: str | None = None,
) -> FieldValue:
    return FieldValue(
        column_key=key,
        value=value,
        status=status,
        confidence=conf,
        evidence=[_EV] if value else [],
        reason=reason,
    )


def _spec(key: str, header: str, col: int, **kw: object) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=header,
        excel_col=col,
        description=header,
        type=FieldType.TEXT,
        **kw,  # type: ignore[arg-type]
    )


def _template() -> TemplateConfig:
    return TemplateConfig(
        id="t1",
        fields=[
            _spec("sys", "System ID", 1, passthrough=True),
            _spec("name", "Full Name", 2),
            _spec("acct", "Account", 3),
        ],
    )


def _client(path: Path, *, prefill: bool = False) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Main"
    ws["A1"], ws["B1"], ws["C1"] = "System ID", "Full  Name", "Account"
    ws["B1"].font = Font(bold=True, italic=True)
    ws.column_dimensions["B"].width = 33
    if prefill:
        ws["A2"] = "SYS-1"
        ws["B2"] = "keep me"
    wb.save(path)
    return path


def _result(rows: list[dict[str, FieldValue]], doc: str = "doc1") -> ExtractionResult:
    return ExtractionResult(
        document_id=doc, template_id="t1", rows=rows, config_hash="h"
    )


def test_writes_values_colours_and_keeps_formatting(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    out = tmp_path / "out" / "o.xlsx"
    rows = [
        {
            "name": _fv("name", "A Person"),
            "acct": _fv("acct", "00123", FieldStatus.LIKELY, 0.9),
        }
    ]
    summary = write_workbook(src, out, _template(), [_result(rows)], OutputConfig())
    assert summary.written == 2 and summary.rows == 1

    ws = load_workbook(out)["Main"]
    assert ws["B2"].value == "A Person"
    assert ws["C2"].value == "00123"  # leading zeros kept (text)
    assert ws["A2"].value is None  # passthrough never written
    assert ws["B2"].fill.start_color.rgb.endswith("C6EFCE")
    assert ws["C2"].fill.start_color.rgb.endswith("FFEB9C")
    assert ws["B1"].font.italic and ws.column_dimensions["B"].width == 33
    # source untouched
    assert load_workbook(src)["Main"]["B2"].value is None


def test_evidence_sheet_has_status_confidence_page_and_box(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    out = tmp_path / "o.xlsx"
    rows = [
        {
            "name": _fv("name", "A Person"),
            "acct": _fv("acct", None, FieldStatus.NOT_FOUND, 0.0, "no label"),
        }
    ]
    write_workbook(src, out, _template(), [_result(rows)])
    ev = load_workbook(out)["torus_evidence"]
    header = [c.value for c in ev[1]]
    recs = [
        dict(zip(header, (c.value for c in r), strict=True))
        for r in ev.iter_rows(min_row=2)
    ]
    assert [r["column_key"] for r in recs] == ["name", "acct"]  # passthrough absent
    name, acct = recs
    assert name["status"] == "confirmed" and name["confidence"] == 0.995
    assert name["cell"] == "B2" and name["written"] == WRITTEN
    assert name["evidence"] == "p3 [0.100,0.200,0.400,0.250] r1"
    assert acct["status"] == "not_found" and acct["note"] == "no label"
    assert acct["evidence"] is None


def test_protected_cells_untouched_unless_config_allows(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx", prefill=True)
    rows = [{"name": _fv("name", "New Value"), "acct": _fv("acct", "9")}]

    out = tmp_path / "a.xlsx"
    s = write_workbook(src, out, _template(), [_result(rows)], OutputConfig())
    ws = load_workbook(out)["Main"]
    assert ws["B2"].value == "keep me" and ws["C2"].value == "9"
    assert s.skipped_protected == 1
    ev = load_workbook(out)["torus_evidence"]
    assert SKIPPED_PROTECTED in [c.value for c in ev["I"]]

    out2 = tmp_path / "b.xlsx"
    write_workbook(
        src, out2, _template(), [_result(rows)], OutputConfig(overwrite_existing=True)
    )
    ws2 = load_workbook(out2)["Main"]
    assert ws2["B2"].value == "New Value"
    assert ws2["A2"].value == "SYS-1"  # passthrough protected even with overwrite


def test_status_policy_skips_value_but_lists_it(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    rows = [{"name": _fv("name", "Maybe", FieldStatus.NEEDS_REVIEW, 0.4, "disagree")}]
    out = tmp_path / "o.xlsx"
    cfg = OutputConfig(write_statuses=[FieldStatus.CONFIRMED])
    s = write_workbook(src, out, _template(), [_result(rows)], cfg)
    wb = load_workbook(out)
    assert wb["Main"]["B2"].value is None
    assert wb["Main"]["B2"].fill.start_color.rgb.endswith("FFC7CE")
    assert s.skipped_status == 1
    assert SKIPPED_STATUS in [c.value for c in wb["torus_evidence"]["I"]]


def test_formula_like_value_stays_text(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    rows = [{"name": _fv("name", "=1+1")}]
    out = tmp_path / "o.xlsx"
    write_workbook(src, out, _template(), [_result(rows)])
    cell = load_workbook(out)["Main"]["B2"]
    assert cell.value == "=1+1" and cell.data_type == "s"


def test_multiple_results_stack_rows_in_order(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    r1 = _result([{"name": _fv("name", "One")}, {"name": _fv("name", "Two")}], "d1")
    r2 = _result([{"name": _fv("name", "Three")}], "d2")
    out = tmp_path / "o.xlsx"
    s = write_workbook(src, out, _template(), [r1, r2])
    ws = load_workbook(out)["Main"]
    assert [ws[f"B{i}"].value for i in (2, 3, 4)] == ["One", "Two", "Three"]
    assert s.rows == 3


def test_refuses_overwriting_the_template_itself(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    with pytest.raises(OutputError, match="differ"):
        write_workbook(src, src, _template(), [_result([])])


def test_stale_template_header_mismatch_fails_loudly(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    tpl = TemplateConfig(id="t1", fields=[_spec("name", "Different Header", 2)])
    with pytest.raises(OutputError, match="does not match"):
        write_workbook(src, tmp_path / "o.xlsx", tpl, [_result([])])


def test_template_id_mismatch_and_sheet_clash(tmp_path: Path) -> None:
    src = _client(tmp_path / "in.xlsx")
    bad = ExtractionResult(document_id="d", template_id="other", config_hash="h")
    with pytest.raises(OutputError, match="other"):
        write_workbook(src, tmp_path / "o.xlsx", _template(), [bad])
    wb = load_workbook(src)
    wb.create_sheet("torus_evidence")
    wb.save(src)
    with pytest.raises(OutputError, match="already has a sheet"):
        write_workbook(src, tmp_path / "o2.xlsx", _template(), [_result([])])


def test_format_evidence_is_location_only() -> None:
    assert format_evidence(_EV) == "p3 [0.100,0.200,0.400,0.250] r1"
