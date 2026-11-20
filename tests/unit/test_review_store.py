"""P9.2 correction store and P9.3 re-export (synthetic data only)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from torus.config.schemas import FieldSpec, OutputConfig, TemplateConfig
from torus.core.models import BBox, Evidence, ExtractionResult, FieldValue
from torus.core.types import FieldStatus, FieldType
from torus.review.queue import ReviewError, build_queue
from torus.review.reexport import apply_corrections, reexport
from torus.review.server import Decision, ReviewApp
from torus.review.store import CorrectionStore

_EV = Evidence(region_id="r1", page_index=1, bbox=BBox(x0=0.1, y0=0.2, x1=0.4, y1=0.3))
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _fv(
    key: str, value: str | None, status: FieldStatus, conf: float = 0.6
) -> FieldValue:
    return FieldValue(
        column_key=key,
        value=value,
        status=status,
        confidence=conf,
        evidence=[_EV] if value else [],
    )


def _result() -> ExtractionResult:
    return ExtractionResult(
        document_id="doc1",
        template_id="t1",
        config_hash="h",
        rows=[
            {
                "name": _fv("name", "alpha", FieldStatus.NEEDS_REVIEW),
                "acct": _fv("acct", "12345", FieldStatus.LIKELY, 0.8),
            },
            {"name": _fv("name", None, FieldStatus.NOT_FOUND, 0.0)},
        ],
    )


def _store(tmp_path: Path) -> CorrectionStore:
    return CorrectionStore(tmp_path / "corr", "rev1", clock=lambda: _T0)


def _items() -> dict[str, object]:
    return {i.item_id: i for i in build_queue(_result())}


def test_record_persists_crop_value_reviewer_time(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = _items()["0:name"]
    store.record(item, Decision("0:name", "correct", "beta"))  # type: ignore[arg-type]
    (rec,) = store.load("doc1")
    assert (rec.value, rec.suggestion, rec.reviewer) == ("beta", "alpha", "rev1")
    assert rec.decided_at == _T0
    assert rec.crop_refs == [_EV]
    assert rec.status_before == FieldStatus.NEEDS_REVIEW


def test_latest_decision_wins_and_history_is_kept(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = _items()["0:name"]
    store.record(item, Decision("0:name", "skip", None))  # type: ignore[arg-type]
    store.record(item, Decision("0:name", "accept", "alpha"))  # type: ignore[arg-type]
    assert len(store.load("doc1")) == 2
    assert store.latest("doc1")["0:name"].action == "accept"


def test_missing_document_is_empty(tmp_path: Path) -> None:
    assert _store(tmp_path).load("nope") == []


def test_unsafe_document_id_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for bad in ("../x", "a/b", "", ".hidden", "a\\b"):
        with pytest.raises(ReviewError):
            store.load(bad)


def test_corrupt_line_names_file_and_line_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = tmp_path / "corr" / "doc1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("secret-content\n", encoding="utf-8")
    with pytest.raises(ReviewError) as exc:
        store.load("doc1")
    assert "doc1.jsonl line 1" in str(exc.value)
    assert "secret-content" not in str(exc.value)


def test_blank_reviewer_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReviewError):
        CorrectionStore(tmp_path, "  ")


def test_from_env_requires_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TORUS_DATA_DIR", raising=False)
    with pytest.raises(ReviewError):
        CorrectionStore.from_env("r")
    monkeypatch.setenv("TORUS_DATA_DIR", str(tmp_path))
    store = CorrectionStore.from_env("r")
    store.record(_items()["0:name"], Decision("0:name", "accept", "alpha"))  # type: ignore[arg-type]
    assert (tmp_path / "corrections" / "doc1.jsonl").is_file()


def test_review_app_persists_through_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    app = ReviewApp(list(build_queue(_result())), lambda _i: None, store)
    app.decide("0:name", "correct", "beta")
    assert store.latest("doc1")["0:name"].value == "beta"


def test_apply_corrections_semantics(tmp_path: Path) -> None:
    store = _store(tmp_path)
    items = _items()
    store.record(items["0:name"], Decision("0:name", "correct", "beta"))  # type: ignore[arg-type]
    store.record(items["0:acct"], Decision("0:acct", "accept", "12345"))  # type: ignore[arg-type]
    original = _result()
    applied = apply_corrections(original, store.latest("doc1"))
    row = applied.result.rows[0]
    assert row["name"].value == "beta"
    assert row["name"].status == FieldStatus.CONFIRMED
    assert row["name"].evidence == [_EV]
    assert row["acct"].value == "12345"
    assert applied.reviewed == {("doc1", 0, "name"), ("doc1", 0, "acct")}
    assert (applied.corrected, applied.accepted) == (1, 1)
    assert original.rows[0]["name"].value == "alpha"  # input untouched


def test_skip_leaves_field_unchanged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_items()["0:name"], Decision("0:name", "skip", None))  # type: ignore[arg-type]
    applied = apply_corrections(_result(), store.latest("doc1"))
    assert applied.result == _result()
    assert applied.reviewed == frozenset()
    assert applied.skipped == 1


def test_stale_decision_is_unapplied_not_forced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(_items()["0:name"], Decision("0:name", "correct", "beta"))  # type: ignore[arg-type]
    changed = _result().model_copy(
        update={
            "rows": [
                {"name": _fv("name", "different", FieldStatus.NEEDS_REVIEW)},
                {},
            ]
        }
    )
    applied = apply_corrections(changed, store.latest("doc1"))
    assert applied.unapplied == ("0:name",)
    assert applied.result.rows[0]["name"].value == "different"


def test_correction_without_evidence_is_unapplied(tmp_path: Path) -> None:
    result = _result().model_copy(
        update={
            "rows": [
                {"name": _fv("name", None, FieldStatus.NEEDS_REVIEW)},
            ]
        }
    )
    store = _store(tmp_path)
    (item,) = build_queue(result)
    store.record(item, Decision(item.item_id, "correct", "beta"))
    applied = apply_corrections(result, store.latest("doc1"))
    assert applied.unapplied == (item.item_id,)
    assert applied.result.rows[0]["name"].value is None


def _template() -> TemplateConfig:
    def spec(key: str, header: str, col: int) -> FieldSpec:
        return FieldSpec(
            key=key,
            excel_header=header,
            excel_col=col,
            description=header,
            type=FieldType.TEXT,
        )

    return TemplateConfig(
        id="t1", fields=[spec("name", "Full Name", 1), spec("acct", "Account", 2)]
    )


def _client(path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws["A1"], ws["B1"] = "Full Name", "Account"
    wb.save(path)
    return path


def test_reexport_marks_reviewed_in_cells_and_evidence_sheet(tmp_path: Path) -> None:
    store = _store(tmp_path)
    items = _items()
    store.record(items["0:name"], Decision("0:name", "correct", "beta"))  # type: ignore[arg-type]
    out = tmp_path / "out.xlsx"
    cfg = OutputConfig()
    summary = reexport(
        _client(tmp_path / "c.xlsx"), out, _template(), [_result()], store, cfg
    )
    assert (summary.corrected, summary.accepted, summary.unapplied) == (1, 0, ())

    wb = load_workbook(out)
    ws = wb.worksheets[0]
    assert ws["A2"].value == "beta"
    assert ws["A2"].fill.start_color.rgb.endswith(cfg.reviewed_color)
    assert ws["B2"].value == "12345"  # undecided likely keeps its status colour
    assert ws["B2"].fill.start_color.rgb.endswith(cfg.colors[FieldStatus.LIKELY])

    ev = wb[cfg.evidence_sheet]
    header = [c.value for c in ev[1]]
    col = header.index("reviewed")
    flags = {(r[2].value, r[1].value): r[col].value for r in ev.iter_rows(min_row=2)}
    assert flags[("name", 2)] == "yes"
    assert flags[("acct", 2)] is None


def test_reexport_without_decisions_matches_plain_write(tmp_path: Path) -> None:
    out = tmp_path / "out.xlsx"
    summary = reexport(
        _client(tmp_path / "c.xlsx"), out, _template(), [_result()], _store(tmp_path)
    )
    assert summary.corrected == summary.accepted == 0
    assert load_workbook(out).worksheets[0]["A2"].value == "alpha"


def test_reviewed_color_must_be_hex() -> None:
    with pytest.raises(ValueError):
        OutputConfig(reviewed_color="red")
