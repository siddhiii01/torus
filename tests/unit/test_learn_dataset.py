"""Unit tests for the corrections → training-set builder (P10.1). Synthetic only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from torus.core.models import BBox, Evidence
from torus.core.types import FieldStatus
from torus.learn.dataset import (
    LearnError,
    SampleKind,
    Split,
    assign_split,
    build_dataset,
    effective_label,
)
from torus.review.store import CorrectionRecord

T0 = datetime(2026, 1, 1, tzinfo=UTC)
BOX = BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.3)


def _rec(
    doc: str,
    item: str,
    *,
    action: str = "correct",
    value: str | None = "ABC",
    suggestion: str | None = None,
    refs: int = 1,
    at: datetime = T0,
) -> CorrectionRecord:
    return CorrectionRecord(
        document_id=doc,
        item_id=item,
        row=0,
        column_key="col",
        action=action,  # type: ignore[arg-type]
        value=value,
        suggestion=suggestion,
        status_before=FieldStatus.NEEDS_REVIEW,
        crop_refs=[
            Evidence(region_id=f"r{i}", page_index=0, bbox=BOX) for i in range(refs)
        ],
        reviewer="tester",
        decided_at=at,
    )


def _pages(_doc: str):  # type: ignore[no-untyped-def]
    image = Image.new("RGB", (100, 100), "white")
    return lambda _i: image


def _by_doc(*recs: CorrectionRecord) -> dict[str, dict[str, CorrectionRecord]]:
    out: dict[str, dict[str, CorrectionRecord]] = {}
    for r in recs:
        out.setdefault(r.document_id, {})[r.item_id] = r
    return out


def test_split_is_deterministic_and_by_document() -> None:
    a = assign_split("doc-1", seed=3)
    assert a == assign_split("doc-1", seed=3)
    splits = {
        assign_split(f"d{i}", val_fraction=0.2, test_fraction=0.2) for i in range(200)
    }
    assert splits == set(Split)


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(LearnError):
        assign_split("d", val_fraction=0.6, test_fraction=0.5)


def test_effective_label_rules() -> None:
    assert effective_label(_rec("d", "1:a", value="X"), include_accepted=True) == "X"
    acc = _rec("d", "1:a", action="accept", value=None, suggestion="S")
    assert effective_label(acc, include_accepted=True) == "S"
    assert effective_label(acc, include_accepted=False) is None
    assert (
        effective_label(_rec("d", "1:a", action="skip"), include_accepted=True) is None
    )
    assert effective_label(_rec("d", "1:a", value="  "), include_accepted=True) is None


def test_line_dataset_writes_files_and_hash(tmp_path: Path) -> None:
    recs = _by_doc(_rec("d1", "0:a", value="HELLO"), _rec("d2", "0:a", value="WORLD"))
    m1 = build_dataset(recs, _pages, tmp_path / "a", kind=SampleKind.LINE)
    m2 = build_dataset(recs, _pages, tmp_path / "b", kind=SampleKind.LINE)
    assert m1.data_hash == m2.data_hash  # reproducible
    assert sum(m1.counts.values()) == 2
    for s in m1.samples:
        assert (tmp_path / "a" / s.image).is_file()
    rows = [
        json.loads(line)
        for sp in Split
        for line in (tmp_path / "a" / f"{sp.value}.jsonl").read_text().splitlines()
    ]
    assert {r["text"] for r in rows} == {"HELLO", "WORLD"}


def test_no_document_in_two_splits(tmp_path: Path) -> None:
    recs = _by_doc(*[_rec(f"d{i}", f"{j}:a") for i in range(30) for j in range(3)])
    m = build_dataset(
        recs,
        _pages,
        tmp_path,
        kind=SampleKind.LINE,
        val_fraction=0.3,
        test_fraction=0.3,
    )
    seen: dict[str, Split] = {}
    for s in m.samples:
        assert seen.setdefault(s.document_id, s.split) == s.split


def test_skips_are_counted(tmp_path: Path) -> None:
    recs = _by_doc(
        _rec("d1", "0:a", refs=2),
        _rec("d1", "1:a", action="skip"),
        _rec("d1", "2:a", at=datetime(2025, 1, 1, tzinfo=UTC)),
        _rec("d1", "3:a"),
        _rec("hidden", "0:a"),
    )
    m = build_dataset(
        recs,
        _pages,
        tmp_path,
        kind=SampleKind.LINE,
        since=T0,
        exclude_documents=["hidden"],
    )
    assert len(m.samples) == 1
    assert m.skipped == {
        "not_single_evidence": 1,
        "no_usable_label": 1,
        "before_since": 1,
        "excluded_document": 1,
    }


def test_char_cell_needs_one_cell_per_character(tmp_path: Path) -> None:
    def splitter(_img: object, _box: BBox) -> list[Image.Image]:
        return [Image.new("RGB", (8, 8), "white") for _ in range(3)]

    recs = _by_doc(_rec("d1", "0:a", value="AB1"), _rec("d1", "1:a", value="ABCD"))
    m = build_dataset(
        recs, _pages, tmp_path, kind=SampleKind.CHAR_CELL, cell_splitter=splitter
    )
    assert sorted(s.label for s in m.samples) == ["1", "A", "B"]
    assert m.skipped == {"cell_count_mismatch": 1}


def test_vlm_rows_carry_field_and_target(tmp_path: Path) -> None:
    recs = _by_doc(_rec("d1", "0:a", value="XYZ"))
    m = build_dataset(
        recs,
        _pages,
        tmp_path,
        kind=SampleKind.FIELD_VLM,
        val_fraction=0,
        test_fraction=0,
    )
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert row["target"] == "XYZ" and row["field"] == "col"
    assert m.counts["train"] == 1
