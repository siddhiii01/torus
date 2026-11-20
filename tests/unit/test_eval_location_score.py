"""Unit tests for field-location scoring (P4.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torus.core.errors import EvalError
from torus.core.models import BBox
from torus.core.types import FieldStatus, RegionKind
from torus.eval.location_score import (
    aggregate_locations,
    score_locations,
)
from torus.eval.locations import (
    load_locations_document,
    resolve_location_prediction_path,
)
from torus.eval.metrics import (
    CaseScore,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
)
from torus.eval.report import (
    RunVersions,
    TimingSummary,
    build_run_report,
    render_html,
    render_markdown,
)
from torus.eval.schema import FieldLocation, LocationsDocument


def _box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _loc(
    key: str,
    *,
    page: int = 0,
    row: int = 0,
    bbox: BBox | None = None,
) -> FieldLocation:
    return FieldLocation(
        column_key=key,
        page_index=page,
        row_index=row,
        bbox=bbox or _box(0.1, 0.1, 0.3, 0.2),
        kind=RegionKind.FIELD_VALUE,
    )


def _empty_field_score() -> CaseScore:
    return CaseScore(
        case_id="aggregate",
        template_id="starter_v1",
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
            counts={s.value: 0 for s in FieldStatus},
            rates={s.value: 0.0 for s in FieldStatus},
            automatic_rate=0.0,
            review_rate=0.0,
        ),
    )


def test_perfect_location_metrics() -> None:
    gold = LocationsDocument(
        fields=[
            _loc("mobile_no", bbox=_box(0.2, 0.2, 0.4, 0.25)),
            _loc("person_name", bbox=_box(0.2, 0.1, 0.5, 0.15)),
        ]
    )
    pred = LocationsDocument(
        fields=[
            _loc("mobile_no", bbox=_box(0.21, 0.205, 0.39, 0.248)),
            _loc("person_name", bbox=_box(0.205, 0.102, 0.49, 0.148)),
        ]
    )
    metrics = score_locations(gold, pred, iou_threshold=0.3)
    assert metrics.expected == 2
    assert metrics.located == 2
    assert metrics.located_rate == 1.0
    assert metrics.unmatched_pred == 0
    assert metrics.mean_iou > 0.5
    assert metrics.missed_columns == []
    assert {c.column_key for c in metrics.per_column} == {"mobile_no", "person_name"}


def test_miss_and_wrong_page() -> None:
    gold = LocationsDocument(
        fields=[
            _loc("mobile_no", page=0, bbox=_box(0.2, 0.2, 0.4, 0.25)),
            _loc("person_name", page=0, bbox=_box(0.2, 0.1, 0.5, 0.15)),
        ]
    )
    pred = LocationsDocument(
        fields=[
            # Overlaps mobile but on the wrong page → miss.
            _loc("mobile_no", page=1, bbox=_box(0.2, 0.2, 0.4, 0.25)),
            # Far from person_name → miss.
            _loc("person_name", bbox=_box(0.7, 0.7, 0.9, 0.8)),
        ]
    )
    metrics = score_locations(gold, pred, iou_threshold=0.3)
    assert metrics.located == 0
    assert metrics.located_rate == 0.0
    assert set(metrics.missed_columns) == {"mobile_no", "person_name"}
    assert metrics.unmatched_pred == 2


def test_unmatched_extra_prediction() -> None:
    gold = LocationsDocument(fields=[_loc("mobile_no")])
    pred = LocationsDocument(
        fields=[
            _loc("mobile_no"),
            _loc("notes", bbox=_box(0.5, 0.5, 0.7, 0.6)),
        ]
    )
    metrics = score_locations(gold, pred, iou_threshold=0.3)
    assert metrics.located == 1
    assert metrics.unmatched_pred == 1
    notes = next(c for c in metrics.per_column if c.column_key == "notes")
    assert notes.expected == 0
    assert notes.unmatched_pred == 1


def test_row_index_disambiguates() -> None:
    gold = LocationsDocument(
        fields=[
            _loc("mobile_no", row=0, bbox=_box(0.1, 0.1, 0.3, 0.2)),
            _loc("mobile_no", row=1, bbox=_box(0.1, 0.5, 0.3, 0.6)),
        ]
    )
    pred = LocationsDocument(
        fields=[
            _loc("mobile_no", row=0, bbox=_box(0.1, 0.1, 0.3, 0.2)),
            # Wrong row for the second gold.
            _loc("mobile_no", row=0, bbox=_box(0.1, 0.5, 0.3, 0.6)),
        ]
    )
    metrics = score_locations(gold, pred, iou_threshold=0.3)
    assert metrics.located == 1
    assert metrics.expected == 2
    assert "mobile_no" in metrics.missed_columns


def test_aggregate_locations() -> None:
    a = score_locations(
        LocationsDocument(fields=[_loc("a")]),
        LocationsDocument(fields=[_loc("a")]),
    )
    b = score_locations(
        LocationsDocument(fields=[_loc("b"), _loc("c")]),
        LocationsDocument(fields=[_loc("b")]),
    )
    agg = aggregate_locations([a, b])
    assert agg.expected == 3
    assert agg.located == 2
    assert agg.located_rate == pytest.approx(2 / 3, abs=1e-3)
    assert "c" in agg.missed_columns


def test_report_includes_location_section() -> None:
    metrics = score_locations(
        LocationsDocument(fields=[_loc("mobile_no")]),
        LocationsDocument(fields=[_loc("mobile_no")]),
    )
    report = build_run_report(
        set_id="mini",
        aggregate=_empty_field_score(),
        case_ids=["case_a"],
        timings=TimingSummary(total_ms=1.0, case_count=1),
        versions=RunVersions(config_hash="abc", template_id="starter_v1"),
        location=metrics,
    )
    md = render_markdown(report)
    assert "## Field location" in md
    assert "Located rate" in md
    assert "`mobile_no`" in md
    assert "9876543210" not in md

    html = render_html(report)
    assert "Field location" in html
    assert "mobile_no" in html


def test_load_locations_document(tmp_path: Path) -> None:
    path = tmp_path / "locations.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "pack_x",
                "fields": [
                    {
                        "column_key": "mobile_no",
                        "page_index": 0,
                        "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    doc = load_locations_document(path, case_id="pack_x")
    assert len(doc.fields) == 1
    assert doc.fields[0].column_key == "mobile_no"


def test_load_locations_rejects_case_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "locations.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "other",
                "fields": [
                    {
                        "column_key": "mobile_no",
                        "page_index": 0,
                        "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvalError, match="case_id"):
        load_locations_document(path, case_id="pack_x")


def test_resolve_location_prediction_path(tmp_path: Path) -> None:
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "pack_x.locations.json").write_text("{}", encoding="utf-8")
    assert resolve_location_prediction_path(pred_dir, "pack_x") is not None
    assert resolve_location_prediction_path(pred_dir, "missing") is None


def test_field_location_rejects_empty_key() -> None:
    with pytest.raises(ValidationError):
        FieldLocation(
            column_key="  ",
            page_index=0,
            bbox=_box(0.1, 0.1, 0.2, 0.2),
        )
