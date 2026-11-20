"""Location eval harness on the synthetic mini set (P4.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures.synth.testset import build_mini_testset

from torus.eval import load_run_report, run_evaluation, run_location_evaluation
from torus.eval.__main__ import main as eval_main
from torus.eval.schema import Split


def _write_location_preds(
    data_dir: Path,
    pred_dir: Path,
    *,
    shift_mobile: bool = False,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    set_root = data_dir / "testsets" / "mini"
    for case_dir in sorted(set_root.iterdir()):
        if not case_dir.is_dir():
            continue
        gold = json.loads((case_dir / "locations.json").read_text(encoding="utf-8"))
        fields = []
        for item in gold["fields"]:
            bbox = dict(item["bbox"])
            if shift_mobile and item["column_key"] == "mobile_no":
                # Push the box far away so IoU fails.
                bbox = {"x0": 0.80, "y0": 0.80, "x1": 0.95, "y1": 0.90}
            fields.append(
                {
                    "column_key": item["column_key"],
                    "row_index": item.get("row_index", 0),
                    "page_index": item["page_index"],
                    "bbox": bbox,
                    "kind": item.get("kind"),
                }
            )
        payload = {"case_id": case_dir.name, "fields": fields}
        (pred_dir / f"{case_dir.name}.locations.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )


def _write_field_preds(data_dir: Path, pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted((data_dir / "testsets" / "mini").iterdir()):
        if not case_dir.is_dir():
            continue
        expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
        (pred_dir / f"{case_dir.name}.json").write_text(
            json.dumps({"rows": expected["rows"]}) + "\n", encoding="utf-8"
        )


def test_run_location_evaluation_detects_miss(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_location_preds(data_dir, pred_dir, shift_mobile=True)
    report_path = tmp_path / "reports" / "locate.md"

    result = run_location_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=report_path,
        data_dir=data_dir,
        split=Split.DEV,
        model_versions={"layout": "synth_shift"},
    )

    loc = result.report.location
    assert loc is not None
    assert loc.expected == 3
    assert loc.located == 2
    assert loc.located_rate == pytest.approx(2 / 3, abs=1e-3)
    assert "mobile_no" in loc.missed_columns

    md = result.paths.markdown.read_text(encoding="utf-8")
    assert "Field location" in md
    assert "Located rate" in md
    assert "mobile_no" in md
    assert "Synthetic Person" not in md
    assert "9876543210" not in md


def test_eval_locate_cli(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_location_preds(data_dir, pred_dir, shift_mobile=False)
    report_path = tmp_path / "out.md"

    code = eval_main(
        [
            "locate",
            "--set",
            "mini",
            "--pred-dir",
            str(pred_dir),
            "--report",
            str(report_path),
            "--data-dir",
            str(data_dir),
            "--all-splits",
        ]
    )
    assert code == 0
    report = load_run_report(report_path.with_suffix(".json"))
    assert report.location is not None
    assert report.location.located_rate == 1.0
    assert report.case_count == 2


def test_run_evaluation_includes_location_when_preds_present(
    tmp_path: Path,
) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_field_preds(data_dir, pred_dir)
    _write_location_preds(data_dir, pred_dir, shift_mobile=False)

    result = run_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=tmp_path / "both.md",
        data_dir=data_dir,
        split=Split.DEV,
    )
    assert result.report.scores.fields_micro.accuracy == 1.0
    assert result.report.location is not None
    assert result.report.location.located_rate == 1.0
    assert result.cases[0].location_scored is True
