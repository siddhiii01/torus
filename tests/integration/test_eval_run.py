"""End-to-end eval harness on the synthetic mini test set (P1.6)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.synth.testset import build_mini_testset

from torus.eval import load_run_report, run_evaluation
from torus.eval.__main__ import main as eval_main
from torus.eval.schema import Split


def _write_identity_preds(data_dir: Path, pred_dir: Path) -> None:
    """Write predictions that match expected.json (perfect score)."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    set_root = data_dir / "testsets" / "mini"
    for case_dir in sorted(set_root.iterdir()):
        if not case_dir.is_dir():
            continue
        expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
        payload = {"case_id": case_dir.name, "rows": expected["rows"]}
        (pred_dir / f"{case_dir.name}.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )


def test_run_evaluation_perfect_on_mini(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_identity_preds(data_dir, pred_dir)
    report_path = tmp_path / "reports" / "eval" / "mini.md"

    result = run_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=report_path,
        data_dir=data_dir,
        split=Split.DEV,
        model_versions={"reader": "identity_fixture"},
    )

    assert result.report.case_count == 1
    assert result.report.scores.fields_micro.accuracy == 1.0
    assert result.report.scores.silent_errors == 0
    assert result.report.scores.rows.f1 == 1.0
    assert result.paths.json_path is not None
    assert result.paths.json_path.is_file()
    assert result.paths.markdown.is_file()

    loaded = load_run_report(result.paths.json_path)
    assert loaded.scores.fields_micro.accuracy == 1.0
    # Metrics-only: report text must not echo synthetic field values.
    md = result.paths.markdown.read_text(encoding="utf-8")
    assert "Synthetic Person" not in md
    assert "9876543210" not in md


def test_eval_run_cli_all_splits(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_identity_preds(data_dir, pred_dir)
    report_path = tmp_path / "out.md"

    code = eval_main(
        [
            "run",
            "--set",
            "mini",
            "--pred-dir",
            str(pred_dir),
            "--report",
            str(report_path),
            "--data-dir",
            str(data_dir),
            "--all-splits",
            "--model-version",
            "backend=identity",
        ]
    )
    assert code == 0
    report = load_run_report(report_path.with_suffix(".json"))
    assert report.case_count == 2
    assert report.scores.silent_errors == 0


def test_run_skips_missing_predictions(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    # Only the hidden case gets a pred; default split=dev should then fail
    # with "no cases scored" after skip — write the dev pred only for one path.
    expected = json.loads(
        (data_dir / "testsets" / "mini" / "case_dev_001" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    (pred_dir / "case_dev_001.json").write_text(
        json.dumps({"rows": expected["rows"]}) + "\n", encoding="utf-8"
    )

    result = run_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=tmp_path / "r.md",
        data_dir=data_dir,
        split=None,
    )
    assert result.report.case_count == 1
    assert result.skipped_missing_pred == ("case_hidden_001",)
