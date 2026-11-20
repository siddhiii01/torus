"""Classification eval harness on the synthetic mini set (P3.5)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.synth.testset import build_mini_testset

from torus.eval import load_run_report, run_classification_evaluation, run_evaluation
from torus.eval.__main__ import main as eval_main
from torus.eval.schema import Split


def _write_page_preds(
    data_dir: Path,
    pred_dir: Path,
    *,
    flip_id_card: bool = False,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    set_root = data_dir / "testsets" / "mini"
    for case_dir in sorted(set_root.iterdir()):
        if not case_dir.is_dir():
            continue
        gold = json.loads((case_dir / "pages.json").read_text(encoding="utf-8"))
        pages = []
        for page in gold["pages"]:
            page_type = page["page_type"]
            if flip_id_card and page_type == "id_card":
                page_type = "other"
            pages.append({"index": page["index"], "page_type": page_type})
        payload = {"case_id": case_dir.name, "pages": pages}
        (pred_dir / f"{case_dir.name}.pages.json").write_text(
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


def test_run_classification_evaluation_confusion(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_page_preds(data_dir, pred_dir, flip_id_card=True)
    report_path = tmp_path / "reports" / "clf.md"

    result = run_classification_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=report_path,
        data_dir=data_dir,
        split=Split.DEV,
        model_versions={"classifier": "synth_flip"},
    )

    clf = result.report.classification
    assert clf is not None
    assert clf.total == 2
    assert clf.correct == 1
    assert clf.accuracy == 0.5
    assert clf.confusion is not None
    assert any(m.gold_type == "id_card" for m in clf.misclassifications)

    md = result.paths.markdown.read_text(encoding="utf-8")
    assert "Confusion matrix" in md
    assert "Misclassifications by gold type" in md
    assert "Precision" in md
    assert "Synthetic Person" not in md


def test_eval_classify_cli(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_page_preds(data_dir, pred_dir, flip_id_card=False)
    report_path = tmp_path / "out.md"

    code = eval_main(
        [
            "classify",
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
    assert report.classification is not None
    assert report.classification.accuracy == 1.0
    assert report.case_count == 2


def test_run_evaluation_includes_classification_when_preds_present(
    tmp_path: Path,
) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    pred_dir = tmp_path / "preds"
    _write_field_preds(data_dir, pred_dir)
    _write_page_preds(data_dir, pred_dir, flip_id_card=False)

    result = run_evaluation(
        set_id="mini",
        pred_dir=pred_dir,
        report_path=tmp_path / "both.md",
        data_dir=data_dir,
        split=Split.DEV,
    )
    assert result.report.scores.fields_micro.accuracy == 1.0
    assert result.report.classification is not None
    assert result.report.classification.accuracy == 1.0
    assert result.cases[0].classification_scored is True
