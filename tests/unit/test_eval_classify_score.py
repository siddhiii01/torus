"""Unit tests for page-type classification scoring (P3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torus.core.errors import EvalError
from torus.core.types import FieldStatus
from torus.eval.classify_score import (
    MISSING_PRED,
    aggregate_classification,
    pair_page_labels,
    score_page_labels,
    score_page_pairs,
)
from torus.eval.metrics import (
    CaseScore,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
)
from torus.eval.page_labels import (
    load_page_labels_document,
    resolve_page_prediction_path,
)
from torus.eval.report import (
    RunVersions,
    TimingSummary,
    build_run_report,
    render_html,
    render_markdown,
)
from torus.eval.schema import PageLabel, PageLabelsDocument


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


def test_perfect_classification_metrics() -> None:
    gold = PageLabelsDocument(
        pages=[
            PageLabel(index=0, page_type="application_form"),
            PageLabel(index=1, page_type="id_card"),
        ]
    )
    pred = PageLabelsDocument(
        pages=[
            PageLabel(index=0, page_type="application_form", confidence=0.9),
            PageLabel(index=1, page_type="id_card", confidence=0.8),
        ]
    )
    metrics = score_page_labels(
        gold, pred, label_order=["application_form", "id_card", "other"]
    )
    assert metrics.total == 2
    assert metrics.correct == 2
    assert metrics.accuracy == 1.0
    assert metrics.misclassifications == []
    assert metrics.confusion is not None
    by_type = {m.page_type: m for m in metrics.per_type}
    assert by_type["application_form"].precision == 1.0
    assert by_type["application_form"].recall == 1.0
    assert by_type["id_card"].f1 == 1.0


def test_precision_recall_and_confusion() -> None:
    # gold: A, A, B, C
    # pred: A, B, B, A
    pairs = [
        ("A", "A"),
        ("A", "B"),
        ("B", "B"),
        ("C", "A"),
    ]
    metrics = score_page_pairs(pairs, label_order=["A", "B", "C"])
    assert metrics.correct == 2
    assert metrics.accuracy == 0.5
    by_type = {m.page_type: m for m in metrics.per_type}
    # A: tp=1, fp=1 (from C→A), fn=1 (A→B) → P=0.5 R=0.5
    assert by_type["A"].tp == 1
    assert by_type["A"].fp == 1
    assert by_type["A"].fn == 1
    assert by_type["A"].precision == 0.5
    assert by_type["A"].recall == 0.5
    # B: tp=1, fp=1 (A→B), fn=0 → P=0.5 R=1.0
    assert by_type["B"].precision == 0.5
    assert by_type["B"].recall == 1.0
    # C: tp=0, fp=0, fn=1 → P=0 R=0
    assert by_type["C"].precision == 0.0
    assert by_type["C"].recall == 0.0

    assert metrics.confusion is not None
    assert metrics.confusion.labels == ["A", "B", "C"]
    # row A: A→A=1, A→B=1, A→C=0
    assert metrics.confusion.counts[0] == [1, 1, 0]
    # row B: B→A=0, B→B=1, B→C=0
    assert metrics.confusion.counts[1] == [0, 1, 0]
    # row C: C→A=1, C→B=0, C→C=0
    assert metrics.confusion.counts[2] == [1, 0, 0]

    mis = {m.gold_type: m for m in metrics.misclassifications}
    assert mis["A"].total == 1
    assert mis["A"].as_predicted == {"B": 1}
    assert mis["C"].as_predicted == {"A": 1}
    assert "B" not in mis


def test_missing_pred_uses_sentinel() -> None:
    gold = [
        PageLabel(index=0, page_type="blank"),
        PageLabel(index=1, page_type="other"),
    ]
    pred = [PageLabel(index=0, page_type="blank")]
    pairs, ug, up = pair_page_labels(gold, pred)
    assert ug == 1
    assert up == 0
    assert pairs[1] == ("other", MISSING_PRED)
    metrics = score_page_pairs(pairs, unmatched_gold=ug, unmatched_pred=up)
    assert metrics.unmatched_gold == 1
    assert MISSING_PRED in (metrics.confusion.labels if metrics.confusion else [])


def test_aggregate_classification() -> None:
    a = score_page_pairs([("form", "form"), ("id", "other")])
    b = score_page_pairs([("id", "id"), ("form", "form")])
    agg = aggregate_classification([a, b], label_order=["form", "id", "other"])
    assert agg.total == 4
    assert agg.correct == 3
    assert agg.accuracy == 0.75


def test_report_renders_confusion_and_misclass() -> None:
    metrics = score_page_pairs(
        [
            ("application_form", "application_form"),
            ("id_card", "other"),
            ("id_card", "id_card"),
        ],
        label_order=["application_form", "id_card", "other"],
    )
    report = build_run_report(
        set_id="clf_synth",
        split="dev",
        aggregate=_empty_field_score(),
        case_ids=["case_a"],
        classification=metrics,
        timings=TimingSummary(
            total_ms=1.0,
            by_stage={"classify_score": 1.0},
            case_count=1,
            mean_case_ms=1.0,
        ),
        versions=RunVersions(
            config_hash="hash",
            template_id="starter_v1",
            model_versions={"classifier": "synth"},
        ),
        created_at="2026-11-19T12:00:00+00:00",
    )
    md = render_markdown(report)
    html = render_html(report)
    assert "Precision" in md
    assert "Recall" in md
    assert "Confusion matrix" in md
    assert "Misclassifications by gold type" in md
    assert "id_card" in md
    assert "other" in md
    assert "Overall accuracy" in md
    assert "Confusion matrix" in html
    assert "Misclassifications by gold type" in html
    # No invented document text
    assert "SECRET" not in md
    assert report.classification is not None
    assert report.page_types  # mirrored from classification


def test_load_page_labels_and_pred_resolve(tmp_path: Path) -> None:
    gold_path = tmp_path / "pages.json"
    gold_path.write_text(
        json.dumps(
            {
                "case_id": "pack_x",
                "pages": [
                    {"index": 0, "page_type": "checklist"},
                    {"index": 1, "page_type": "blank"},
                ],
            }
        ),
        encoding="utf-8",
    )
    doc = load_page_labels_document(gold_path, case_id="pack_x")
    assert len(doc.pages) == 2

    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "pack_x.pages.json").write_text(
        json.dumps({"pages": [{"index": 0, "page_type": "checklist"}]}),
        encoding="utf-8",
    )
    assert resolve_page_prediction_path(pred_dir, "pack_x") is not None
    assert resolve_page_prediction_path(pred_dir, "missing") is None


def test_duplicate_page_index_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate page index"):
        PageLabelsDocument(
            pages=[
                PageLabel(index=0, page_type="a"),
                PageLabel(index=0, page_type="b"),
            ]
        )


def test_case_id_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "pages.json"
    path.write_text(
        json.dumps({"case_id": "other", "pages": [{"index": 0, "page_type": "blank"}]}),
        encoding="utf-8",
    )
    with pytest.raises(EvalError, match="case_id"):
        load_page_labels_document(path, case_id="expected_case")
