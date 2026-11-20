"""Unit tests for torus.eval report writer (P1.3) — no document text in output."""

from __future__ import annotations

from pathlib import Path

from torus.core.types import FieldStatus
from torus.eval.metrics import (
    CaseScore,
    FieldMetrics,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
)
from torus.eval.report import (
    EvalRunReport,
    PageTypeMetrics,
    RunVersions,
    TimingSummary,
    build_run_report,
    render_html,
    render_markdown,
    write_report,
)

# Distinctive synthetic tokens that must NEVER appear in rendered reports.
# These stand in for real field values / OCR text (data-safety litmus).
_FORBIDDEN = (
    "SECRET_PERSON_NAME_XYZ",
    "99 Fake Street Nowhere",
    "RAW_OCR_LINE_SHOULD_NOT_LEAK",
    "<script>alert(1)</script>",
)


def _status_rates() -> StatusRates:
    counts = {s.value: 0 for s in FieldStatus}
    counts[FieldStatus.CONFIRMED.value] = 8
    counts[FieldStatus.LIKELY.value] = 1
    counts[FieldStatus.NEEDS_REVIEW.value] = 1
    counts[FieldStatus.NOT_FOUND.value] = 0
    total = sum(counts.values())
    rates = {k: round(v / total, 4) for k, v in counts.items()}
    return StatusRates(
        total=total,
        counts=counts,
        rates=rates,
        automatic_rate=rates[FieldStatus.CONFIRMED.value],
        review_rate=rates[FieldStatus.NEEDS_REVIEW.value],
    )


def _field(key: str, *, silent: int = 0, accuracy: float = 1.0) -> FieldMetrics:
    compared = 10
    correct = int(round(accuracy * compared))
    return FieldMetrics(
        key=key,
        tp=correct,
        fp=compared - correct,
        fn=0,
        compared=compared,
        correct=correct,
        accuracy=accuracy,
        precision=1.0 if correct == compared else 0.9,
        recall=1.0,
        f1=0.95,
        silent_errors=silent,
        status_counts={
            FieldStatus.CONFIRMED.value: correct,
            FieldStatus.LIKELY.value: 0,
            FieldStatus.NEEDS_REVIEW.value: compared - correct,
            FieldStatus.NOT_FOUND.value: 0,
        },
    )


def _aggregate_score() -> CaseScore:
    per_field = {
        "person_name": _field("person_name", silent=1, accuracy=0.9),
        "pin_code": _field("pin_code", silent=0, accuracy=1.0),
    }
    return CaseScore(
        case_id="aggregate",
        template_id="starter_v1",
        rows=RowAlignmentMetrics(
            gold=2,
            pred=2,
            aligned=2,
            unmatched_gold=0,
            unmatched_pred=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
        ),
        fields_micro=MicroMetrics(
            tp=19,
            fp=1,
            fn=0,
            compared=20,
            correct=19,
            accuracy=0.95,
            precision=0.95,
            recall=1.0,
            f1=0.9744,
        ),
        per_field=per_field,
        silent_errors=1,
        status_rates=_status_rates(),
        mismatches=[{"field": "person_name", "match": False}],
    )


def _sample_report() -> EvalRunReport:
    return build_run_report(
        set_id="synth_smoke",
        split="dev",
        aggregate=_aggregate_score(),
        case_ids=["case_a", "case_b"],
        page_types=[
            PageTypeMetrics(
                page_type="application_form",
                support=4,
                correct=4,
                accuracy=1.0,
                tp=4,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
            ),
            PageTypeMetrics(
                page_type="id_card",
                support=2,
                correct=1,
                accuracy=0.5,
                tp=1,
                fp=0,
                fn=1,
                precision=1.0,
                recall=0.5,
                f1=0.6667,
            ),
        ],
        timings=TimingSummary(
            total_ms=1234.5,
            by_stage={"ingest": 100.0, "read": 900.0, "map": 234.5},
            case_count=2,
            mean_case_ms=617.25,
        ),
        versions=RunVersions(
            config_hash="abc123def4567890",
            template_id="starter_v1",
            model_versions={"reader_a": "1.0.0", "mapper": "0.3"},
        ),
        created_at="2026-11-19T07:00:00+00:00",
    )


def test_build_run_report_counts_cases() -> None:
    report = _sample_report()
    assert report.case_count == 2
    assert report.set_id == "synth_smoke"
    assert report.scores.silent_errors == 1
    assert report.versions.config_hash.startswith("abc123")


def test_markdown_has_required_sections() -> None:
    md = render_markdown(_sample_report())
    assert "# Torus evaluation report" in md
    assert "## Summary" in md
    assert "## Per column" in md
    assert "## Per page type" in md
    assert "Precision" in md
    assert "Recall" in md
    assert "## Timing" in md
    assert "## Versions" in md
    assert "person_name" in md
    assert "application_form" in md
    assert "silent" in md.lower()
    assert "confirmed" in md
    assert "needs_review" in md
    assert "abc123def456" in md
    assert "reader_a" in md


def test_html_has_tables_and_escapes() -> None:
    # Inject a page type that would break HTML if not escaped.
    report = _sample_report()
    nasty = list(report.page_types) + [
        PageTypeMetrics(
            page_type='form & "quote"<x>',
            support=1,
            correct=1,
            accuracy=1.0,
        )
    ]
    report = report.model_copy(update={"page_types": nasty})
    html = render_html(report)
    assert "<table>" in html
    assert "person_name" in html
    assert "&amp;" in html or "&quot;" in html or "&lt;" in html
    assert "<x>" not in html
    assert 'form & "quote"<x>' not in html


def test_report_never_contains_forbidden_field_values() -> None:
    """Score rows that hold forbidden strings, then render — values must not leak."""
    from torus.config.schemas import (
        FieldSpec,
        MatchRule,
        NormalizeKind,
        RowKeyConfig,
        RowKeyPart,
        TemplateConfig,
    )
    from torus.core.types import Charset, FieldType
    from torus.eval.score import score_rows

    template = TemplateConfig(
        id="forbidden_v1",
        version=1,
        description="Synthetic template for data-safety report litmus",
        row_key=RowKeyConfig(
            primary=[
                RowKeyPart(sources=["person_name"], normalize=NormalizeKind.PERSON),
            ],
            fuzzy_threshold=0.8,
        ),
        fields=[
            FieldSpec(
                key="person_name",
                excel_header="Name",
                description="Person name",
                type=FieldType.NAME,
                charset=Charset.LETTERS,
                match=MatchRule.NORMALISED,
                critical=True,
            ),
            FieldSpec(
                key="address_line1",
                excel_header="Line1",
                description="Address line",
                type=FieldType.ADDRESS,
                match=MatchRule.EXACT,
            ),
            FieldSpec(
                key="ocr_note",
                excel_header="Note",
                description="Free text",
                type=FieldType.TEXT,
                match=MatchRule.EXACT,
            ),
        ],
    )
    gold = [
        {
            "person_name": _FORBIDDEN[0],
            "address_line1": _FORBIDDEN[1],
            "ocr_note": _FORBIDDEN[2],
        }
    ]
    pred = [
        {
            "person_name": _FORBIDDEN[0],
            "address_line1": _FORBIDDEN[1],
            "ocr_note": _FORBIDDEN[3],
        }
    ]
    scored = score_rows(gold, pred, template, case_id="pack_forbidden")
    # Mismatch metadata may exist but must never carry cell text.
    dumped = scored.model_dump()
    blob = str(dumped)
    for token in _FORBIDDEN:
        assert token not in blob

    report = build_run_report(
        set_id="forbidden_fixture",
        split="dev",
        aggregate=scored,
        case_ids=["pack_forbidden"],
        page_types=[],
        timings=TimingSummary(
            total_ms=10.0,
            by_stage={"score": 10.0},
            case_count=1,
            mean_case_ms=10.0,
        ),
        versions=RunVersions(
            config_hash="hash_only",
            template_id="forbidden_v1",
            model_versions={},
        ),
        created_at="2026-11-19T07:00:00+00:00",
    )

    md = render_markdown(report)
    html_out = render_html(report)
    combined = md + "\n" + html_out

    for token in _FORBIDDEN:
        assert token not in combined, f"forbidden token leaked into report: {token!r}"

    assert "person_name" in md
    assert "silent" in md.lower()


def test_write_report_writes_md_and_html(tmp_path: Path) -> None:
    report = _sample_report()
    md_path = tmp_path / "eval" / "latest.md"
    paths = write_report(report, md_path)
    assert paths.markdown.is_file()
    assert paths.html.is_file()
    assert paths.markdown.read_text(encoding="utf-8").startswith("# Torus")
    assert "<html" in paths.html.read_text(encoding="utf-8").lower()

    for token in _FORBIDDEN:
        assert token not in paths.markdown.read_text(encoding="utf-8")
        assert token not in paths.html.read_text(encoding="utf-8")


def test_empty_page_types_section() -> None:
    report = _sample_report().model_copy(update={"page_types": []})
    md = render_markdown(report)
    assert "No page-type classification metrics" in md
    html = render_html(report)
    assert "No page-type classification metrics" in html


def test_timing_merge_helper() -> None:
    from torus.eval.report import merge_timings

    summary = merge_timings(
        [
            {"ingest": 10.0, "read": 20.0},
            {"ingest": 5.0, "read": 25.0, "map": 3.0},
        ]
    )
    assert summary.case_count == 2
    assert summary.by_stage["ingest"] == 15.0
    assert summary.by_stage["read"] == 45.0
    assert summary.by_stage["map"] == 3.0
    assert summary.total_ms == 63.0
    assert summary.mean_case_ms == 31.5
