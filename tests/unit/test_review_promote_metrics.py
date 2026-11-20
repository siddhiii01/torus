"""P9.4 promote-to-test-set and P9.5 review metrics (synthetic data only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from torus.config.schemas import FieldSpec, TemplateConfig
from torus.core.errors import EvalError
from torus.core.models import BBox, Evidence, ExtractionResult, FieldValue
from torus.core.types import FieldStatus, FieldType
from torus.eval.report import build_run_report, render_html, render_markdown
from torus.eval.review_metrics import (
    DecisionEvent,
    PackReviewInput,
    aggregate_review_metrics,
)
from torus.eval.validate import validate_case_dir
from torus.review.promote import gold_rows, promote_reviewed
from torus.review.queue import ReviewError, build_queue
from torus.review.server import Decision
from torus.review.store import CorrectionStore

_EV = Evidence(region_id="r1", page_index=0, bbox=BBox(x0=0.1, y0=0.2, x1=0.4, y1=0.3))
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _fv(key: str, value: str | None, status: FieldStatus) -> FieldValue:
    return FieldValue(
        column_key=key,
        value=value,
        status=status,
        confidence=0.7,
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
                "acct": _fv("acct", "12345", FieldStatus.LIKELY),
                "city": _fv("city", "zeta", FieldStatus.CONFIRMED),
                "note": _fv("note", None, FieldStatus.NOT_FOUND),
            }
        ],
    )


def _template() -> TemplateConfig:
    return TemplateConfig(
        id="t1",
        fields=[
            FieldSpec(
                key=k,
                excel_header=k,
                excel_col=i,
                description=k,
                type=FieldType.TEXT,
            )
            for i, k in enumerate(["name", "acct", "city", "note"], start=1)
        ],
    )


class _Clock:
    def __init__(self) -> None:
        self.t = _T0

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


def _decide(
    store: CorrectionStore, item_id: str, action: str, value: str | None
) -> None:
    item = next(i for i in build_queue(_result()) if i.item_id == item_id)
    store.record(item, Decision(item_id, action, value))  # type: ignore[arg-type]


def test_gold_only_from_reviewed_values(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c", "r")
    _decide(store, "0:name", "correct", "beta")
    _decide(store, "0:acct", "skip", None)
    rows, n = gold_rows(_result(), store)
    assert rows == [{"name": "beta"}]  # skip, confirmed and not_found excluded
    assert n == 1


def test_include_unreviewed_confirmed_is_opt_in(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c", "r")
    _decide(store, "0:acct", "accept", "12345")
    rows, _ = gold_rows(_result(), store, include_unreviewed_confirmed=True)
    assert rows == [{"acct": "12345", "city": "zeta"}]


def test_stale_decision_blocks_promotion(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c", "r")
    _decide(store, "0:name", "correct", "beta")
    changed = _result().model_copy(
        update={"rows": [{"name": _fv("name", "other", FieldStatus.NEEDS_REVIEW)}]}
    )
    with pytest.raises(ReviewError):
        gold_rows(changed, store)


def test_promote_writes_valid_case(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c", "r")
    _decide(store, "0:name", "correct", "beta")
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    data = tmp_path / "data"
    written = promote_reviewed(
        _result(), pdf, store, _template(), set_id="s", case_id="c1", data_dir=data
    )
    expected = json.loads(written.expected_path.read_text(encoding="utf-8"))
    assert expected["rows"] == [{"name": "beta"}]
    meta = json.loads(written.meta_path.read_text(encoding="utf-8"))
    assert meta["notes"] == "promoted from review"
    validate_case_dir(
        written.path,
        set_id="s",
        template_keys=frozenset(f.key for f in _template().fields),
        require_source=True,
    )
    with pytest.raises(EvalError):  # refuses to overwrite without force
        promote_reviewed(
            _result(), pdf, store, _template(), set_id="s", case_id="c1", data_dir=data
        )


def test_promote_refuses_unreviewed_pack(tmp_path: Path) -> None:
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    with pytest.raises(EvalError):
        promote_reviewed(
            _result(),
            pdf,
            CorrectionStore(tmp_path / "c", "r"),
            _template(),
            set_id="s",
            case_id="c1",
            include_unreviewed_confirmed=True,
            data_dir=tmp_path / "d",
        )


def _ev(key: str, action: str, seconds: float) -> DecisionEvent:
    return DecisionEvent(key, action, _T0 + timedelta(seconds=seconds))


def test_review_metrics_counts_timing_and_columns() -> None:
    history = [
        _ev("name", "correct", 0),
        _ev("acct", "accept", 10),
        _ev("name", "correct", 40),  # re-decision still costs time
        _ev("acct", "skip", 1000),  # long pause: a break, not review time
    ]
    final = [_ev("name", "correct", 40), _ev("acct", "accept", 10)]
    m = aggregate_review_metrics(
        [PackReviewInput(_result(), final, history)], max_gap_s=120
    )
    assert (m.packs, m.fields_needing_review) == (1, 2)
    assert (m.accepted, m.corrected, m.skipped, m.undecided) == (1, 1, 0, 0)
    assert m.correction_rate == 0.5
    assert m.timed_fields == 2 and m.seconds_per_field == 20.0
    assert m.most_corrected[0].column_key == "name"
    assert m.most_corrected[0].correction_rate == 1.0


def test_review_metrics_empty_and_untimed() -> None:
    m = aggregate_review_metrics(
        [PackReviewInput(_result(), [], [_ev("name", "accept", 0)])], max_gap_s=60
    )
    assert m.undecided == 2
    assert m.seconds_per_field is None
    assert m.most_corrected == []


def test_report_renders_review_section_without_values() -> None:
    m = aggregate_review_metrics(
        [
            PackReviewInput(
                _result(),
                [_ev("name", "correct", 0)],
                [_ev("name", "correct", 0), _ev("acct", "accept", 30)],
            )
        ],
        max_gap_s=120,
    )
    from torus.eval.report import RunVersions, TimingSummary
    from torus.eval.score import aggregate_scores

    report = build_run_report(
        set_id="s",
        aggregate=aggregate_scores([]),
        case_ids=[],
        timings=TimingSummary(total_ms=0, case_count=0),
        versions=RunVersions(config_hash="h", template_id="t1"),
        review=m,
    )
    md = render_markdown(report)
    assert "## Review effort" in md and "Most-corrected columns" in md
    assert "`name`" in md and "alpha" not in md
    assert "Review effort" in render_html(report)
