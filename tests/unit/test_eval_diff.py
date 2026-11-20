"""Unit tests for torus.eval diff regression gate (P1.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torus.config.schemas import RegressionGates
from torus.core.errors import EvalError
from torus.core.types import FieldStatus
from torus.eval.diff import DiffResult, compare_runs, format_diff_markdown
from torus.eval.metrics import (
    CaseScore,
    FieldMetrics,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
)
from torus.eval.report import (
    EvalRunReport,
    RunVersions,
    TimingSummary,
    build_run_report,
    load_run_report,
    write_run_report_json,
)


def _status_rates() -> StatusRates:
    counts = {s.value: 0 for s in FieldStatus}
    counts[FieldStatus.CONFIRMED.value] = 9
    counts[FieldStatus.NEEDS_REVIEW.value] = 1
    total = sum(counts.values())
    rates = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return StatusRates(
        total=total,
        counts=counts,
        rates=rates,
        automatic_rate=rates[FieldStatus.CONFIRMED.value],
        review_rate=rates[FieldStatus.NEEDS_REVIEW.value],
    )


def _field(
    key: str, *, accuracy: float, silent: int = 0, compared: int = 10
) -> FieldMetrics:
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


def _score(
    *,
    silent: int,
    fields: dict[str, FieldMetrics],
) -> CaseScore:
    compared = sum(f.compared for f in fields.values())
    correct = sum(f.correct for f in fields.values())
    acc = (correct / compared) if compared else 0.0
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
            tp=correct,
            fp=compared - correct,
            fn=0,
            compared=compared,
            correct=correct,
            accuracy=acc,
            precision=acc,
            recall=1.0,
            f1=0.95,
        ),
        per_field=fields,
        silent_errors=silent,
        status_rates=_status_rates(),
    )


def _report(
    *,
    silent: int,
    fields: dict[str, FieldMetrics],
    set_id: str = "synth_smoke",
    created_at: str = "2026-11-19T08:00:00+00:00",
) -> EvalRunReport:
    return build_run_report(
        set_id=set_id,
        split="dev",
        aggregate=_score(silent=silent, fields=fields),
        case_ids=["case_a", "case_b"],
        timings=TimingSummary(
            total_ms=100.0,
            by_stage={"score": 100.0},
            case_count=2,
            mean_case_ms=50.0,
        ),
        versions=RunVersions(
            config_hash="hash_baseline",
            template_id="starter_v1",
            model_versions={},
        ),
        created_at=created_at,
    )


_GATES = RegressionGates(noise_margin=0.02, silent_error_tolerance=0)


def test_diff_passes_when_equal() -> None:
    fields = {
        "person_name": _field("person_name", accuracy=0.90),
        "pin_code": _field("pin_code", accuracy=1.0),
    }
    baseline = _report(silent=0, fields=fields)
    candidate = _report(
        silent=0,
        fields=fields,
        created_at="2026-11-19T09:00:00+00:00",
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is True
    assert result.silent_error_delta == 0
    assert result.failures == ()
    assert result.column_deltas["person_name"].accuracy_delta == 0.0


def test_diff_fails_when_silent_errors_rise() -> None:
    fields = {"pin_code": _field("pin_code", accuracy=1.0)}
    baseline = _report(silent=0, fields=fields)
    candidate = _report(
        silent=1,
        fields={"pin_code": _field("pin_code", accuracy=1.0, silent=1)},
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is False
    assert result.silent_error_delta == 1
    assert any("silent" in f.lower() for f in result.failures)


def test_diff_allows_silent_error_within_tolerance() -> None:
    gates = RegressionGates(noise_margin=0.02, silent_error_tolerance=1)
    fields = {"pin_code": _field("pin_code", accuracy=1.0)}
    baseline = _report(silent=0, fields=fields)
    candidate = _report(silent=1, fields=fields)
    result = compare_runs(baseline, candidate, gates=gates)
    assert result.passed is True
    assert result.silent_error_delta == 1


def test_diff_fails_when_column_drops_beyond_noise_margin() -> None:
    baseline = _report(
        silent=0,
        fields={
            "person_name": _field("person_name", accuracy=0.95),
            "pin_code": _field("pin_code", accuracy=1.0),
        },
    )
    # Drop of 0.05 > noise_margin 0.02
    candidate = _report(
        silent=0,
        fields={
            "person_name": _field("person_name", accuracy=0.90),
            "pin_code": _field("pin_code", accuracy=1.0),
        },
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is False
    assert result.column_deltas["person_name"].regressed is True
    assert any("person_name" in f for f in result.failures)
    assert result.column_deltas["pin_code"].regressed is False


def test_diff_passes_when_drop_within_noise_margin() -> None:
    baseline = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.95)},
    )
    # Drop of 0.01 <= noise_margin 0.02
    candidate = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.94)},
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is True
    assert result.column_deltas["person_name"].accuracy_delta == pytest.approx(-0.01)


def test_diff_improvement_does_not_fail() -> None:
    baseline = _report(
        silent=2,
        fields={"person_name": _field("person_name", accuracy=0.80, silent=2)},
    )
    candidate = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.95)},
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is True
    assert result.silent_error_delta == -2
    assert result.column_deltas["person_name"].accuracy_delta == pytest.approx(0.15)


def test_diff_new_column_only_in_candidate_is_not_regression() -> None:
    baseline = _report(
        silent=0,
        fields={"pin_code": _field("pin_code", accuracy=1.0)},
    )
    candidate = _report(
        silent=0,
        fields={
            "pin_code": _field("pin_code", accuracy=1.0),
            "mobile_no": _field("mobile_no", accuracy=0.5),
        },
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is True
    assert "mobile_no" in result.column_deltas
    assert result.column_deltas["mobile_no"].baseline_accuracy is None
    assert result.column_deltas["mobile_no"].regressed is False


def test_diff_missing_column_in_candidate_is_regression() -> None:
    """A scored baseline column absent from the candidate counts as a drop to 0."""
    baseline = _report(
        silent=0,
        fields={
            "person_name": _field("person_name", accuracy=0.90),
            "pin_code": _field("pin_code", accuracy=1.0),
        },
    )
    candidate = _report(
        silent=0,
        fields={"pin_code": _field("pin_code", accuracy=1.0)},
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    assert result.passed is False
    assert result.column_deltas["person_name"].regressed is True
    assert result.column_deltas["person_name"].candidate_accuracy == 0.0


def test_format_diff_markdown_has_no_field_values() -> None:
    baseline = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.95)},
    )
    candidate = _report(
        silent=1,
        fields={"person_name": _field("person_name", accuracy=0.80, silent=1)},
    )
    result = compare_runs(baseline, candidate, gates=_GATES)
    md = format_diff_markdown(result)
    assert "## Summary" in md or "Silent errors" in md
    assert "person_name" in md
    # Distinctive value tokens must never appear (metrics-only).
    assert "SECRET_" not in md
    assert result.passed is False


def test_json_round_trip_for_diff(tmp_path: Path) -> None:
    report = _report(
        silent=0,
        fields={"pin_code": _field("pin_code", accuracy=1.0)},
    )
    path = tmp_path / "run.json"
    write_run_report_json(report, path)
    loaded = load_run_report(path)
    assert loaded.set_id == report.set_id
    assert loaded.scores.silent_errors == 0
    assert loaded.scores.per_field["pin_code"].accuracy == 1.0


def test_load_run_report_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(EvalError):
        load_run_report(path)


def test_load_run_report_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvalError):
        load_run_report(tmp_path / "missing.json")


def test_write_report_also_writes_json_sidecar(tmp_path: Path) -> None:
    from torus.eval.report import write_report

    report = _report(
        silent=0,
        fields={"pin_code": _field("pin_code", accuracy=1.0)},
    )
    paths = write_report(report, tmp_path / "latest.md")
    assert paths.json_path is not None
    assert paths.json_path.is_file()
    loaded = load_run_report(paths.json_path)
    assert loaded.scores.silent_errors == report.scores.silent_errors


def test_cli_diff_exit_codes(tmp_path: Path) -> None:
    from torus.eval.__main__ import main

    baseline = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.95)},
    )
    good = _report(
        silent=0,
        fields={"person_name": _field("person_name", accuracy=0.95)},
        created_at="2026-11-19T10:00:00+00:00",
    )
    bad = _report(
        silent=1,
        fields={"person_name": _field("person_name", accuracy=0.80, silent=1)},
        created_at="2026-11-19T11:00:00+00:00",
    )
    base_path = tmp_path / "base.json"
    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"
    write_run_report_json(baseline, base_path)
    write_run_report_json(good, good_path)
    write_run_report_json(bad, bad_path)

    assert main(["diff", str(base_path), str(good_path)]) == 0
    assert main(["diff", str(base_path), str(bad_path)]) == 1


def test_diff_result_model_is_frozen() -> None:
    from pydantic import ValidationError

    result = DiffResult(
        passed=True,
        noise_margin=0.02,
        silent_error_tolerance=0,
        baseline_silent_errors=0,
        candidate_silent_errors=0,
        silent_error_delta=0,
        baseline_set_id="a",
        candidate_set_id="a",
        baseline_created_at="t0",
        candidate_created_at="t1",
        column_deltas={},
        failures=(),
    )
    assert result.passed is True
    with pytest.raises(ValidationError):
        result.passed = False  # type: ignore[misc]


def test_load_accepts_eval_run_report_payload(tmp_path: Path) -> None:
    """JSON written via model_dump must load without extra wrappers."""
    report = _report(
        silent=0,
        fields={"pin_code": _field("pin_code", accuracy=1.0)},
    )
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_run_report(path)
    assert loaded.versions.template_id == "starter_v1"
