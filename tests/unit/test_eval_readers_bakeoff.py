"""Unit tests for the reader bake-off (P5.8). Synthetic data and fake readers only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.synth.testset import MINI_SET_ID, build_mini_testset
from torus.config import default_config_dir, load_config
from torus.core.models import ReadHint, Reading, Region
from torus.eval.__main__ import main as eval_main
from torus.eval.readers_bakeoff import (
    CellStats,
    Outcome,
    aggregate,
    hint_for_field,
    propose_defaults,
    run_bakeoff,
    score_outcome,
)

CONFIG = load_config(default_config_dir())
SPECS = {f.key: f for f in CONFIG.templates["starter_v1"].fields}


def _reading(text: str = "", *, error: str | None = None, ms: float = 5.0) -> Reading:
    return Reading(
        region_id="r",
        reader_name="x",
        reader_version="0",
        text=text,
        time_ms=ms,
        error=error,
    )


def test_score_outcome_uses_the_field_match_rule() -> None:
    spec = SPECS["mobile_no"]  # phone: digits-only comparison
    ok = score_outcome("a", spec, _reading("98765 43210"), "9876543210")
    bad = score_outcome("a", spec, _reading("9876543211"), "9876543210")
    assert ok.correct and not bad.correct
    assert ok.field_type == "phone_in"


def test_error_is_never_correct_and_empty_is_distinct() -> None:
    spec = SPECS["address_pincode"]
    failed = score_outcome("a", spec, _reading("560001", error="boom"), "560001")
    blank = score_outcome("a", spec, _reading("  "), "560001")
    assert (failed.correct, failed.error, failed.empty) == (False, True, False)
    assert (blank.correct, blank.error, blank.empty) == (False, False, True)


def test_hint_comes_from_the_field_spec() -> None:
    spec = SPECS["address_pincode"]
    hint = hint_for_field(spec)
    assert hint.charset == spec.charset and hint.expected_length == spec.length


def _o(reader: str, ftype: str, correct: bool, ms: float = 10.0) -> Outcome:
    return Outcome(reader, ftype, correct, False, False, ms)


def test_aggregate_groups_and_computes_rates_and_latency() -> None:
    items = [_o("a", "pin", True, 10), _o("a", "pin", False, 30), _o("b", "pin", True)]
    stats = {(s.reader, s.field_type): s for s in aggregate(items)}
    a = stats[("a", "pin")]
    assert (a.n, a.correct, a.accuracy) == (2, 1, 0.5)
    assert a.median_ms == 20 and a.p95_ms == 30
    assert stats[("b", "pin")].accuracy == 1.0


def _cell(
    reader: str, ftype: str, acc: float, n: int = 20, ms: float = 10
) -> CellStats:
    return CellStats(
        reader=reader,
        field_type=ftype,
        n=n,
        correct=round(acc * n),
        errors=0,
        empty=0,
        accuracy=acc,
        median_ms=ms,
        p95_ms=ms,
    )


def test_proposal_picks_best_plus_best_other_family() -> None:
    families = {"a": "f1", "b": "f1", "c": "f2"}
    stats = [_cell("a", "pin", 0.9), _cell("b", "pin", 0.8), _cell("c", "pin", 0.6)]
    proposals, enabled = propose_defaults(stats, families)
    assert proposals[0].primary == "a" and proposals[0].second_family == "c"
    assert enabled == ["a", "c"]


def test_proposal_breaks_ties_on_speed_and_flags_single_family() -> None:
    families = {"a": "f1", "b": "f1"}
    stats = [_cell("a", "pin", 0.9, ms=50), _cell("b", "pin", 0.9, ms=5)]
    proposals, _ = propose_defaults(stats, families)
    assert proposals[0].primary == "b"
    assert proposals[0].second_family is None and "second family" in proposals[0].note


def test_too_few_samples_gives_no_proposal() -> None:
    proposals, enabled = propose_defaults([_cell("a", "pin", 1.0, n=3)], {"a": "f"})
    assert proposals[0].primary is None and enabled == []


class _ScriptedRunner:
    """Answers each column from a table; wrong on person_name on purpose."""

    def __init__(self) -> None:
        self.hints: list[ReadHint] = []

    def read(
        self, reader_name: str, image: Any, region: Region, hint: ReadHint | None = None
    ) -> Reading:
        assert hint is not None
        self.hints.append(hint)
        column = region.id.split(":")[1]
        answers = {
            "person_name": "Someone Else",
            "mobile_no": "9876543210",
            "address_pincode": "560001",
        }
        return _reading(answers[column])


def test_run_bakeoff_end_to_end_writes_metrics_only_report(tmp_path: Path) -> None:
    data = build_mini_testset(tmp_path / "data")
    runner = _ScriptedRunner()
    report, (md_path, json_path) = run_bakeoff(
        set_id=MINI_SET_ID,
        report_path=tmp_path / "out" / "bakeoff.md",
        reader_names=["stub_echo"],
        data_dir=data,
        runner=runner,
        min_samples=1,
    )
    assert report.case_count == 1  # dev split only
    assert report.field_samples == 3
    acc = {s.field_type: s.accuracy for s in report.stats}
    assert acc["phone_in"] == 1.0 and acc["pin"] == 1.0 and acc["name"] == 0.0
    assert report.readers_run == ["stub_echo"]
    assert runner.hints and runner.hints[0] != ReadHint()  # spec-derived hints

    text = md_path.read_text(encoding="utf-8") + json_path.read_text(encoding="utf-8")
    for leaked in ("Synthetic Person", "9876543210", "560001", "Someone Else"):
        assert leaked not in text
    assert json.loads(json_path.read_text(encoding="utf-8"))["set_id"] == MINI_SET_ID


def test_unknown_reader_is_a_clear_error(tmp_path: Path) -> None:
    data = build_mini_testset(tmp_path / "data")
    from torus.core.errors import EvalError

    with pytest.raises(EvalError, match="unknown reader"):
        run_bakeoff(
            set_id=MINI_SET_ID,
            report_path=tmp_path / "r.md",
            reader_names=["nope"],
            data_dir=data,
            runner=_ScriptedRunner(),
        )


def test_cli_runs_with_real_registry_and_skips_unrunnable_readers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = build_mini_testset(tmp_path / "data")
    code = eval_main(
        [
            "readers",
            "--set",
            MINI_SET_ID,
            "--report",
            str(tmp_path / "b.md"),
            "--data-dir",
            str(data),
            "--reader",
            "stub_echo",
            "--reader",
            "olmocr",  # needs 16 GB VRAM / a server: must skip, not crash
        ]
    )
    assert code == 0
    assert "readers_skipped=1" in capsys.readouterr().out
    report = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert "olmocr" in report["readers_skipped"]
    assert report["readers_run"] == ["stub_echo"]
