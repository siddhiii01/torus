"""P12.4: time-to-onboard log and summary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from torus.core.errors import EvalError
from torus.eval.__main__ import main
from torus.eval.onboarding import format_summary, load_times, record

T0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def test_days_between_start_and_done(tmp_path: Path) -> None:
    log = tmp_path / "o.jsonl"
    record("t1", "started", log=log, at=T0)
    record("t1", "evaluated", log=log, at=T0 + timedelta(days=2))
    record("t1", "done", log=log, at=T0 + timedelta(days=3, hours=12))
    record("t2", "started", log=log, at=T0)
    times = {t.template_id: t for t in load_times(log)}
    assert times["t1"].days == pytest.approx(3.5)
    assert times["t2"].days is None


def test_summary_flags_goal(tmp_path: Path) -> None:
    log = tmp_path / "o.jsonl"
    for tid, days in (("fast", 2), ("slow", 12)):
        record(tid, "started", log=log, at=T0)
        record(tid, "done", log=log, at=T0 + timedelta(days=days))
    text = format_summary(load_times(log))
    assert "| fast | 2.0 | yes |" in text and "| slow | 12.0 | no |" in text
    assert "mean 7.0" in text


def test_rejects_unknown_milestone_and_naive_time(tmp_path: Path) -> None:
    with pytest.raises(EvalError):
        record("t", "nope", log=tmp_path / "o.jsonl")
    with pytest.raises(EvalError):
        record("t", "started", log=tmp_path / "o.jsonl", at=datetime(2026, 1, 1))


def test_cli_record_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = str(tmp_path / "o.jsonl")
    assert main(["onboarding", "record", "t9", "started", "--log", log]) == 0
    assert main(["onboarding", "summary", "--log", log]) == 0
    assert "| t9 | in progress | n/a |" in capsys.readouterr().out
    assert main(["onboarding", "record", "--log", log]) == 2
