"""Time-to-onboard metric for new templates (P12.4).

A tiny append-only log of milestones per template id, so the "days, not weeks"
goal is measured instead of guessed. Entries hold ids, a milestone name and a
timestamp only: never document content.

Usage::

    python -m torus.eval onboarding record <template_id> started
    python -m torus.eval onboarding record <template_id> done
    python -m torus.eval onboarding summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from torus.core.errors import EvalError

STARTED = "started"
DONE = "done"
MILESTONES = (STARTED, "specs_written", "packs_labelled", "evaluated", DONE)
DEFAULT_LOG = Path("reports/eval/onboarding.jsonl")
GOAL_DAYS = 5.0  # "days, not weeks": a working week


@dataclass(frozen=True)
class OnboardingTime:
    template_id: str
    started: datetime
    done: datetime | None

    @property
    def days(self) -> float | None:
        if self.done is None:
            return None
        return (self.done - self.started).total_seconds() / 86400.0


def record(
    template_id: str,
    milestone: str,
    *,
    log: Path = DEFAULT_LOG,
    at: datetime | None = None,
) -> None:
    """Append one milestone; ``at`` defaults to now (UTC)."""
    if milestone not in MILESTONES:
        raise EvalError(f"unknown milestone {milestone!r}; use one of {MILESTONES}")
    when = at or datetime.now(UTC)
    if when.tzinfo is None:
        raise EvalError("timestamp must be timezone-aware")
    log.parent.mkdir(parents=True, exist_ok=True)
    line = {"template_id": template_id, "milestone": milestone, "at": when.isoformat()}
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")


def load_times(log: Path = DEFAULT_LOG) -> list[OnboardingTime]:
    """First ``started`` and last ``done`` per template, in order of first start."""
    if not log.is_file():
        return []
    started: dict[str, datetime] = {}
    done: dict[str, datetime] = {}
    for raw in log.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            tid, milestone = str(row["template_id"]), str(row["milestone"])
            at = datetime.fromisoformat(row["at"])
        except (ValueError, KeyError, TypeError) as exc:
            raise EvalError(f"bad onboarding log line in {log.name}") from exc
        if milestone == STARTED:
            started.setdefault(tid, at)
        elif milestone == DONE:
            done[tid] = at
    return [OnboardingTime(t, s, done.get(t)) for t, s in started.items()]


def format_summary(times: list[OnboardingTime], goal_days: float = GOAL_DAYS) -> str:
    """Markdown table: days per template and whether it met the goal."""
    lines = [
        "| template | days | within goal |",
        "|---|---|---|",
    ]
    for t in times:
        if t.days is None:
            lines.append(f"| {t.template_id} | in progress | n/a |")
        else:
            ok = "yes" if t.days <= goal_days else "no"
            lines.append(f"| {t.template_id} | {t.days:.1f} | {ok} |")
    finished = [t.days for t in times if t.days is not None]
    if finished:
        lines.append("")
        lines.append(
            f"Onboarded: {len(finished)}; mean {sum(finished) / len(finished):.1f} "
            f"days; goal {goal_days:g} days."
        )
    return "\n".join(lines)
