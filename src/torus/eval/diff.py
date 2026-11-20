"""Compare two evaluation runs and apply the regression gate (P1.4).

Fails when silent errors rise beyond ``silent_error_tolerance`` or any column
accuracy drops by more than ``noise_margin`` (from ``config/pipeline.yaml``).
Reports and diffs never carry document text — metrics and column keys only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import RegressionGates
from torus.eval.report import EvalRunReport, load_run_report, write_run_report_json

# Re-export JSON I/O so callers can ``from torus.eval.diff import load_run_report``.
__all__ = [
    "ColumnDelta",
    "DiffResult",
    "compare_runs",
    "format_diff_markdown",
    "load_run_report",
    "write_run_report_json",
]


class ColumnDelta(BaseModel):
    """Per-column accuracy change between baseline and candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    baseline_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy_delta: float = Field(
        description="candidate − baseline (negative means a drop).",
    )
    baseline_silent_errors: int | None = Field(default=None, ge=0)
    candidate_silent_errors: int | None = Field(default=None, ge=0)
    regressed: bool = Field(
        description="True when accuracy drop exceeds the noise margin.",
    )


class DiffResult(BaseModel):
    """Outcome of comparing two eval runs against regression gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    noise_margin: float = Field(ge=0.0, le=1.0)
    silent_error_tolerance: int = Field(ge=0)
    baseline_silent_errors: int = Field(ge=0)
    candidate_silent_errors: int = Field(ge=0)
    silent_error_delta: int = Field(
        description="candidate − baseline (positive means more silent errors).",
    )
    baseline_set_id: str
    candidate_set_id: str
    baseline_created_at: str
    candidate_created_at: str
    column_deltas: dict[str, ColumnDelta] = Field(default_factory=dict)
    failures: tuple[str, ...] = ()


def compare_runs(
    baseline: EvalRunReport,
    candidate: EvalRunReport,
    *,
    gates: RegressionGates,
) -> DiffResult:
    """Compare *candidate* against *baseline* using *gates*.

    Gate rules (AGENTS.md regression gate / ``config/pipeline.yaml``):

    - Silent-error increase above ``silent_error_tolerance`` → fail.
    - Any column whose accuracy drops by more than ``noise_margin`` → fail.
    - Columns present only in the candidate are not regressions.
    - Columns present only in the baseline are treated as candidate accuracy 0.
    """
    base_silent = baseline.scores.silent_errors
    cand_silent = candidate.scores.silent_errors
    silent_delta = cand_silent - base_silent

    failures: list[str] = []
    if silent_delta > gates.silent_error_tolerance:
        failures.append(
            f"silent errors rose by {silent_delta} "
            f"(baseline={base_silent}, candidate={cand_silent}, "
            f"tolerance={gates.silent_error_tolerance})"
        )

    base_fields = baseline.scores.per_field
    cand_fields = candidate.scores.per_field
    keys = sorted(set(base_fields) | set(cand_fields))
    column_deltas: dict[str, ColumnDelta] = {}

    for key in keys:
        b_fm = base_fields.get(key)
        c_fm = cand_fields.get(key)
        b_acc = b_fm.accuracy if b_fm is not None else None
        c_acc = c_fm.accuracy if c_fm is not None else None

        if b_acc is None and c_acc is not None:
            # New column in candidate: not a regression.
            delta = 0.0
            regressed = False
            effective_c = c_acc
        elif b_acc is not None and c_acc is None:
            # Dropped column: treat as accuracy 0 for the gate.
            effective_c = 0.0
            delta = effective_c - b_acc
            drop = b_acc - effective_c
            regressed = drop > gates.noise_margin
        else:
            assert b_acc is not None and c_acc is not None
            effective_c = c_acc
            delta = c_acc - b_acc
            drop = b_acc - c_acc
            regressed = drop > gates.noise_margin

        if regressed:
            assert b_acc is not None
            drop_amt = b_acc - effective_c
            failures.append(
                f"column `{key}` accuracy dropped by {drop_amt:.4f} "
                f"(baseline={b_acc:.4f}, candidate={effective_c:.4f}, "
                f"noise_margin={gates.noise_margin})"
            )

        column_deltas[key] = ColumnDelta(
            key=key,
            baseline_accuracy=b_acc,
            candidate_accuracy=effective_c if b_acc is not None else c_acc,
            accuracy_delta=delta,
            baseline_silent_errors=(b_fm.silent_errors if b_fm is not None else None),
            candidate_silent_errors=(c_fm.silent_errors if c_fm is not None else None),
            regressed=regressed,
        )

    return DiffResult(
        passed=not failures,
        noise_margin=gates.noise_margin,
        silent_error_tolerance=gates.silent_error_tolerance,
        baseline_silent_errors=base_silent,
        candidate_silent_errors=cand_silent,
        silent_error_delta=silent_delta,
        baseline_set_id=baseline.set_id,
        candidate_set_id=candidate.set_id,
        baseline_created_at=baseline.created_at,
        candidate_created_at=candidate.created_at,
        column_deltas=column_deltas,
        failures=tuple(failures),
    )


def format_diff_markdown(result: DiffResult) -> str:
    """Render a metrics-only Markdown summary of a :class:`DiffResult`."""
    status = "PASS" if result.passed else "FAIL"
    lines: list[str] = [
        "# Torus evaluation diff",
        "",
        f"- **Result:** {status}",
        (
            f"- **Baseline set:** `{result.baseline_set_id}` "
            f"({result.baseline_created_at})"
        ),
        (
            f"- **Candidate set:** `{result.candidate_set_id}` "
            f"({result.candidate_created_at})"
        ),
        f"- **Noise margin:** {result.noise_margin}",
        f"- **Silent-error tolerance:** {result.silent_error_tolerance}",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Silent errors | {result.baseline_silent_errors} | "
            f"{result.candidate_silent_errors} | {result.silent_error_delta:+d} |"
        ),
        "",
        "## Per column",
        "",
        "| Column | Baseline acc | Candidate acc | Delta | Regressed |",
        "|---|---:|---:|---:|:---:|",
    ]
    for key in sorted(result.column_deltas):
        d = result.column_deltas[key]
        b = _fmt_acc(d.baseline_accuracy)
        c = _fmt_acc(d.candidate_accuracy)
        delta = f"{d.accuracy_delta:+.4f}"
        flag = "yes" if d.regressed else "no"
        lines.append(f"| `{key}` | {b} | {c} | {delta} | {flag} |")

    if result.failures:
        lines.extend(["", "## Failures", ""])
        for reason in result.failures:
            lines.append(f"- {reason}")
    else:
        lines.extend(["", "## Failures", "", "None."])

    lines.append("")
    return "\n".join(lines)


def _fmt_acc(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"
