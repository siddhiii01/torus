"""Promotion rule (P10.4): new weights replace old ones only on measured gain.

Inputs are two hidden-set evaluation reports (metrics only). Promote iff:

* both reports are on the hidden split, the same set and the same cases;
* the candidate has zero silent errors;
* the regression gate (``compare_runs``) passes;
* micro accuracy and automatic rate are each no lower, and at least one is
  strictly higher.

After promotion the calibrator must be re-fitted (P6.3). A ``recalibrate`` hook
runs inside the promotion and rolls it back if it fails; without a hook the
result says ``calibration_pending`` (the calibrator already refuses to run
against changed reader versions, so a missed re-fit fails loudly).
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from torus.config.schemas import RegressionGates
from torus.eval.diff import compare_runs
from torus.eval.report import EvalRunReport
from torus.learn.dataset import LearnError
from torus.learn.registry import ModelRegistry


class PromotionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    promoted: bool
    reader: str
    version: str
    previous_version: str | None
    reasons: tuple[str, ...]
    accuracy: tuple[float, float]  # (baseline, candidate)
    automatic_rate: tuple[float, float]
    silent_errors: tuple[int, int]
    calibration_pending: bool


def evaluate_promotion(
    baseline: EvalRunReport,
    candidate: EvalRunReport,
    *,
    gates: RegressionGates,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if baseline.split != "hidden" or candidate.split != "hidden":
        reasons.append("both reports must be on the hidden split")
    if baseline.set_id != candidate.set_id or sorted(baseline.case_ids) != sorted(
        candidate.case_ids
    ):
        reasons.append("reports must cover the same set and cases")
    if candidate.case_count == 0 or candidate.scores.fields_micro.compared == 0:
        reasons.append("candidate report has nothing scored")
    if candidate.scores.silent_errors != 0:
        reasons.append(
            f"silent errors must be 0 (got {candidate.scores.silent_errors})"
        )
    reasons.extend(compare_runs(baseline, candidate, gates=gates).failures)

    b_acc, c_acc = (
        baseline.scores.fields_micro.accuracy,
        candidate.scores.fields_micro.accuracy,
    )
    b_auto = baseline.scores.status_rates.automatic_rate
    c_auto = candidate.scores.status_rates.automatic_rate
    if c_acc < b_acc:
        reasons.append("micro accuracy is lower than baseline")
    if c_auto < b_auto:
        reasons.append("automatic rate is lower than baseline")
    if not (c_acc > b_acc or c_auto > b_auto):
        reasons.append("no improvement in accuracy or automatic rate")
    return not reasons, reasons


def promote(
    registry: ModelRegistry,
    reader: str,
    version: str,
    baseline: EvalRunReport,
    candidate: EvalRunReport,
    *,
    gates: RegressionGates,
    recalibrate: Callable[[], None] | None = None,
) -> PromotionDecision:
    """Apply the rule; on success move ``current`` and run the re-fit hook."""
    registry.get(reader, version)  # fail early on an unknown version
    ok, reasons = evaluate_promotion(baseline, candidate, gates=gates)
    before = registry.current(reader)
    pending = recalibrate is None
    if ok:
        registry.set_current(reader, version)
        if recalibrate is not None:
            try:
                recalibrate()
            except Exception as exc:  # roll back: never leave stale confidence live
                registry.restore_current(reader, before)
                raise LearnError("recalibration failed; promotion rolled back") from exc
    return PromotionDecision(
        promoted=ok,
        reader=reader,
        version=version,
        previous_version=before.version if before else None,
        reasons=tuple(reasons),
        accuracy=(
            baseline.scores.fields_micro.accuracy,
            candidate.scores.fields_micro.accuracy,
        ),
        automatic_rate=(
            baseline.scores.status_rates.automatic_rate,
            candidate.scores.status_rates.automatic_rate,
        ),
        silent_errors=(baseline.scores.silent_errors, candidate.scores.silent_errors),
        calibration_pending=ok and pending,
    )
