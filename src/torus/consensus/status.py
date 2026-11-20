"""Status assignment (P6.4) and threshold selection from the dev set.

``confirmed`` needs the agreement bar (P6.2) *and* calibrated P(correct) at or
above ``confidence.confirmed_min``. The cut-off is not chosen by hand: it is the
lowest value that leaves zero wrong values among confirmed dev outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import ConfidenceThresholds
from torus.consensus.agreement import AgreementDecision, AgreementReason
from torus.consensus.vote import VoteResult
from torus.core.errors import ConsensusError
from torus.core.types import FieldStatus

LOW_CONFIDENCE = "low_confidence"
NO_EVIDENCE = "no_evidence"
NOT_LOCATED = "not_located"
NO_READING = "no_reading"


class StatusDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: FieldStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    @property
    def reason(self) -> str | None:
        """Text for ``FieldValue.reason``; None when confirmed."""
        return ",".join(self.reasons) if self.reasons else None


def assign_status(
    *,
    located: bool,
    vote_result: VoteResult,
    agreement: AgreementDecision,
    probability: float,
    thresholds: ConfidenceThresholds,
    has_evidence: bool = True,
) -> StatusDecision:
    """Decide the status of one voted value.

    A failed validator or missing evidence caps the status at ``needs_review``:
    a value is never "corrected" or trusted past a failed rule.
    """
    if not located:
        return StatusDecision(
            status=FieldStatus.NOT_FOUND, confidence=0.0, reasons=[NOT_LOCATED]
        )
    if not vote_result.candidate.text:
        return StatusDecision(
            status=FieldStatus.NOT_FOUND, confidence=0.0, reasons=[NO_READING]
        )

    blockers = [r.value for r in agreement.reasons]
    if not has_evidence:
        blockers.append(NO_EVIDENCE)
    hard_block = (
        AgreementReason.VALIDATOR_FAILED in agreement.reasons or not has_evidence
    )

    if hard_block:
        return StatusDecision(
            status=FieldStatus.NEEDS_REVIEW, confidence=probability, reasons=blockers
        )
    if agreement.met and probability >= thresholds.confirmed_min:
        return StatusDecision(status=FieldStatus.CONFIRMED, confidence=probability)
    if probability >= thresholds.likely_min:
        reasons = blockers
        if agreement.met:
            reasons = [LOW_CONFIDENCE]
        return StatusDecision(
            status=FieldStatus.LIKELY, confidence=probability, reasons=reasons
        )
    return StatusDecision(
        status=FieldStatus.NEEDS_REVIEW,
        confidence=probability,
        reasons=[*blockers, LOW_CONFIDENCE],
    )


@dataclass(frozen=True)
class StatusSample:
    """One dev-set outcome used to pick thresholds."""

    probability: float
    agreement_met: bool
    correct: bool
    blocked: bool = False  # failed validator or no evidence


class ThresholdReport(BaseModel):
    """Dev-set outcome of a threshold choice. Counts only, no text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    confirmed: int
    likely: int
    needs_review: int
    silent_errors: int = Field(description="Wrong values marked confirmed.")
    likely_errors: int
    automatic_rate: float = Field(description="Confirmed share of all samples.")


def evaluate_thresholds(
    samples: Sequence[StatusSample], thresholds: ConfidenceThresholds
) -> ThresholdReport:
    confirmed = likely = review = silent = likely_wrong = 0
    for s in samples:
        if s.blocked:
            review += 1
        elif s.agreement_met and s.probability >= thresholds.confirmed_min:
            confirmed += 1
            silent += not s.correct
        elif s.probability >= thresholds.likely_min:
            likely += 1
            likely_wrong += not s.correct
        else:
            review += 1
    n = len(samples)
    return ThresholdReport(
        n=n,
        confirmed=confirmed,
        likely=likely,
        needs_review=review,
        silent_errors=silent,
        likely_errors=likely_wrong,
        automatic_rate=confirmed / n if n else 0.0,
    )


def fit_thresholds(
    samples: Sequence[StatusSample],
    *,
    likely_target_precision: float,
    margin: float = 0.0,
) -> ConfidenceThresholds:
    """Pick ``confirmed_min`` / ``likely_min`` from dev outcomes.

    ``confirmed_min`` is just above the highest probability of any *wrong* value
    that met the agreement bar (plus ``margin``), so dev has zero silent errors.
    ``likely_min`` is the lowest cut-off at which every value at or above it has
    precision ``>= likely_target_precision``, capped at ``confirmed_min``.
    Raises ``ConsensusError`` when no valid confirmed cut-off exists.
    """
    if not samples:
        raise ConsensusError("no samples to fit thresholds")
    eligible = [s for s in samples if s.agreement_met and not s.blocked]
    wrong_p = [s.probability for s in eligible if not s.correct]
    confirmed_min = 0.0
    if wrong_p:
        confirmed_min = math.nextafter(max(wrong_p), 2.0) + margin
    if confirmed_min > 1.0:
        raise ConsensusError(
            "a wrong value reaches probability 1.0; cannot reach zero silent "
            "errors, improve readers or features"
        )
    confirmed_min = max(confirmed_min, 0.0)

    usable = sorted(
        (s for s in samples if not s.blocked), key=lambda s: s.probability, reverse=True
    )
    likely_min = confirmed_min
    right = 0
    for i, s in enumerate(usable, start=1):
        right += s.correct
        # Only judge a cut-off after a whole group of equal probabilities.
        last_of_group = i == len(usable) or usable[i].probability != s.probability
        if last_of_group and right / i >= likely_target_precision:
            likely_min = min(likely_min, s.probability)
    return ConfidenceThresholds(
        confirmed_min=min(confirmed_min, 1.0), likely_min=min(likely_min, confirmed_min)
    )
