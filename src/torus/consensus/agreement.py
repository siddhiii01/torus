"""Agreement rules (P6.2): may this voted value be confirmed?

Critical fields need agreement from distinct reader families. Non-critical
fields need one strong agreeing reading plus a validator pass. The rules come
from ``config/pipeline.yaml`` (``voting``). This only says whether the *agreement*
bar is met; status assignment (P6.4) combines it with calibrated confidence.
A failed validator always blocks, for both kinds of field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import VotingRules
from torus.consensus.vote import VoteResult
from torus.core.models import Reading, ValidationResult


class AgreementReason(StrEnum):
    """Why the agreement bar was not met."""

    NO_READINGS = "no_readings"
    TOO_FEW_READINGS = "too_few_readings"
    TOO_FEW_FAMILIES = "too_few_families"
    READER_DISSENT = "reader_dissent"
    NO_STRONG_READING = "no_strong_reading"
    NO_VALIDATOR = "no_validator"
    VALIDATOR_FAILED = "validator_failed"


class AgreementDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    met: bool
    reasons: list[AgreementReason] = Field(default_factory=list)


def reading_strength(reading: Reading) -> float | None:
    """Raw score, else mean per-character confidence, else unknown."""
    if reading.raw_score is not None:
        return reading.raw_score
    confs = reading.char_confidences
    if confs:
        return sum(confs) / len(confs)
    return None


def check_agreement(
    result: VoteResult,
    readings: Sequence[Reading],
    *,
    critical: bool,
    validator_results: Sequence[ValidationResult],
    rules: VotingRules,
) -> AgreementDecision:
    """Apply the configured agreement rule for a critical or non-critical field."""
    reasons: list[AgreementReason] = []
    if result.n_readings == 0:
        return AgreementDecision(met=False, reasons=[AgreementReason.NO_READINGS])

    if any(not v.passed for v in validator_results):
        reasons.append(AgreementReason.VALIDATOR_FAILED)

    if critical:
        if len(result.agreeing_families) < rules.critical_min_families:
            reasons.append(AgreementReason.TOO_FEW_FAMILIES)
        # An unresolved disagreement is never confirmable, even with enough
        # agreeing families: escalation (P6.5) or review decides.
        if result.dissenting_readers:
            reasons.append(AgreementReason.READER_DISSENT)
    else:
        if result.n_readings < rules.noncritical_min_readers:
            reasons.append(AgreementReason.TOO_FEW_READINGS)
        if not _has_strong_agreeing(result, readings, rules.strong_reading_min_score):
            reasons.append(AgreementReason.NO_STRONG_READING)
        if rules.noncritical_require_validator and not validator_results:
            reasons.append(AgreementReason.NO_VALIDATOR)

    return AgreementDecision(met=not reasons, reasons=reasons)


def _has_strong_agreeing(
    result: VoteResult, readings: Sequence[Reading], min_score: float
) -> bool:
    by_name: Mapping[str, Reading] = {r.reader_name: r for r in readings}
    for name in result.candidate.agreeing_readers:
        reading = by_name.get(name)
        if reading is None:
            continue
        strength = reading_strength(reading)
        if strength is not None and strength >= min_score:
            return True
    return False
