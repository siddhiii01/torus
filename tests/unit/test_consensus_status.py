"""P6.4 status assignment and threshold fitting (synthetic outcomes)."""

from __future__ import annotations

import numpy as np
import pytest

from torus.config import load_config
from torus.config.schemas import ConfidenceThresholds
from torus.consensus import (
    AgreementDecision,
    AgreementReason,
    StatusSample,
    assign_status,
    evaluate_thresholds,
    fit_thresholds,
    vote,
)
from torus.core.errors import ConsensusError
from torus.core.models import Reading
from torus.core.types import FieldStatus

TH = ConfidenceThresholds(confirmed_min=0.99, likely_min=0.85)
MET = AgreementDecision(met=True)


def _vote(text: str = "42"):  # type: ignore[no-untyped-def]
    r = Reading(region_id="r", reader_name="a", reader_version="1", text=text)
    return vote("r", [r])


def _assign(p: float, agreement: AgreementDecision = MET, **kw):  # type: ignore[no-untyped-def]
    return assign_status(
        located=kw.pop("located", True),
        vote_result=kw.pop("vr", _vote()),
        agreement=agreement,
        probability=p,
        thresholds=TH,
        **kw,
    )


def test_confirmed_needs_agreement_and_probability() -> None:
    assert _assign(0.995).status == FieldStatus.CONFIRMED
    assert _assign(0.995).reason is None
    unmet = AgreementDecision(met=False, reasons=[AgreementReason.TOO_FEW_FAMILIES])
    d = _assign(0.999, unmet)
    assert d.status == FieldStatus.LIKELY
    assert d.reasons == ["too_few_families"]


def test_met_but_below_confirmed_is_likely_then_review() -> None:
    d = _assign(0.9)
    assert d.status == FieldStatus.LIKELY and d.reasons == ["low_confidence"]
    low = _assign(0.5)
    assert low.status == FieldStatus.NEEDS_REVIEW
    assert "low_confidence" in low.reasons


def test_failed_validator_or_missing_evidence_caps_at_review() -> None:
    bad = AgreementDecision(met=False, reasons=[AgreementReason.VALIDATOR_FAILED])
    assert _assign(0.999, bad).status == FieldStatus.NEEDS_REVIEW
    d = _assign(0.999, has_evidence=False)
    assert d.status == FieldStatus.NEEDS_REVIEW and "no_evidence" in d.reasons


def test_not_found_cases() -> None:
    assert _assign(0.9, located=False).status == FieldStatus.NOT_FOUND
    assert _assign(0.9, vr=vote("r", [])).status == FieldStatus.NOT_FOUND


def test_fit_gives_zero_silent_errors_on_dev() -> None:
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(3000):
        p = float(rng.uniform(0.3, 1.0))
        samples.append(StatusSample(p, True, bool(rng.random() < p)))
    th = fit_thresholds(samples, likely_target_precision=0.9)
    rep = evaluate_thresholds(samples, th)
    assert rep.silent_errors == 0
    assert rep.confirmed > 0
    assert th.likely_min <= th.confirmed_min


def test_wrong_values_failing_agreement_do_not_raise_the_bar() -> None:
    samples = [
        StatusSample(
            0.99, False, False
        ),  # wrong, but agreement unmet → never confirmed
        StatusSample(0.9, True, True),
    ]
    th = fit_thresholds(samples, likely_target_precision=0.9)
    assert th.confirmed_min == 0.0
    assert evaluate_thresholds(samples, th).silent_errors == 0


def test_impossible_when_a_wrong_value_has_probability_one() -> None:
    with pytest.raises(ConsensusError):
        fit_thresholds([StatusSample(1.0, True, False)], likely_target_precision=0.9)
    with pytest.raises(ConsensusError):
        fit_thresholds([], likely_target_precision=0.9)


def test_blocked_samples_are_never_confirmed_or_likely() -> None:
    s = [StatusSample(0.999, True, False, blocked=True)]
    rep = evaluate_thresholds(s, TH)
    assert rep.needs_review == 1 and rep.silent_errors == 0


def test_shipped_thresholds_are_valid() -> None:
    cfg = load_config()
    assert cfg.pipeline.confidence.likely_min <= cfg.pipeline.confidence.confirmed_min
