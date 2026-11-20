"""P6.2 agreement rules on synthetic readings."""

from __future__ import annotations

from torus.config import load_config
from torus.config.schemas import VotingRules
from torus.consensus import AgreementReason, check_agreement, vote
from torus.core.models import Reading, ValidationResult

RULES = VotingRules(critical_min_families=2, noncritical_min_readers=1)
FAMILIES = {"p1": "paddle", "p2": "paddle", "t1": "tess", "v1": "vlm"}
OK = [ValidationResult(validator_name="fmt", passed=True)]
BAD = [ValidationResult(validator_name="fmt", passed=False)]


def _r(name: str, text: str, score: float | None = 0.95) -> Reading:
    return Reading(
        region_id="r", reader_name=name, reader_version="1", text=text, raw_score=score
    )


def _check(readings: list[Reading], *, critical: bool, vr=OK):  # type: ignore[no-untyped-def]
    res = vote("r", readings, families=FAMILIES)
    return check_agreement(
        res, readings, critical=critical, validator_results=vr, rules=RULES
    )


def test_critical_needs_two_families() -> None:
    same_family = _check([_r("p1", "42"), _r("p2", "42")], critical=True)
    assert not same_family.met
    assert AgreementReason.TOO_FEW_FAMILIES in same_family.reasons
    assert _check([_r("p1", "42"), _r("t1", "42")], critical=True).met


def test_critical_blocked_by_dissent_even_with_two_families() -> None:
    d = _check([_r("p1", "42"), _r("t1", "42"), _r("v1", "43")], critical=True)
    assert not d.met
    assert d.reasons == [AgreementReason.READER_DISSENT]


def test_validator_failure_blocks_both_kinds() -> None:
    both = [_r("p1", "42"), _r("t1", "42")]
    assert (
        AgreementReason.VALIDATOR_FAILED in _check(both, critical=True, vr=BAD).reasons
    )
    assert (
        AgreementReason.VALIDATOR_FAILED in _check(both, critical=False, vr=BAD).reasons
    )


def test_noncritical_one_strong_reading_plus_validator() -> None:
    assert _check([_r("p1", "Ab")], critical=False).met


def test_noncritical_weak_or_unscored_reading_is_not_strong() -> None:
    assert (
        AgreementReason.NO_STRONG_READING
        in _check([_r("p1", "Ab", score=0.5)], critical=False).reasons
    )
    assert (
        AgreementReason.NO_STRONG_READING
        in _check([_r("p1", "Ab", score=None)], critical=False).reasons
    )


def test_noncritical_without_validator_is_never_vacuously_met() -> None:
    d = _check([_r("p1", "Ab")], critical=False, vr=[])
    assert d.reasons == [AgreementReason.NO_VALIDATOR]
    relaxed = VotingRules(
        critical_min_families=2,
        noncritical_min_readers=1,
        noncritical_require_validator=False,
    )
    res = vote("r", [_r("p1", "Ab")])
    assert check_agreement(
        res, [_r("p1", "Ab")], critical=False, validator_results=[], rules=relaxed
    ).met


def test_strong_reading_must_agree_with_voted_text() -> None:
    readings = [
        _r("p1", "Ab", score=0.5),
        _r("t1", "Ab", score=0.5),
        _r("v1", "Xy", 0.99),
    ]
    res = vote("r", readings, weights={"p1": 1.0, "t1": 1.0, "v1": 1.0})
    d = check_agreement(
        res, readings, critical=False, validator_results=OK, rules=RULES
    )
    assert res.candidate.text == "Ab"
    assert AgreementReason.NO_STRONG_READING in d.reasons


def test_no_readings() -> None:
    res = vote("r", [])
    d = check_agreement(res, [], critical=True, validator_results=OK, rules=RULES)
    assert d.reasons == [AgreementReason.NO_READINGS]


def test_shipped_config_voting_rules_load() -> None:
    cfg = load_config()
    assert cfg.pipeline.voting.critical_min_families == 2
    assert cfg.pipeline.voting.strong_reading_min_score == 0.9
