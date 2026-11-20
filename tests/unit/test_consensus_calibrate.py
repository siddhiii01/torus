"""P6.3 calibration on synthetic outcomes (no documents)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from torus.consensus import (
    FEATURE_NAMES,
    CalibrationSample,
    Calibrator,
    calibration_metrics,
    extract_features,
    fit_calibrator,
    reader_fingerprint,
    reliability_report,
    vote,
)
from torus.core.errors import ConsensusError
from torus.core.models import Reading, ValidationResult

FP = reader_fingerprint([("a", "1"), ("b", "1")])


def _synth(n: int, seed: int, field_type: str = "pin") -> list[CalibrationSample]:
    """Outcome depends on agreement + validator, with label noise."""
    rng = np.random.default_rng(seed)
    out: list[CalibrationSample] = []
    for _ in range(n):
        agree = float(rng.uniform(0.2, 1.0))
        fams = float(rng.integers(1, 4))
        failed = float(rng.random() < 0.2)
        p_true = 1 / (1 + np.exp(-(6 * (agree - 0.6) + 1.0 * (fams - 1) - 3 * failed)))
        feats = [agree, agree, fams, 3.0, agree, 1 - failed, failed, 0.9, 1.0]
        out.append(CalibrationSample(field_type, feats, bool(rng.random() < p_true)))
    return out


def test_fit_is_deterministic_and_recovers_signal() -> None:
    train = _synth(2000, seed=1)
    a = fit_calibrator(train, reader_fingerprint=FP)
    b = fit_calibrator(train, reader_fingerprint=FP)
    assert a == b
    hi = a.predict([1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 0.0, 0.9, 1.0])
    lo = a.predict([0.3, 0.3, 1.0, 3.0, 0.3, 0.0, 1.0, 0.9, 1.0])
    assert hi > 0.9 > 0.5 > lo


def test_well_calibrated_on_held_out_data() -> None:
    cal = fit_calibrator(_synth(4000, seed=2), reader_fingerprint=FP)
    report = calibration_metrics(cal, _synth(4000, seed=3))
    assert report.expected_calibration_error < 0.05
    assert "ECE=" in report.to_markdown()


def test_requires_both_outcomes_and_samples() -> None:
    allgood = [CalibrationSample("pin", [0.5] * len(FEATURE_NAMES), True)] * 5
    with pytest.raises(ConsensusError):
        fit_calibrator(allgood, reader_fingerprint=FP)
    with pytest.raises(ConsensusError):
        fit_calibrator([], reader_fingerprint=FP)
    with pytest.raises(ConsensusError):
        fit_calibrator([CalibrationSample("pin", [0.5], True)], reader_fingerprint=FP)


def test_stale_calibrator_refuses_to_predict() -> None:
    cal = fit_calibrator(_synth(300, seed=4), reader_fingerprint=FP)
    other = reader_fingerprint([("a", "2"), ("b", "1")])
    assert cal.is_stale(other)
    feats = [0.9] * len(FEATURE_NAMES)
    with pytest.raises(ConsensusError):
        cal.predict(feats, current_fingerprint=other)
    assert 0.0 <= cal.predict(feats, current_fingerprint=FP) <= 1.0


def test_per_field_type_model_only_when_enough_data(tmp_path: Path) -> None:
    samples = _synth(300, 5, "pin") + _synth(10, 6, "name")
    cal = fit_calibrator(samples, reader_fingerprint=FP, min_per_type=50)
    assert set(cal.by_field_type) == {"pin"}
    path = tmp_path / "cal.json"
    cal.save(path)
    assert Calibrator.load(path) == cal
    feats = [0.7] * len(FEATURE_NAMES)
    assert cal.predict(feats, "name") == cal.predict(feats, None)


def test_extract_features_order_and_values() -> None:
    def r(n: str, t: str) -> Reading:
        return Reading(
            region_id="r", reader_name=n, reader_version="1", text=t, raw_score=0.8
        )

    readings = [r("a", "42"), r("b", "42")]
    res = vote("r", readings, families={"a": "x", "b": "y"})
    ok = [ValidationResult(validator_name="v", passed=True)]
    f = dict(zip(FEATURE_NAMES, extract_features(res, readings, ok, 0.7), strict=True))
    assert f["n_agreeing_families"] == 2.0
    assert f["best_agreeing_strength"] == 0.8
    assert f["validator_passed"] == 1.0 and f["validator_failed"] == 0.0
    assert f["page_quality"] == 0.7 and f["page_quality_known"] == 1.0
    unknown = dict(
        zip(FEATURE_NAMES, extract_features(res, readings, [], None), strict=True)
    )
    assert unknown["page_quality_known"] == 0.0 and unknown["validator_passed"] == 0.0


def test_reliability_report_bins() -> None:
    rep = reliability_report([0.05, 0.05, 0.95, 0.95], [False, False, True, False])
    assert rep.n == 4
    assert rep.bins[0].observed_rate == 0.0
    assert rep.bins[9].observed_rate == 0.5
    assert rep.expected_calibration_error == pytest.approx(0.5 * 0.45 + 0.5 * 0.05)
    with pytest.raises(ConsensusError):
        reliability_report([], [])
