"""Confidence calibration (P6.3): agreement features → P(value is correct).

An L2-regularised logistic regression fitted on dev-set outcomes. Nothing here
is a hand-made formula: the probability is whatever the dev set measured. The
fit records a fingerprint of the readers it was fitted with, so a reader change
makes the calibrator stale until it is re-fitted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from torus.consensus.agreement import reading_strength
from torus.consensus.vote import VoteResult
from torus.core.errors import ConsensusError
from torus.core.models import Reading, ValidationResult

FEATURE_NAMES: tuple[str, ...] = (
    "weighted_agreement",
    "min_column_agreement",
    "n_agreeing_families",
    "n_readings",
    "best_agreeing_strength",
    "validator_passed",
    "validator_failed",
    "page_quality",
    "page_quality_known",
)

Vector = NDArray[np.float64]


def extract_features(
    result: VoteResult,
    readings: Sequence[Reading],
    validator_results: Sequence[ValidationResult],
    page_quality: float | None,
) -> list[float]:
    """Agreement features for one voted region, in ``FEATURE_NAMES`` order."""
    by_name = {r.reader_name: r for r in readings}
    strengths = [
        s
        for name in result.candidate.agreeing_readers
        if (reading := by_name.get(name)) is not None
        and (s := reading_strength(reading)) is not None
    ]
    failed = any(not v.passed for v in validator_results)
    passed = bool(validator_results) and not failed
    return [
        result.candidate.confidence,
        result.min_column_agreement,
        float(len(result.agreeing_families)),
        float(result.n_readings),
        max(strengths, default=0.0),
        float(passed),
        float(failed),
        page_quality if page_quality is not None else 0.0,
        float(page_quality is not None),
    ]


def reader_fingerprint(readers: Iterable[tuple[str, str]]) -> str:
    """Stable hash of the (reader name, version) set a calibrator is fitted for."""
    joined = "|".join(f"{n}@{v}" for n, v in sorted(set(readers)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CalibrationSample:
    """One dev-set outcome: features of a voted value and whether it was right."""

    field_type: str
    features: Sequence[float]
    correct: bool


class LogisticModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    n_samples: int = Field(ge=1)
    n_correct: int = Field(ge=0)

    def predict(self, features: Sequence[float]) -> float:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        z = float(x @ np.asarray(self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0))))


class Calibrator(BaseModel):
    """Global model plus optional per-field-type models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_names: list[str]
    reader_fingerprint: str
    global_model: LogisticModel
    by_field_type: dict[str, LogisticModel] = Field(default_factory=dict)

    def is_stale(self, current_fingerprint: str) -> bool:
        return self.reader_fingerprint != current_fingerprint

    def predict(
        self,
        features: Sequence[float],
        field_type: str | None = None,
        *,
        current_fingerprint: str | None = None,
    ) -> float:
        """P(correct). Raises when readers changed since the fit (re-fit needed)."""
        if current_fingerprint is not None and self.is_stale(current_fingerprint):
            raise ConsensusError("calibrator is stale: readers changed, re-fit it")
        if len(features) != len(self.feature_names):
            raise ConsensusError("feature vector does not match the calibrator")
        model = self.by_field_type.get(field_type or "", self.global_model)
        return model.predict(features)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def fit_calibrator(
    samples: Sequence[CalibrationSample],
    *,
    reader_fingerprint: str,
    min_per_type: int = 50,
    l2: float = 1.0,
) -> Calibrator:
    """Fit the global model and one model per field type with enough outcomes.

    Raises ``ConsensusError`` unless the dev set has both right and wrong
    values: without wrong ones a probability cannot be measured.
    """
    if not samples:
        raise ConsensusError("no calibration samples")
    width = len(FEATURE_NAMES)
    if any(len(s.features) != width for s in samples):
        raise ConsensusError("sample feature vectors do not match FEATURE_NAMES")
    global_model = _fit(samples, l2)
    per_type: dict[str, LogisticModel] = {}
    for ftype in sorted({s.field_type for s in samples}):
        subset = [s for s in samples if s.field_type == ftype]
        outcomes = {s.correct for s in subset}
        if len(subset) >= min_per_type and len(outcomes) == 2:
            per_type[ftype] = _fit(subset, l2)
    return Calibrator(
        feature_names=list(FEATURE_NAMES),
        reader_fingerprint=reader_fingerprint,
        global_model=global_model,
        by_field_type=per_type,
    )


def _fit(samples: Sequence[CalibrationSample], l2: float) -> LogisticModel:
    x = np.asarray([list(s.features) for s in samples], dtype=np.float64)
    y = np.asarray([1.0 if s.correct else 0.0 for s in samples])
    if y.min() == y.max():
        raise ConsensusError(
            "calibration needs both correct and incorrect dev outcomes"
        )
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0  # constant feature: contributes nothing
    z = np.hstack([(x - mean) / scale, np.ones((len(y), 1))])
    penalty = np.eye(z.shape[1]) * l2
    penalty[-1, -1] = 0.0  # never shrink the intercept
    w = np.zeros(z.shape[1])
    prev = _objective(z, y, w, penalty)
    for _ in range(100):
        p = _sigmoid(z @ w)
        grad = z.T @ (p - y) + penalty @ w
        hess = (z.T * (p * (1 - p))) @ z + penalty + 1e-9 * np.eye(z.shape[1])
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-6:  # backtrack so the objective never increases
            cand = w - t * step
            cur = _objective(z, y, cand, penalty)
            if cur <= prev:
                break
            t /= 2
        else:
            break
        converged = abs(prev - cur) < 1e-10
        w, prev = cand, cur
        if converged:
            break
    return LogisticModel(
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=w[:-1].tolist(),
        intercept=float(w[-1]),
        n_samples=len(y),
        n_correct=int(y.sum()),
    )


def _sigmoid(z: Vector) -> Vector:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _objective(z: Vector, y: Vector, w: Vector, penalty: NDArray[np.float64]) -> float:
    s = np.clip(z @ w, -35.0, 35.0)
    nll = float(np.sum(np.logaddexp(0.0, s) - y * s))
    return nll + 0.5 * float(w @ penalty @ w)


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


class ReliabilityReport(BaseModel):
    """Calibration-plot data: predicted vs observed accuracy per bin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bins: list[ReliabilityBin]
    expected_calibration_error: float
    brier_score: float
    n: int

    def to_markdown(self, bar_width: int = 20) -> str:
        """Table plus text bars (no plotting dependency, no document text)."""
        lines = [
            f"n={self.n}  ECE={self.expected_calibration_error:.4f}  "
            f"Brier={self.brier_score:.4f}",
            "",
            "| bin | n | predicted | observed | plot (| observed, : ideal) |",
            "|---|---|---|---|---|",
        ]
        for b in self.bins:
            if b.count == 0 or b.observed_rate is None or b.mean_predicted is None:
                lines.append(f"| {b.lower:.1f}–{b.upper:.1f} | 0 | – | – | |")
                continue
            bar = ["·"] * bar_width
            for i in range(round(b.observed_rate * bar_width)):
                bar[min(i, bar_width - 1)] = "|"
            ideal = min(round(b.mean_predicted * bar_width), bar_width) - 1
            if ideal >= 0:
                bar[ideal] = ":" if bar[ideal] != "|" else "+"
            lines.append(
                f"| {b.lower:.1f}–{b.upper:.1f} | {b.count} | "
                f"{b.mean_predicted:.3f} | {b.observed_rate:.3f} | {''.join(bar)} |"
            )
        return "\n".join(lines)


def reliability_report(
    predicted: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> ReliabilityReport:
    """Bin predictions and compare them with observed accuracy."""
    if len(predicted) != len(correct) or not predicted:
        raise ConsensusError("predicted and correct must be equal-length and non-empty")
    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        lo, hi = b / bins, (b + 1) / bins
        if n == 0:
            out.append(
                ReliabilityBin(
                    lower=lo, upper=hi, count=0, mean_predicted=None, observed_rate=None
                )
            )
            continue
        mp, obs = float(p[mask].mean()), float(y[mask].mean())
        ece += n / len(p) * abs(mp - obs)
        out.append(
            ReliabilityBin(
                lower=lo, upper=hi, count=n, mean_predicted=mp, observed_rate=obs
            )
        )
    return ReliabilityReport(
        bins=out,
        expected_calibration_error=ece,
        brier_score=float(np.mean((p - y) ** 2)),
        n=len(p),
    )


def calibration_metrics(
    calibrator: Calibrator, samples: Iterable[CalibrationSample]
) -> ReliabilityReport:
    """Reliability of ``calibrator`` on a sample set (use the hidden set to judge)."""
    rows = list(samples)
    preds = [calibrator.predict(s.features, s.field_type) for s in rows]
    return reliability_report(preds, [s.correct for s in rows])


__all__ = [
    "FEATURE_NAMES",
    "CalibrationSample",
    "Calibrator",
    "ReliabilityBin",
    "ReliabilityReport",
    "calibration_metrics",
    "extract_features",
    "fit_calibrator",
    "reader_fingerprint",
    "reliability_report",
]
