"""Character-level alignment, voting, and calibrated confidence."""

from torus.consensus.agreement import (
    AgreementDecision,
    AgreementReason,
    check_agreement,
)
from torus.consensus.align import align_texts, edit_distance
from torus.consensus.calibrate import (
    FEATURE_NAMES,
    CalibrationSample,
    Calibrator,
    ReliabilityReport,
    calibration_metrics,
    extract_features,
    fit_calibrator,
    reader_fingerprint,
    reliability_report,
)
from torus.consensus.escalate import EscalationOutcome, escalate, grow_bbox
from torus.consensus.status import (
    StatusDecision,
    StatusSample,
    ThresholdReport,
    assign_status,
    evaluate_thresholds,
    fit_thresholds,
)
from torus.consensus.vote import VoteResult, vote
from torus.consensus.weights import (
    ReaderWeights,
    WeightSample,
    fit_reader_weights,
)

__all__ = [
    "EscalationOutcome",
    "escalate",
    "grow_bbox",
    "StatusDecision",
    "StatusSample",
    "ThresholdReport",
    "assign_status",
    "evaluate_thresholds",
    "fit_thresholds",
    "FEATURE_NAMES",
    "CalibrationSample",
    "Calibrator",
    "ReliabilityReport",
    "AgreementDecision",
    "AgreementReason",
    "ReaderWeights",
    "VoteResult",
    "WeightSample",
    "align_texts",
    "calibration_metrics",
    "check_agreement",
    "extract_features",
    "fit_calibrator",
    "reader_fingerprint",
    "reliability_report",
    "edit_distance",
    "fit_reader_weights",
    "vote",
]
