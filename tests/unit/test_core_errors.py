"""Unit tests for torus.core typed errors (P0.3)."""

from __future__ import annotations

import pytest

from torus.core import (
    ClassifyError,
    ConfigError,
    ConsensusError,
    EvalError,
    EvidenceRequiredError,
    ImageNotLoadedError,
    IngestError,
    LayoutError,
    MappingError,
    OutputError,
    ReaderError,
    TorusError,
    TorusValidationError,
)


@pytest.mark.parametrize(
    "exc_type",
    [
        ConfigError,
        IngestError,
        ImageNotLoadedError,
        LayoutError,
        ClassifyError,
        ReaderError,
        ConsensusError,
        TorusValidationError,
        EvidenceRequiredError,
        MappingError,
        OutputError,
        EvalError,
    ],
)
def test_errors_are_torus_errors(exc_type: type[TorusError]) -> None:
    err = exc_type("msg")
    assert isinstance(err, TorusError)
    assert str(err) == "msg"


def test_evidence_required_is_validation_error() -> None:
    assert issubclass(EvidenceRequiredError, TorusValidationError)
