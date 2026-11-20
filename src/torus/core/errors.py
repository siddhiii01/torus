"""Typed exceptions for the Torus pipeline.

Stages raise these instead of bare ``Exception``. Reader failures that should
not stop the pipeline are recorded on ``Reading.error`` instead of raising.
"""

from __future__ import annotations


class TorusError(Exception):
    """Base class for all Torus errors."""


class ConfigError(TorusError):
    """Invalid, missing, or inconsistent configuration."""


class IngestError(TorusError):
    """Failure loading or rendering an input file into pages."""


class ImageNotLoadedError(TorusError):
    """Page image was requested but no image or loader is available."""


class LayoutError(TorusError):
    """Failure locating regions or anchors on a page."""


class ClassifyError(TorusError):
    """Hard page-classification failure (bad config, corrupt weights).

    Soft strategy failures return a low-confidence prediction instead.
    """


class ReaderError(TorusError):
    """Hard reader failure (unavailable model, bad contract).

    Soft failures are recorded on ``Reading.error`` instead.
    """


class ConsensusError(TorusError):
    """Failure during voting / candidate construction."""


class TorusValidationError(TorusError):
    """Value failed a validator or an evidence/status invariant.

    Named distinctly from ``pydantic.ValidationError``.
    """


class EvidenceRequiredError(TorusValidationError):
    """A FieldValue carries a non-empty value but has no evidence regions."""


class MappingError(TorusError):
    """Failure mapping candidates onto template columns."""


class OutputError(TorusError):
    """Failure writing Excel or other pipeline outputs."""


class EvalError(TorusError):
    """Invalid test-set layout, case files, or evaluation inputs."""
