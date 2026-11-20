"""Pydantic data models passed between pipeline stages.

These are the frozen interchange types. Stages depend on them (and the
protocols), never on each other's internals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from torus.core.errors import EvidenceRequiredError, ImageNotLoadedError
from torus.core.types import (
    Charset,
    FieldStatus,
    OrientationDegrees,
    RegionKind,
)


class BBox(BaseModel):
    """Axis-aligned box in normalised page coordinates (0–1, top-left origin)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered_and_in_unit_square(self) -> BBox:
        for name, value in (
            ("x0", self.x0),
            ("y0", self.y0),
            ("x1", self.x1),
            ("y1", self.y1),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)
        if self.x0 >= self.x1:
            msg = f"x0 must be < x1, got {self.x0} >= {self.x1}"
            raise ValueError(msg)
        if self.y0 >= self.y1:
            msg = f"y0 must be < y1, got {self.y0} >= {self.y1}"
            raise ValueError(msg)
        return self


class LabelAnchor(BaseModel):
    """Optional label that anchors a value region (found at run time)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None = None
    bbox: BBox | None = None


class Page(BaseModel):
    """One rendered page. Image pixels stay lazy until accessed."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    orientation: OrientationDegrees = 0
    page_type: str | None = None
    blank: bool = False
    low_quality: bool = False
    rotated: bool = False
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    _image: Any | None = PrivateAttr(default=None)
    _image_loader: Callable[[], Any] | None = PrivateAttr(default=None)

    def bind_image(self, image: Any) -> None:
        """Attach an already-loaded image (tests, in-memory pipelines)."""
        self._image = image
        self._image_loader = None

    def bind_image_loader(self, loader: Callable[[], Any]) -> None:
        """Attach a lazy loader; pixels are fetched on first ``image`` access."""
        self._image_loader = loader
        self._image = None

    @property
    def image(self) -> Any:
        if self._image is not None:
            return self._image
        if self._image_loader is not None:
            self._image = self._image_loader()
            return self._image
        raise ImageNotLoadedError(
            f"page {self.index} has no image or image loader bound"
        )

    @property
    def has_image(self) -> bool:
        return self._image is not None or self._image_loader is not None


class Document(BaseModel):
    """A source file split into pages."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(min_length=1, description="Content hash of the source file.")
    source_path: str
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class Region(BaseModel):
    """A located region on a page, ready to be read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    bbox: BBox
    kind: RegionKind
    label_anchor: LabelAnchor | None = None


class Reading(BaseModel):
    """One reader's output for one region.

    Soft failures set ``error`` and leave ``text`` empty; the pipeline continues.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    region_id: str = Field(min_length=1)
    reader_name: str = Field(min_length=1)
    reader_version: str = Field(min_length=1)
    text: str = ""
    char_confidences: list[float] | None = None
    raw_score: float | None = Field(default=None, ge=0.0, le=1.0)
    time_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None

    @field_validator("char_confidences")
    @classmethod
    def _char_confidences_in_unit_interval(
        cls, value: list[float] | None
    ) -> list[float] | None:
        if value is None:
            return value
        for score in value:
            if not 0.0 <= score <= 1.0:
                msg = f"char confidence must be in [0, 1], got {score}"
                raise ValueError(msg)
        return value


class Candidate(BaseModel):
    """Voted value for a region after consensus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region_id: str = Field(min_length=1)
    text: str
    agreeing_readers: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Evidence(BaseModel):
    """Source region for a mapped field value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    bbox: BBox


class ValidationResult(BaseModel):
    """Outcome of one validator on one value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validator_name: str = Field(min_length=1)
    passed: bool
    message: str | None = None


class ValidationContext(BaseModel):
    """Context available to validators (other fields, lookups, page types)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str = Field(min_length=1)
    row_values: Mapping[str, str | None] = Field(default_factory=dict)
    document_id: str | None = None
    page_types: tuple[str, ...] = ()
    extras: Mapping[str, str] = Field(default_factory=dict)


class FieldValue(BaseModel):
    """One Excel column value for one output row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str = Field(min_length=1)
    value: str | None = None
    status: FieldStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    validator_results: list[ValidationResult] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def _value_requires_evidence(self) -> FieldValue:
        if self.value is not None and self.value != "" and not self.evidence:
            raise EvidenceRequiredError(
                f"field {self.column_key!r} has a value but no evidence regions"
            )
        return self


class ReaderCaps(BaseModel):
    """What a reader can handle; used for routing and doctor checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scripts: list[str] = Field(default_factory=list)
    handwriting: bool = False
    printed: bool = True
    layout: bool = False
    per_char_confidence: bool = False


class ReadHint(BaseModel):
    """Decoding constraints taken from the field spec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    charset: Charset = Charset.ANY
    script: str | None = None
    language: str | None = None
    expected_length: int | None = Field(default=None, ge=0)
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)


class PageTypePrediction(BaseModel):
    """Classifier output for one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Final pipeline output for one document against one template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    rows: list[dict[str, FieldValue]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    config_hash: str = Field(min_length=1)
