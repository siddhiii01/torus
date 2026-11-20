"""Pydantic schemas for evaluation test-set case files (P1.1 / P3.5 / P4.8).

On-disk layout (under ``TORUS_DATA_DIR``)::

    testsets/<set>/<case>/
      source.pdf
      expected.json
      meta.json
      pages.json       # optional (P3.5)
      locations.json   # optional (P4.8)

See ``docs/TESTSET_FORMAT.md``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from torus.core.models import BBox
from torus.core.types import RegionKind


class Split(StrEnum):
    """Which evaluation split a case belongs to."""

    DEV = "dev"
    HIDDEN = "hidden"


class ScanType(StrEnum):
    """How the pack was captured (metadata tag only)."""

    DIGITAL = "digital"
    SCAN = "scan"
    PHOTO = "photo"
    FAX = "fax"
    MIXED = "mixed"


class HandwritingLevel(StrEnum):
    """Rough handwriting density on the pack (metadata tag only)."""

    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class CaseTags(BaseModel):
    """Optional tags used to slice reports; never document body text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int | None = Field(default=None, ge=1900, le=2100)
    scan_type: ScanType | None = None
    handwriting_level: HandwritingLevel | None = None


class CaseMeta(BaseModel):
    """Contents of ``meta.json`` for one labelled case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    split: Split
    template_id: str = Field(min_length=1)
    tags: CaseTags = Field(default_factory=CaseTags)
    page_count_expected: int | None = Field(default=None, ge=0)
    notes: str = Field(
        default="",
        description="Operator notes about quality/split — not extracted field values.",
    )

    @field_validator("case_id", "template_id")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return stripped


# Cell values in expected rows: JSON scalars only (no nested objects/lists).
ExpectedCell = str | int | float | bool | None


class ExpectedDocument(BaseModel):
    """Contents of ``expected.json``: gold column values for scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_id: str = Field(min_length=1)
    case_id: str | None = Field(
        default=None,
        description="Optional; when set must match meta.case_id and the folder name.",
    )
    version: int = Field(default=1, ge=1)
    rows: list[dict[str, ExpectedCell]] = Field(default_factory=list)

    @field_validator("template_id")
    @classmethod
    def _strip_template_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "template_id must be a non-empty string"
            raise ValueError(msg)
        return stripped

    @field_validator("case_id")
    @classmethod
    def _strip_case_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            msg = "case_id must be non-empty when provided"
            raise ValueError(msg)
        return stripped

    @field_validator("rows")
    @classmethod
    def _validate_rows(
        cls, rows: list[dict[str, Any]]
    ) -> list[dict[str, ExpectedCell]]:
        cleaned: list[dict[str, ExpectedCell]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                msg = f"rows[{index}] must be an object mapping column key → value"
                raise ValueError(msg)
            out: dict[str, ExpectedCell] = {}
            for key, value in row.items():
                if not isinstance(key, str) or not key.strip():
                    msg = f"rows[{index}] has an empty or non-string column key"
                    raise ValueError(msg)
                if key.startswith("_"):
                    # Reserved for harness-internal bookkeeping; not template columns.
                    continue
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    msg = (
                        f"rows[{index}][{key!r}] must be a JSON scalar "
                        f"(str|int|float|bool|null), got {type(value).__name__}"
                    )
                    raise ValueError(msg)
                out[key.strip()] = value
            cleaned.append(out)
        return cleaned

    @model_validator(mode="after")
    def _at_least_schema_ok(self) -> ExpectedDocument:
        # Empty rows is valid (e.g. blank pack); scoring treats it as zero gold rows.
        return self

    def column_keys(self) -> frozenset[str]:
        """Union of column keys present across all rows."""
        keys: set[str] = set()
        for row in self.rows:
            keys.update(row)
        return frozenset(keys)


class PageLabel(BaseModel):
    """Gold or predicted page type for one page index (no page content)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    page_type: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("page_type")
    @classmethod
    def _strip_page_type(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "page_type must be non-empty"
            raise ValueError(msg)
        return stripped


class PageLabelsDocument(BaseModel):
    """Contents of ``pages.json`` (gold) or a page-type prediction file.

    Labels are type ids only — never OCR text or crops.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str | None = Field(
        default=None,
        description="Optional; when set must match meta.case_id / folder name.",
    )
    version: int = Field(default=1, ge=1)
    pages: list[PageLabel] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def _strip_case_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            msg = "case_id must be non-empty when provided"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def _unique_indices(self) -> PageLabelsDocument:
        seen: set[int] = set()
        for page in self.pages:
            if page.index in seen:
                msg = f"duplicate page index {page.index}"
                raise ValueError(msg)
            seen.add(page.index)
        return self

    def by_index(self) -> dict[int, str]:
        """Map page index → page_type id."""
        return {page.index: page.page_type for page in self.pages}


class FieldLocation(BaseModel):
    """Gold or predicted value-region box for one template column (P4.8).

    Geometry only — never OCR text, crops, or field values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    bbox: BBox
    row_index: int = Field(
        default=0,
        ge=0,
        description="Index into expected.json rows when a pack has multiple rows.",
    )
    kind: RegionKind | None = Field(
        default=None,
        description="Optional region kind; matching does not require kind equality.",
    )

    @field_validator("column_key")
    @classmethod
    def _strip_column_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "column_key must be non-empty"
            raise ValueError(msg)
        return stripped


class LocationsDocument(BaseModel):
    """Contents of ``locations.json`` (gold) or a location prediction file.

    Used to score whether layout found the right region *before* reading.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str | None = Field(
        default=None,
        description="Optional; when set must match meta.case_id / folder name.",
    )
    version: int = Field(default=1, ge=1)
    fields: list[FieldLocation] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def _strip_case_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            msg = "case_id must be non-empty when provided"
            raise ValueError(msg)
        return stripped

    def column_keys(self) -> frozenset[str]:
        """Set of column keys present in this document."""
        return frozenset(item.column_key for item in self.fields)
