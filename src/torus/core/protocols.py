"""Stage protocols. Implementations live elsewhere; core only defines contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torus.core.models import (
    Candidate,
    Document,
    FieldValue,
    Page,
    PageTypePrediction,
    ReaderCaps,
    ReadHint,
    Reading,
    Region,
    ValidationContext,
    ValidationResult,
)

# Pixel buffer passed to readers. Concrete type is typically ``PIL.Image.Image``;
# kept as a structural alias so core stays free of import-time imaging side effects.
type Image = object

# One output row: Excel column key → FieldValue.
type Row = dict[str, FieldValue]


@runtime_checkable
class TemplateSpec(Protocol):
    """Minimal template contract for the field mapper.

    Full YAML schemas live in ``torus.config`` (P0.4); they must expose at least
    an ``id`` so results can record which template was used.
    """

    @property
    def id(self) -> str: ...


@runtime_checkable
class Reader(Protocol):
    """Plug-in reader: locate is done elsewhere; this only reads a crop/region."""

    name: str

    def capabilities(self) -> ReaderCaps: ...

    def read(self, image: Image, region: Region, hint: ReadHint) -> Reading: ...


@runtime_checkable
class PageClassifier(Protocol):
    """Assigns a page type (and confidence) to one page."""

    def classify(self, page: Page) -> PageTypePrediction: ...


@runtime_checkable
class Validator(Protocol):
    """Checks one string value; never rewrites it."""

    name: str

    def check(self, value: str, ctx: ValidationContext) -> ValidationResult: ...


@runtime_checkable
class FieldMapper(Protocol):
    """Maps candidates + template field specs onto Excel column values."""

    def map(
        self,
        doc: Document,
        candidates: list[Candidate],
        template: TemplateSpec,
    ) -> list[Row]: ...
