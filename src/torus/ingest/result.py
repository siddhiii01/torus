"""Ingest result: Document plus per-page embedded-text assessment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from torus.core.models import Document
from torus.ingest.embedded_text import PageEmbeddedText


class IngestResult(BaseModel):
    """Outcome of loading one file into pages.

    ``embedded_text`` carries usability flags and, when usable, words for
    downstream stages. Never log word text or dump this object into logs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    document: Document
    embedded_text: tuple[PageEmbeddedText, ...] = ()
    dpi: int = Field(ge=72, le=1200)

    def usable_embedded_pages(self) -> list[PageEmbeddedText]:
        return [p for p in self.embedded_text if p.usable]
