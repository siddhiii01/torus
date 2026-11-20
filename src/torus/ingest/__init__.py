"""PDF/image ingest into Page objects (P2.1)."""

from torus.ingest.embedded_text import (
    EmbeddedWord,
    PageEmbeddedText,
    assess_words,
)
from torus.ingest.formats import SUPPORTED_SUFFIXES, SourceFormat, detect_format
from torus.ingest.loader import ingest
from torus.ingest.result import IngestResult

__all__ = [
    "SUPPORTED_SUFFIXES",
    "EmbeddedWord",
    "IngestResult",
    "PageEmbeddedText",
    "SourceFormat",
    "assess_words",
    "detect_format",
    "ingest",
]
