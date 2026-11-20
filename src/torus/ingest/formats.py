"""Supported ingest formats and suffix routing."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from torus.core.errors import IngestError


class SourceFormat(StrEnum):
    """Normalised source format labels stored on Document.metadata."""

    PDF = "pdf"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"


_SUFFIX_MAP: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".jpg": SourceFormat.JPEG,
    ".jpeg": SourceFormat.JPEG,
    ".png": SourceFormat.PNG,
    ".tif": SourceFormat.TIFF,
    ".tiff": SourceFormat.TIFF,
}

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(_SUFFIX_MAP)


def detect_format(path: Path) -> SourceFormat:
    """Map a file suffix to a :class:`SourceFormat`, or raise :class:`IngestError`."""
    suffix = path.suffix.lower()
    fmt = _SUFFIX_MAP.get(suffix)
    if fmt is None:
        known = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise IngestError(
            f"unsupported file type {suffix!r} for {path.name!r}; supported: {known}"
        )
    return fmt
