"""Raster image ingest: JPEG/PNG/TIFF (including multi-page), with EXIF fix."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from torus.core.errors import IngestError
from torus.core.models import Page
from torus.ingest.exif import apply_exif_orientation
from torus.ingest.formats import SourceFormat

if TYPE_CHECKING:
    from PIL import Image as PILImage


def load_image_pages(path: Path, *, source_format: SourceFormat) -> list[Page]:
    """Load one or more pages from a raster image file.

    EXIF orientation is applied so phone photos are upright. Multi-page TIFF
    frames become separate :class:`~torus.core.models.Page` objects in order.
    """
    from PIL import Image

    try:
        with Image.open(path) as opened:
            frames = _iter_frames(opened, source_format=source_format)
            pages: list[Page] = []
            for index, frame in enumerate(frames):
                rgb = frame.convert("RGB")
                upright, _transposed = apply_exif_orientation(rgb)
                width, height = upright.size
                page = Page(index=index, width=width, height=height)
                page.bind_image(upright.copy())
                pages.append(page)
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"cannot open image: {path.name}") from exc

    if not pages:
        raise IngestError(f"image has no frames: {path.name}")
    return pages


def _iter_frames(opened: PILImage.Image, *, source_format: SourceFormat) -> list[Any]:
    """Return a list of frame copies (multi-page TIFF → many frames)."""
    if source_format != SourceFormat.TIFF:
        return [opened.copy()]

    frames: list[Any] = []
    try:
        n_frames = int(getattr(opened, "n_frames", 1))
    except (TypeError, ValueError):
        n_frames = 1

    for index in range(max(1, n_frames)):
        try:
            opened.seek(index)
        except EOFError as exc:
            if index == 0:
                raise IngestError("TIFF has no readable frames") from exc
            break
        frames.append(opened.copy())
    return frames
