"""Crop and page-in-context PNGs for the review UI (P9.1).

Boxes are normalised (0–1, top-left origin), so nothing here knows a page size
or a layout. Input images are PIL images (what ``torus.ingest`` produces).
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from torus.core.models import BBox

_MAX_CROP_SIDE = 900
_MAX_PAGE_SIDE = 1600
_HIGHLIGHT = (255, 0, 0)
_OTHER = (255, 160, 0)


def _to_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _fit(image: Any, max_side: int) -> Any:
    from PIL import Image

    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _pixel_box(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(box.x0 * width)
    y0 = int(box.y0 * height)
    x1 = max(x0 + 1, round(box.x1 * width))
    y1 = max(y0 + 1, round(box.y1 * height))
    return x0, y0, min(x1, width), min(y1, height)


def crop_png(image: Any, box: BBox, *, pad: float = 0.01) -> bytes:
    """PNG of ``box`` with ``pad`` (fraction of page) of context on each side."""
    padded = BBox(
        x0=max(0.0, box.x0 - pad),
        y0=max(0.0, box.y0 - pad),
        x1=min(1.0, box.x1 + pad),
        y1=min(1.0, box.y1 + pad),
    )
    left, top, right, bottom = _pixel_box(padded, image.width, image.height)
    return _to_png(_fit(image.crop((left, top, right, bottom)), _MAX_CROP_SIDE))


def page_png(
    image: Any,
    highlight: BBox,
    *,
    others: Sequence[BBox] = (),
) -> bytes:
    """Whole page, downscaled, with ``highlight`` boxed in red.

    ``others`` (other evidence of the same field) are boxed in orange.
    """
    from PIL import ImageDraw

    scaled = _fit(image.convert("RGB"), _MAX_PAGE_SIDE)
    draw = ImageDraw.Draw(scaled)
    for box in others:
        draw.rectangle(
            _pixel_box(box, scaled.width, scaled.height), outline=_OTHER, width=2
        )
    draw.rectangle(
        _pixel_box(highlight, scaled.width, scaled.height),
        outline=_HIGHLIGHT,
        width=3,
    )
    return _to_png(scaled)
