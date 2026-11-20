"""Pixel ↔ normalised bbox helpers for synthetic fixtures."""

from __future__ import annotations

from typing import Literal

from torus.core import BBox

OrientationDegrees = Literal[0, 90, 180, 270]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def pixel_bbox_to_norm(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> BBox:
    """Convert pixel coordinates to a normalised ``BBox`` (top-left origin)."""
    if width <= 0 or height <= 0:
        msg = f"width and height must be > 0, got {width}x{height}"
        raise ValueError(msg)
    nx0 = clamp01(x0 / width)
    ny0 = clamp01(y0 / height)
    nx1 = clamp01(x1 / width)
    ny1 = clamp01(y1 / height)
    # Guarantee a non-empty box after clamping (at least one pixel).
    if nx1 <= nx0:
        nx1 = clamp01(nx0 + 1.0 / width)
    if ny1 <= ny0:
        ny1 = clamp01(ny0 + 1.0 / height)
    if nx1 <= nx0 or ny1 <= ny0:
        msg = "bbox collapsed after normalisation"
        raise ValueError(msg)
    return BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)


def rotate_norm_bbox(
    bbox: BBox,
    *,
    degrees: OrientationDegrees,
    src_width: int,
    src_height: int,
) -> tuple[BBox, int, int]:
    """Map a normalised bbox through a clockwise page rotation.

    Returns ``(new_bbox, new_width, new_height)`` in the rotated page space.
    """
    if degrees == 0:
        return bbox, src_width, src_height

    corners = (
        (bbox.x0 * src_width, bbox.y0 * src_height),
        (bbox.x1 * src_width, bbox.y0 * src_height),
        (bbox.x0 * src_width, bbox.y1 * src_height),
        (bbox.x1 * src_width, bbox.y1 * src_height),
    )
    if degrees == 90:
        mapped = [(src_height - y, x) for x, y in corners]
        new_w, new_h = src_height, src_width
    elif degrees == 180:
        mapped = [(src_width - x, src_height - y) for x, y in corners]
        new_w, new_h = src_width, src_height
    elif degrees == 270:
        mapped = [(y, src_width - x) for x, y in corners]
        new_w, new_h = src_height, src_width
    else:
        msg = f"unsupported rotation {degrees}"
        raise ValueError(msg)

    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return (
        pixel_bbox_to_norm(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
            width=new_w,
            height=new_h,
        ),
        new_w,
        new_h,
    )
