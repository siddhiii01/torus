"""Geometry helpers for layout regions (normalised and pixel space)."""

from __future__ import annotations

from torus.core.models import BBox


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def pixel_to_norm_bbox(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> BBox:
    """Convert a pixel box to a normalised ``BBox`` (top-left origin)."""
    if width <= 0 or height <= 0:
        msg = f"width and height must be > 0, got {width}x{height}"
        raise ValueError(msg)
    nx0 = clamp01(x0 / width)
    ny0 = clamp01(y0 / height)
    nx1 = clamp01(x1 / width)
    ny1 = clamp01(y1 / height)
    if nx1 <= nx0:
        nx1 = clamp01(nx0 + 1.0 / width)
    if ny1 <= ny0:
        ny1 = clamp01(ny0 + 1.0 / height)
    if nx1 <= nx0 or ny1 <= ny0:
        msg = "bbox collapsed after normalisation"
        raise ValueError(msg)
    return BBox(x0=nx0, y0=ny0, x1=nx1, y1=ny1)


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two normalised boxes."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_intersection_over_gt(pred: BBox, gt: BBox) -> float:
    """Intersection over ground-truth area (location recall-friendly)."""
    ix0 = max(pred.x0, gt.x0)
    iy0 = max(pred.y0, gt.y0)
    ix1 = min(pred.x1, gt.x1)
    iy1 = min(pred.y1, gt.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    gt_area = bbox_area(gt)
    if gt_area <= 0.0:
        return 0.0
    return inter / gt_area
