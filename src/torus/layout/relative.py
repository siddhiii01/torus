"""Search for a value region relative to a detected label (no fixed coords)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from torus.config.schemas import LabelAnchorConfig
from torus.core.models import BBox, Region


class RelativePlacement(StrEnum):
    RIGHT = "right"
    SAME_ROW = "same_row"
    BELOW = "below"


def _height(bbox: BBox) -> float:
    return max(0.0, bbox.y1 - bbox.y0)


def _width(bbox: BBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0)


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    """Intersection height / min height."""
    iy0 = max(a.y0, b.y0)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, iy1 - iy0)
    denom = min(_height(a), _height(b))
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    ix0 = max(a.x0, b.x0)
    ix1 = min(a.x1, b.x1)
    inter = max(0.0, ix1 - ix0)
    denom = min(_width(a), _width(b))
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _centre_y(bbox: BBox) -> float:
    return (bbox.y0 + bbox.y1) / 2.0


def qualifies(
    label: BBox,
    candidate: BBox,
    placement: RelativePlacement,
    config: LabelAnchorConfig,
) -> bool:
    """Whether ``candidate`` sits in the named relation to ``label``."""
    if placement is RelativePlacement.RIGHT:
        gap = candidate.x0 - label.x1
        if gap < -1e-6 or gap > config.max_right_gap:
            return False
        return _y_overlap_ratio(label, candidate) >= config.row_y_overlap_min

    if placement is RelativePlacement.SAME_ROW:
        # Same row but possibly with a larger horizontal gap / slight overlap.
        if candidate.x0 < label.x0:
            return False
        gap = candidate.x0 - label.x1
        if gap > config.max_right_gap:
            return False
        # Centres close relative to label height (or absolute small band).
        band = max(_height(label), 0.02)
        if abs(_centre_y(label) - _centre_y(candidate)) > band:
            return False
        return _y_overlap_ratio(label, candidate) >= config.row_y_overlap_min * 0.5

    if placement is RelativePlacement.BELOW:
        gap = candidate.y0 - label.y1
        if gap < -1e-6 or gap > config.max_below_gap:
            return False
        return _x_overlap_ratio(label, candidate) >= config.below_x_overlap_min

    return False


def _closeness(label: BBox, candidate: BBox, placement: RelativePlacement) -> float:
    """Lower is better; used to pick among candidates for one placement."""
    if placement is RelativePlacement.BELOW:
        return max(0.0, candidate.y0 - label.y1) + 0.01 * abs(label.x0 - candidate.x0)
    return max(0.0, candidate.x0 - label.x1) + 0.01 * abs(
        _centre_y(label) - _centre_y(candidate)
    )


def find_value_near_label(
    label_bbox: BBox,
    candidates: Sequence[Region],
    *,
    config: LabelAnchorConfig | None = None,
    exclude_ids: frozenset[str] | None = None,
) -> tuple[Region, RelativePlacement] | None:
    """Pick the best value region relative to a label.

    Search order follows ``config.prefer``. Among ties for one placement,
    the geometrically closest candidate wins.
    """
    cfg = config if config is not None else LabelAnchorConfig()
    blocked = exclude_ids if exclude_ids is not None else frozenset()
    value_kinds = frozenset(cfg.value_kinds)

    usable = [
        r for r in candidates if r.id not in blocked and r.kind.value in value_kinds
    ]
    if not usable:
        return None

    for name in cfg.prefer:
        placement = RelativePlacement(name)
        hits = [r for r in usable if qualifies(label_bbox, r.bbox, placement, cfg)]
        if not hits:
            continue
        hits.sort(key=lambda r: _closeness(label_bbox, r.bbox, placement))
        return hits[0], placement
    return None
