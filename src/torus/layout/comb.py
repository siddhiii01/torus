"""Comb / character-box detection, stroke-preserving inpaint, cell split (P4.3).

All geometry is derived from image structure (equal-pitch vertical rulings in a
wide band). No absolute page coordinates.
"""

from __future__ import annotations

import hashlib
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import LayoutCombConfig
from torus.core.models import BBox, Region
from torus.core.types import RegionKind
from torus.layout.geometry import bbox_iou, pixel_to_norm_bbox

PixelBox = tuple[int, int, int, int]  # x0, y0, x1, y1
GrayU8 = NDArray[np.uint8]
RgbU8 = NDArray[np.uint8]


class CombCell(BaseModel):
    """One character cell inside a detected comb."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    bbox: BBox
    region: Region


class CombProcessResult(BaseModel):
    """Detection + cell split + optional inpainted crop for one comb."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    region: Region
    cells: tuple[CombCell, ...]
    # RGB crop of the comb after ruling removal (page-local crop, not full page).
    inpainted_crop: Image.Image | None = None
    # Pixel box of the comb on the source page (for aligning the crop).
    pixel_bbox: tuple[int, int, int, int] | None = None


def detect_comb_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutCombConfig | None = None,
) -> list[Region]:
    """Locate comb / character-box rows as ``RegionKind.COMB`` regions."""
    cfg = config if config is not None else LayoutCombConfig()
    if not cfg.enabled:
        return []

    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    if height < 16 or width < 32:
        return []

    gray = cast(GrayU8, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray, cfg.ink_threshold)
    candidates = _find_comb_pixel_boxes(ink, cfg)
    regions: list[Region] = []
    for box in candidates:
        regions.append(_make_comb_region(box, width, height, page_index))
    return regions


def split_comb_cells(
    image: Image.Image,
    comb: Region,
    *,
    config: LayoutCombConfig | None = None,
) -> list[CombCell]:
    """Split a comb region into per-character cell regions."""
    cfg = config if config is not None else LayoutCombConfig()
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    box = _norm_to_pixel(comb.bbox, width, height)
    gray = cast(GrayU8, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray, cfg.ink_threshold)
    cell_boxes = _split_pixel_cells(ink, box, cfg)
    return [
        _make_cell(i, cell, width, height, comb.page_index, comb.id)
        for i, cell in enumerate(cell_boxes)
    ]


def inpaint_comb_lines(
    image: Image.Image,
    comb: Region,
    *,
    config: LayoutCombConfig | None = None,
) -> Image.Image:
    """Return a crop of ``comb`` with ruled lines removed, strokes preserved.

    Crossing handwriting is protected: line-mask pixels that continue as ink
    perpendicular to the ruling are kept out of the inpaint mask.
    """
    cfg = config if config is not None else LayoutCombConfig()
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _norm_to_pixel(comb.bbox, width, height)
    crop = rgb[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return Image.fromarray(crop)

    gray = cast(GrayU8, cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray, cfg.ink_threshold)
    thin_v, thin_h = _ruling_masks(ink, cfg)
    line_mask = cast(GrayU8, cv2.bitwise_or(thin_v, thin_h))
    protected = _protect_crossing_strokes(
        ink,
        line_mask,
        radius=cfg.stroke_protect_radius,
        vertical_mask=thin_v,
        horizontal_mask=thin_h,
    )
    if int(np.count_nonzero(protected)) == 0:
        return Image.fromarray(crop)

    cleaned = cast(
        RgbU8,
        cv2.inpaint(crop, protected, cfg.inpaint_radius, cv2.INPAINT_TELEA),
    )
    return Image.fromarray(cleaned)


def process_comb(
    image: Image.Image,
    comb: Region,
    *,
    config: LayoutCombConfig | None = None,
) -> CombProcessResult:
    """Detect cells and inpaint rulings for one comb region."""
    cfg = config if config is not None else LayoutCombConfig()
    cells = split_comb_cells(image, comb, config=cfg)
    inpainted = inpaint_comb_lines(image, comb, config=cfg)
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    pixel_bbox = _norm_to_pixel(comb.bbox, width, height)
    return CombProcessResult(
        region=comb,
        cells=tuple(cells),
        inpainted_crop=inpainted,
        pixel_bbox=pixel_bbox,
    )


def merge_combs_into_regions(
    regions: list[Region],
    combs: list[Region],
    *,
    prefer_over_table_iou: float = 0.25,
) -> list[Region]:
    """Append comb regions, dropping table cells that heavily overlap a comb."""
    if not combs:
        return list(regions)

    kept: list[Region] = []
    for region in regions:
        if region.kind != RegionKind.TABLE_CELL:
            kept.append(region)
            continue
        if any(
            bbox_iou(region.bbox, comb.bbox) >= prefer_over_table_iou for comb in combs
        ):
            continue
        kept.append(region)
    # Drop duplicate combs that heavily overlap each other.
    merged_combs: list[Region] = []
    for comb in combs:
        if any(bbox_iou(comb.bbox, other.bbox) >= 0.5 for other in merged_combs):
            continue
        merged_combs.append(comb)
    return kept + merged_combs


# --- internals -----------------------------------------------------------------


def _ink_mask(gray: GrayU8, threshold: int) -> GrayU8:
    # Adaptive-ish: keep pixels darker than threshold (forms are white paper).
    mask = ((gray < threshold).astype(np.uint8) * 255).astype(np.uint8)
    return mask


def _find_comb_pixel_boxes(ink: GrayU8, cfg: LayoutCombConfig) -> list[PixelBox]:
    height, width = ink.shape[:2]
    # Short vertical kernels catch comb dividers; long horizontals catch rules.
    v_len = max(8, int(0.04 * height))
    h_len = max(16, int(0.08 * width))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

    vertical = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, v_close)
    vertical = cast(GrayU8, cv2.morphologyEx(vertical, cv2.MORPH_OPEN, v_kernel))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, h_close)
    horizontal = cast(GrayU8, cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, h_kernel))

    if int(np.count_nonzero(vertical)) < 8:
        return []

    # Candidate bands: dilate V∪H and take connected components.
    corner = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid = cv2.bitwise_or(
        cv2.dilate(vertical, corner),
        cv2.dilate(horizontal, corner)
        if int(np.count_nonzero(horizontal)) > 0
        else vertical,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)

    boxes: list[PixelBox] = []
    min_area = max(80, int(0.0004 * width * height))
    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or w < 40 or h < 12:
            continue
        if w / max(h, 1) < cfg.min_aspect:
            continue

        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        roi_v = vertical[y0:y1, x0:x1]
        xs = _vertical_divider_xs(roi_v, cfg)
        if xs is None:
            continue
        # Map local xs back to page coords; outer box from first/last divider
        # with a small inset so we stay inside the rulings.
        page_xs = [x0 + v for v in xs]
        cell_count = len(page_xs) - 1
        if cell_count < cfg.min_cells or cell_count > cfg.max_cells:
            continue
        pitches = [page_xs[j + 1] - page_xs[j] for j in range(cell_count)]
        if not _equal_pitch(pitches, cfg.max_pitch_cv):
            continue
        boxes.append((page_xs[0], y0, page_xs[-1], y1))

    return _nms_boxes(boxes, iou=0.45)


def _vertical_divider_xs(
    vertical_roi: GrayU8,
    cfg: LayoutCombConfig,
) -> list[int] | None:
    """Return sorted x positions of vertical dividers inside a ROI, or None."""
    h, w = vertical_roi.shape[:2]
    if h < 8 or w < 24:
        return None
    cover_min = max(3, int(cfg.v_cover_min * h))
    col_sum = np.sum(vertical_roi > 0, axis=0)
    if int(col_sum.max()) < cover_min:
        return None
    thresh = max(cover_min, int(0.4 * float(col_sum.max())))
    raw = [int(i) for i, v in enumerate(col_sum) if int(v) >= thresh]
    if not raw:
        return None
    xs = _cluster_positions(raw, tol=max(2, int(0.03 * w)))
    if len(xs) < cfg.min_cells + 1:
        return None
    return xs


def _equal_pitch(pitches: list[int], max_cv: float) -> bool:
    if not pitches:
        return False
    arr = np.asarray(pitches, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 1.0:
        return False
    cv = float(arr.std() / mean)
    return cv <= max_cv


def _cluster_positions(positions: list[int], *, tol: int) -> list[int]:
    ordered = sorted(positions)
    groups: list[list[int]] = [[ordered[0]]]
    for p in ordered[1:]:
        if p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(sum(g) / len(g))) for g in groups]


def _split_pixel_cells(
    ink: GrayU8,
    box: PixelBox,
    cfg: LayoutCombConfig,
) -> list[PixelBox]:
    x0, y0, x1, y1 = box
    roi = ink[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    if h < 4 or w < 8:
        return [box]

    v_len = max(6, int(cfg.v_line_height_ratio * h))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    vertical = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, v_close)
    vertical = cast(GrayU8, cv2.morphologyEx(vertical, cv2.MORPH_OPEN, v_kernel))
    xs_local = _vertical_divider_xs(vertical, cfg)
    if xs_local is None or len(xs_local) < 2:
        # Fall back: equal split using estimated cell count from aspect.
        est = max(cfg.min_cells, min(cfg.max_cells, int(round(w / max(h * 0.7, 1)))))
        pitch = w / est
        return [
            (
                x0 + int(round(i * pitch)),
                y0,
                x0 + int(round((i + 1) * pitch)),
                y1,
            )
            for i in range(est)
        ]

    cells: list[PixelBox] = []
    for i in range(len(xs_local) - 1):
        cx0 = x0 + xs_local[i]
        cx1 = x0 + xs_local[i + 1]
        # Inset by 1px so the shared wall is not inside the cell crop.
        cells.append((min(cx0 + 1, cx1 - 1), y0, cx1, y1))
    return cells


def _ruling_masks(ink: GrayU8, cfg: LayoutCombConfig) -> tuple[GrayU8, GrayU8]:
    """Return ``(vertical_mask, horizontal_mask)`` of thin ruled lines."""
    h, w = ink.shape[:2]
    v_len = max(6, int(cfg.v_line_height_ratio * h))
    h_len = max(8, int(cfg.h_line_width_ratio * w))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vertical = cast(GrayU8, cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel))
    horizontal = cast(GrayU8, cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel))
    thin_v = cast(
        GrayU8,
        cv2.morphologyEx(
            vertical, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        ),
    )
    thin_h = cast(
        GrayU8,
        cv2.morphologyEx(
            horizontal,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        ),
    )
    return thin_v, thin_h


def _protect_crossing_strokes(
    ink: GrayU8,
    line_mask: GrayU8,
    *,
    radius: int,
    vertical_mask: GrayU8 | None = None,
    horizontal_mask: GrayU8 | None = None,
) -> GrayU8:
    """Clear line-mask pixels where ink continues *across* the ruling.

    Vertical rulings are cleared only when ink exists on both left and right.
    Horizontal rulings are cleared only when ink exists above and below.
    Applying both checks to a combined mask would erase continuous rulings.
    """
    h, w = ink.shape[:2]
    if h == 0 or w == 0:
        return line_mask
    ink_bin = ink > 0
    if vertical_mask is None and horizontal_mask is None:
        # Treat the whole mask as both orientations (caller supplies one mask).
        v_lines = line_mask > 0
        h_lines = line_mask > 0
    else:
        v_lines = (
            (vertical_mask > 0) if vertical_mask is not None else np.zeros_like(ink_bin)
        )
        h_lines = (
            (horizontal_mask > 0)
            if horizontal_mask is not None
            else np.zeros_like(ink_bin)
        )

    keep = (line_mask > 0).copy()

    for dx in range(1, radius + 2):
        if dx >= w:
            break
        left = np.zeros_like(ink_bin)
        right = np.zeros_like(ink_bin)
        left[:, dx:] = ink_bin[:, :-dx]
        right[:, :-dx] = ink_bin[:, dx:]
        # Only clear vertical-ruling pixels at true crossings.
        keep = keep & ~(v_lines & left & right)

    for dy in range(1, radius + 2):
        if dy >= h:
            break
        above = np.zeros_like(ink_bin)
        below = np.zeros_like(ink_bin)
        above[dy:, :] = ink_bin[:-dy, :]
        below[:-dy, :] = ink_bin[dy:, :]
        keep = keep & ~(h_lines & above & below)

    out = (keep.astype(np.uint8) * 255).astype(np.uint8)
    return out


def _make_comb_region(
    box: PixelBox,
    width: int,
    height: int,
    page_index: int,
) -> Region:
    bbox = pixel_to_norm_bbox(
        x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=width, height=height
    )
    key = f"{page_index}:comb:{bbox.x0:.4f}:{bbox.y0:.4f}:{bbox.x1:.4f}:{bbox.y1:.4f}"
    rid = f"comb-{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    return Region(id=rid, page_index=page_index, bbox=bbox, kind=RegionKind.COMB)


def _make_cell(
    index: int,
    box: PixelBox,
    width: int,
    height: int,
    page_index: int,
    comb_id: str,
) -> CombCell:
    bbox = pixel_to_norm_bbox(
        x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=width, height=height
    )
    region = Region(
        id=f"{comb_id}-cell-{index}",
        page_index=page_index,
        bbox=bbox,
        kind=RegionKind.FIELD_VALUE,
    )
    return CombCell(index=index, bbox=bbox, region=region)


def _norm_to_pixel(bbox: BBox, width: int, height: int) -> PixelBox:
    x0 = int(max(0, min(width - 1, round(bbox.x0 * width))))
    y0 = int(max(0, min(height - 1, round(bbox.y0 * height))))
    x1 = int(max(x0 + 1, min(width, round(bbox.x1 * width))))
    y1 = int(max(y0 + 1, min(height, round(bbox.y1 * height))))
    return (x0, y0, x1, y1)


def _nms_boxes(boxes: list[PixelBox], *, iou: float) -> list[PixelBox]:
    if not boxes:
        return []
    # Prefer wider boxes (full comb row over fragments).
    ordered = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[PixelBox] = []
    for box in ordered:
        nb = pixel_to_norm_bbox(
            x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=10_000, height=10_000
        )
        if any(
            bbox_iou(
                nb,
                pixel_to_norm_bbox(
                    x0=k[0], y0=k[1], x1=k[2], y1=k[3], width=10_000, height=10_000
                ),
            )
            >= iou
            for k in kept
        ):
            continue
        kept.append(box)
    return kept
