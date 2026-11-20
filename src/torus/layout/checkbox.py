"""Checkbox / radio detection, fill-state, adjacent label mapping (P4.4).

All geometry is derived from image structure (near-square outlines, circular
outlines, relative size). No absolute page coordinates.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import LayoutCheckboxConfig
from torus.core.models import BBox, LabelAnchor, Region
from torus.core.types import RegionKind
from torus.layout.geometry import bbox_iou, pixel_to_norm_bbox

PixelBox = tuple[int, int, int, int]  # x0, y0, x1, y1
GrayU8 = NDArray[np.uint8]
RgbU8 = NDArray[np.uint8]

CheckboxShape = Literal["square", "circle"]


class CheckboxFillState(StrEnum):
    """Measured fill of a detected control."""

    EMPTY = "empty"
    TICKED = "ticked"
    CROSSED = "crossed"


class CheckboxDetection(BaseModel):
    """One detected checkbox/radio with fill measurement and optional label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: Region
    shape: CheckboxShape
    fill_ratio: float = Field(ge=0.0, le=1.0)
    state: CheckboxFillState
    pixel_bbox: tuple[int, int, int, int] | None = None


def detect_checkbox_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutCheckboxConfig | None = None,
) -> list[Region]:
    """Locate checkbox / radio controls as ``RegionKind.CHECKBOX`` regions."""
    detections = detect_checkboxes(image, page_index=page_index, config=config)
    return [d.region for d in detections]


def detect_checkboxes(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutCheckboxConfig | None = None,
    label_regions: list[Region] | None = None,
) -> list[CheckboxDetection]:
    """Detect controls, measure fill, optionally map adjacent labels."""
    cfg = config if config is not None else LayoutCheckboxConfig()
    if not cfg.enabled:
        return []

    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    if height < 16 or width < 16:
        return []

    gray = cast(GrayU8, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray, cfg.ink_threshold)

    candidates = _find_control_boxes(gray, ink, width, height, cfg)
    candidates = _nms_shape_boxes(candidates, iou=cfg.nms_iou)

    detections: list[CheckboxDetection] = []
    for box, shape in candidates:
        fill_ratio, state = _measure_fill(ink, box, cfg)
        region = _make_checkbox_region(box, width, height, page_index)
        detections.append(
            CheckboxDetection(
                region=region,
                shape=shape,
                fill_ratio=fill_ratio,
                state=state,
                pixel_bbox=box,
            )
        )

    if label_regions:
        detections = map_checkbox_labels(detections, label_regions, config=cfg)
    return detections


def measure_checkbox_fill(
    image: Image.Image,
    region: Region,
    *,
    config: LayoutCheckboxConfig | None = None,
) -> tuple[float, CheckboxFillState]:
    """Return ``(fill_ratio, state)`` for an existing checkbox region."""
    cfg = config if config is not None else LayoutCheckboxConfig()
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    box = _norm_to_pixel(region.bbox, width, height)
    gray = cast(GrayU8, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray, cfg.ink_threshold)
    return _measure_fill(ink, box, cfg)


def map_checkbox_labels(
    detections: list[CheckboxDetection],
    label_regions: list[Region],
    *,
    config: LayoutCheckboxConfig | None = None,
) -> list[CheckboxDetection]:
    """Attach the nearest adjacent option label to each checkbox.

    Prefers labels to the right of the control (same row), then left, then
    below. Each label region is used at most once.
    """
    cfg = config if config is not None else LayoutCheckboxConfig()
    labels = [
        r
        for r in label_regions
        if r.kind in (RegionKind.TEXT_LINE, RegionKind.FIELD_VALUE)
    ]
    used: set[str] = set()
    out: list[CheckboxDetection] = []

    # Process left-to-right, top-to-bottom so groups stay stable.
    ordered = sorted(
        detections,
        key=lambda d: (d.region.bbox.y0, d.region.bbox.x0),
    )
    for det in ordered:
        match = _best_label(det.region.bbox, labels, used, cfg)
        if match is None:
            out.append(det)
            continue
        used.add(match.id)
        text = match.label_anchor.text if match.label_anchor else None
        # Prefer explicit label text carried on the region via a prior OCR pass;
        # synth fixtures stash the option string in RegionTruth, not on Region.
        anchor = LabelAnchor(text=text, bbox=match.bbox)
        region = det.region.model_copy(update={"label_anchor": anchor})
        out.append(det.model_copy(update={"region": region}))
    return out


def map_labels_by_bbox(
    detections: list[CheckboxDetection],
    label_regions: list[Region],
    label_texts: dict[str, str],
    *,
    config: LayoutCheckboxConfig | None = None,
) -> list[CheckboxDetection]:
    """Map labels spatially, then copy option text from ``label_texts``."""
    mapped = map_checkbox_labels(detections, label_regions, config=config)
    out: list[CheckboxDetection] = []
    for det in mapped:
        anchor = det.region.label_anchor
        if anchor is None or anchor.bbox is None:
            out.append(det)
            continue
        text: str | None = None
        best_iou = 0.0
        best_id: str | None = None
        for lab in label_regions:
            score = bbox_iou(anchor.bbox, lab.bbox)
            if score > best_iou:
                best_iou = score
                best_id = lab.id
        if best_id is not None and best_id in label_texts and best_iou >= 0.5:
            text = label_texts[best_id]
        elif anchor.text:
            text = anchor.text
        new_anchor = LabelAnchor(text=text, bbox=anchor.bbox)
        region = det.region.model_copy(update={"label_anchor": new_anchor})
        out.append(det.model_copy(update={"region": region}))
    return out


def merge_checkboxes_into_regions(
    existing: list[Region],
    checkboxes: list[Region],
    *,
    nms_iou: float = 0.4,
) -> list[Region]:
    """Insert checkbox regions; drop overlapping same-kind duplicates."""
    merged = list(existing)
    for cand in checkboxes:
        if any(
            r.kind == RegionKind.CHECKBOX and bbox_iou(r.bbox, cand.bbox) >= nms_iou
            for r in merged
        ):
            continue
        merged.append(cand)
    return merged


def process_checkboxes(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutCheckboxConfig | None = None,
    label_regions: list[Region] | None = None,
    label_texts: dict[str, str] | None = None,
) -> list[CheckboxDetection]:
    """End-to-end: detect, measure fill, map adjacent labels (+ optional text)."""
    detections = detect_checkboxes(
        image,
        page_index=page_index,
        config=config,
        label_regions=label_regions,
    )
    if label_regions and label_texts:
        detections = map_labels_by_bbox(
            detections, label_regions, label_texts, config=config
        )
    return detections


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _ink_mask(gray: GrayU8, threshold: int) -> GrayU8:
    # Dark ink → 255.
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return cast(GrayU8, binary)


def _find_control_boxes(
    gray: GrayU8,
    ink: GrayU8,
    width: int,
    height: int,
    cfg: LayoutCheckboxConfig,
) -> list[tuple[PixelBox, CheckboxShape]]:
    """Locate square checkboxes and circular radios via contours + Hough."""
    # Floor at 16 px so typical glyph strokes (font ~18) are not controls.
    min_side = max(16, int(min(width, height) * cfg.min_side_ratio))
    max_side = max(min_side + 1, int(min(width, height) * cfg.max_side_ratio))

    found: list[tuple[PixelBox, CheckboxShape]] = []
    found.extend(_contour_control_boxes(ink, min_side, max_side, cfg))
    found.extend(_contour_control_boxes(_edge_mask(gray), min_side, max_side, cfg))
    found.extend(_hough_circle_boxes(gray, ink, width, height, min_side, max_side, cfg))
    return found


def _edge_mask(gray: GrayU8) -> GrayU8:
    blurred = cast(GrayU8, cv2.GaussianBlur(gray, (3, 3), 0))
    edges = cast(GrayU8, cv2.Canny(blurred, 40, 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cast(GrayU8, cv2.dilate(edges, kernel, iterations=1))


def _contour_control_boxes(
    mask: GrayU8,
    min_side: int,
    max_side: int,
    cfg: LayoutCheckboxConfig,
) -> list[tuple[PixelBox, CheckboxShape]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[PixelBox, CheckboxShape]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue
        aspect = w / max(h, 1)
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        peri = cv2.arcLength(contour, True)
        if peri < 2.5 * (w + h) * 0.5:
            continue
        area = float(cv2.contourArea(contour))
        rect_area = float(max(w * h, 1))
        extent = area / rect_area
        # Outline-like contours occupy a modest fraction of the bbox.
        if extent < 0.08 or extent > 0.95:
            continue
        box: PixelBox = (x, y, x + w, y + h)
        if not _looks_like_control_frame(mask, box, cfg):
            continue
        circularity = 4.0 * np.pi * area / max(peri * peri, 1e-6)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        n_approx = len(approx)
        # Prefer square when the polygon looks rectangular.
        if 4 <= n_approx <= 6:
            shape: CheckboxShape = "square"
        elif circularity >= cfg.min_circularity and n_approx >= 6:
            shape = "circle"
        elif abs(aspect - 1.0) <= 0.25:
            shape = "square"
        else:
            continue
        found.append((box, shape))
    return found


def _hough_circle_boxes(
    gray: GrayU8,
    ink: GrayU8,
    width: int,
    height: int,
    min_side: int,
    max_side: int,
    cfg: LayoutCheckboxConfig,
) -> list[tuple[PixelBox, CheckboxShape]]:
    # Hough cost explodes on full-resolution scans, so search a downscaled copy
    # and map circles back. Keep the smallest radius at >= 4 px after scaling.
    scale = min(1.0, cfg.hough_max_side / max(width, height))
    scale = max(scale, min(1.0, 4.0 / max(min_side / 2.0, 1.0)))
    if scale < 1.0:
        small = cast(
            GrayU8,
            cv2.resize(
                gray,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ),
        )
    else:
        small = gray
        scale = 1.0
    min_r = max(4 if scale < 1.0 else 5, int(min_side * scale) // 2)
    max_r = max(min_r + 1, int(max_side * scale) // 2)
    blurred = cast(GrayU8, cv2.GaussianBlur(small, (5, 5), 0))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(max(int(min_side * scale), 8)),
        param1=100,
        param2=cfg.hough_param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    found: list[tuple[PixelBox, CheckboxShape]] = []
    if circles is None:
        return found
    for cx, cy, r in circles[0]:
        r_i = int(round(float(r) / scale))
        cx_i = int(round(float(cx) / scale))
        cy_i = int(round(float(cy) / scale))
        x0 = max(0, cx_i - r_i)
        y0 = max(0, cy_i - r_i)
        x1 = min(width, cx_i + r_i + 1)
        y1 = min(height, cy_i + r_i + 1)
        if x1 - x0 < min_side or y1 - y0 < min_side:
            continue
        if x1 - x0 > max_side or y1 - y0 > max_side:
            continue
        box = (x0, y0, x1, y1)
        if not _looks_like_control_frame(ink, box, cfg):
            continue
        found.append((box, "circle"))
    return found


def _looks_like_control_frame(
    ink: GrayU8,
    box: PixelBox,
    cfg: LayoutCheckboxConfig,
) -> bool:
    """True when the bbox has a dark perimeter ring and a non-solid interior."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w < 6 or h < 6:
        return False
    border = max(1, min(3, min(w, h) // 8))
    roi = ink[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    peri = np.zeros_like(roi)
    peri[:border, :] = 1
    peri[-border:, :] = 1
    peri[:, :border] = 1
    peri[:, -border:] = 1
    peri_n = int(np.count_nonzero(peri))
    if peri_n == 0:
        return False
    peri_fill = float(np.count_nonzero(roi[peri.astype(bool)])) / float(peri_n)

    inner = _interior_roi(ink, box, border=border + 1)
    if inner.size == 0:
        return peri_fill >= 0.25
    inner_fill = float(np.count_nonzero(inner)) / float(inner.size)
    if inner_fill > cfg.max_solid_fill:
        return False
    # Empty or marked controls both need a visible frame ring.
    ring_score = peri_fill - 0.35 * inner_fill
    return peri_fill >= 0.30 and ring_score >= 0.12


def _nms_shape_boxes(
    items: list[tuple[PixelBox, CheckboxShape]],
    *,
    iou: float,
) -> list[tuple[PixelBox, CheckboxShape]]:
    if not items:
        return []
    ordered = sorted(
        items,
        key=lambda it: (it[0][2] - it[0][0]) * (it[0][3] - it[0][1]),
        reverse=True,
    )
    kept: list[tuple[PixelBox, CheckboxShape]] = []
    for box, shape in ordered:
        nb = pixel_to_norm_bbox(
            x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=10_000, height=10_000
        )
        if any(
            bbox_iou(
                nb,
                pixel_to_norm_bbox(
                    x0=k[0][0],
                    y0=k[0][1],
                    x1=k[0][2],
                    y1=k[0][3],
                    width=10_000,
                    height=10_000,
                ),
            )
            >= iou
            for k in kept
        ):
            continue
        kept.append((box, shape))
    return kept


# ---------------------------------------------------------------------------
# Fill measurement
# ---------------------------------------------------------------------------


def _interior_roi(ink: GrayU8, box: PixelBox, *, border: int) -> GrayU8:
    x0, y0, x1, y1 = box
    ix0 = min(x1 - 1, x0 + border)
    iy0 = min(y1 - 1, y0 + border)
    ix1 = max(ix0 + 1, x1 - border)
    iy1 = max(iy0 + 1, y1 - border)
    if ix1 <= ix0 or iy1 <= iy0:
        return ink[0:0, 0:0]
    return ink[iy0:iy1, ix0:ix1]


def _measure_fill(
    ink: GrayU8,
    box: PixelBox,
    cfg: LayoutCheckboxConfig,
) -> tuple[float, CheckboxFillState]:
    border = max(2, int(round(min(box[2] - box[0], box[3] - box[1]) * 0.18)))
    interior = _interior_roi(ink, box, border=border)
    if interior.size == 0:
        return 0.0, CheckboxFillState.EMPTY

    fill_ratio = float(np.count_nonzero(interior)) / float(interior.size)
    if fill_ratio < cfg.empty_fill_max:
        return fill_ratio, CheckboxFillState.EMPTY

    diag_main, diag_anti = _diagonal_ink_ratios(interior)
    # Both diagonals inked → X (crossed). One diagonal or L-shaped → tick.
    if (
        diag_main >= cfg.diagonal_ink_min
        and diag_anti >= cfg.diagonal_ink_min
        and fill_ratio >= cfg.cross_fill_min
    ):
        return fill_ratio, CheckboxFillState.CROSSED

    if fill_ratio >= cfg.tick_fill_min:
        # Prefer tick when only one diagonal is strong, or ink is asymmetric.
        if abs(diag_main - diag_anti) >= cfg.diagonal_asymmetry_min:
            return fill_ratio, CheckboxFillState.TICKED
        if diag_main >= cfg.diagonal_ink_min or diag_anti >= cfg.diagonal_ink_min:
            # Ambiguous: if both moderate, call crossed; else ticked.
            if min(diag_main, diag_anti) >= cfg.diagonal_ink_min * 0.7:
                return fill_ratio, CheckboxFillState.CROSSED
            return fill_ratio, CheckboxFillState.TICKED
        return fill_ratio, CheckboxFillState.TICKED

    return fill_ratio, CheckboxFillState.EMPTY


def _diagonal_ink_ratios(interior: GrayU8) -> tuple[float, float]:
    """Fraction of main / anti-diagonal band pixels that are ink."""
    h, w = interior.shape[:2]
    if h < 3 or w < 3:
        return 0.0, 0.0
    yy, xx = np.indices((h, w))
    # Band width ~ 15% of min side.
    band = max(1.0, 0.15 * min(h, w))
    # Main diagonal: y/h ≈ x/w
    main_dist = np.abs(yy * (w - 1) - xx * (h - 1)) / max(h + w, 1)
    anti_dist = np.abs(yy * (w - 1) - (w - 1 - xx) * (h - 1)) / max(h + w, 1)
    # Convert index distance into approx pixel band via scale.
    scale = max(h, w) / max(h + w, 1)
    main_mask = main_dist * (1.0 / max(scale, 1e-6)) <= band
    anti_mask = anti_dist * (1.0 / max(scale, 1e-6)) <= band
    main_n = int(np.count_nonzero(main_mask))
    anti_n = int(np.count_nonzero(anti_mask))
    main_ink = (
        float(np.count_nonzero(interior[main_mask])) / float(main_n) if main_n else 0.0
    )
    anti_ink = (
        float(np.count_nonzero(interior[anti_mask])) / float(anti_n) if anti_n else 0.0
    )
    return main_ink, anti_ink


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def _best_label(
    box: BBox,
    labels: list[Region],
    used: set[str],
    cfg: LayoutCheckboxConfig,
) -> Region | None:
    candidates: list[tuple[float, Region]] = []
    for lab in labels:
        if lab.id in used:
            continue
        score = _label_affinity(box, lab.bbox, cfg)
        if score is None:
            continue
        candidates.append((score, lab))
    if not candidates:
        return None
    # Higher affinity is better; break ties by distance (encoded in score).
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _label_affinity(
    control: BBox,
    label: BBox,
    cfg: LayoutCheckboxConfig,
) -> float | None:
    """Score how well ``label`` sits as the option text for ``control``."""
    y_overlap = _y_overlap_ratio(control, label)
    x_overlap = _x_overlap_ratio(control, label)
    ctrl_h = max(control.y1 - control.y0, 1e-6)
    ctrl_w = max(control.x1 - control.x0, 1e-6)

    # Right of control, same row.
    right_gap = label.x0 - control.x1
    if (
        right_gap >= -0.01
        and right_gap <= cfg.max_label_gap
        and y_overlap >= cfg.row_y_overlap_min
    ):
        return 3.0 - right_gap / max(cfg.max_label_gap, 1e-6)

    # Left of control, same row.
    left_gap = control.x0 - label.x1
    if (
        left_gap >= -0.01
        and left_gap <= cfg.max_label_gap
        and y_overlap >= cfg.row_y_overlap_min
    ):
        return 2.0 - left_gap / max(cfg.max_label_gap, 1e-6)

    # Below control (rare; stacked captions).
    below_gap = label.y0 - control.y1
    if below_gap >= -0.01 and below_gap <= cfg.max_label_gap and x_overlap >= 0.2:
        return 1.0 - below_gap / max(cfg.max_label_gap, 1e-6)

    # Centres nearly aligned vertically within a band — soft same-row.
    cy_c = (control.y0 + control.y1) / 2.0
    cy_l = (label.y0 + label.y1) / 2.0
    if abs(cy_c - cy_l) <= ctrl_h and abs(label.x0 - control.x1) <= cfg.max_label_gap:
        gap = abs(label.x0 - control.x1)
        return 1.5 - gap / max(cfg.max_label_gap, 1e-6)

    # Very close centres (overlapping mis-detect) — reject.
    _ = ctrl_w
    return None


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    iy0 = max(a.y0, b.y0)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, iy1 - iy0)
    denom = min(max(0.0, a.y1 - a.y0), max(0.0, b.y1 - b.y0))
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    ix0 = max(a.x0, b.x0)
    ix1 = min(a.x1, b.x1)
    inter = max(0.0, ix1 - ix0)
    denom = min(max(0.0, a.x1 - a.x0), max(0.0, b.x1 - b.x0))
    if denom <= 0.0:
        return 0.0
    return inter / denom


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------


def _make_checkbox_region(
    box: PixelBox,
    width: int,
    height: int,
    page_index: int,
) -> Region:
    bbox = pixel_to_norm_bbox(
        x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=width, height=height
    )
    key = (
        f"{page_index}:checkbox:{bbox.x0:.4f}:{bbox.y0:.4f}:{bbox.x1:.4f}:{bbox.y1:.4f}"
    )
    rid = f"checkbox-{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    return Region(id=rid, page_index=page_index, bbox=bbox, kind=RegionKind.CHECKBOX)


def _norm_to_pixel(bbox: BBox, width: int, height: int) -> PixelBox:
    x0 = int(max(0, min(width - 1, round(bbox.x0 * width))))
    y0 = int(max(0, min(height - 1, round(bbox.y0 * height))))
    x1 = int(max(x0 + 1, min(width, round(bbox.x1 * width))))
    y1 = int(max(y0 + 1, min(height, round(bbox.y1 * height))))
    return (x0, y0, x1, y1)
