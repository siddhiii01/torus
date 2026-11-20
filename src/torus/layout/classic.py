"""Classic (OpenCV) layout detectors — no model weights required.

Finds text lines / blocks, table cells, stamps, signatures and photos from
image structure alone. All thresholds come from ``LayoutClassicConfig``;
nothing is keyed to absolute page coordinates.
"""

from __future__ import annotations

import hashlib
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from torus.config.schemas import LayoutClassicConfig
from torus.core.models import BBox, Region
from torus.core.types import RegionKind
from torus.layout.geometry import pixel_to_norm_bbox

PixelBox = tuple[int, int, int, int]  # x0, y0, x1, y1


def detect_classic_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutClassicConfig | None = None,
) -> list[Region]:
    """Run all classic detectors and return merged ``Region`` objects."""
    cfg = config if config is not None else LayoutClassicConfig()
    if not cfg.enabled:
        return []

    rgb = cast(NDArray[np.uint8], np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return []

    gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    ink = _ink_mask(gray)

    regions: list[Region] = []
    occupied: list[tuple[RegionKind, BBox]] = []
    photo_boxes: list[PixelBox] = []

    if cfg.detect_photos:
        for box in _detect_photos(rgb, gray, cfg):
            region = _make_region(RegionKind.PHOTO, box, width, height, page_index)
            regions.append(region)
            occupied.append((region.kind, region.bbox))
            photo_boxes.append(box)

    if cfg.detect_stamps:
        for box in _detect_stamps(rgb, cfg):
            if _box_overlaps_any(box, photo_boxes, iou=0.25):
                continue
            region = _make_region(RegionKind.STAMP, box, width, height, page_index)
            if not _suppressed(region, occupied, cfg.nms_iou):
                regions.append(region)
                occupied.append((region.kind, region.bbox))

    table_boxes: list[PixelBox] = []
    if cfg.detect_tables:
        for box in _detect_table_cells(ink, cfg):
            region = _make_region(RegionKind.TABLE_CELL, box, width, height, page_index)
            if not _suppressed(region, occupied, cfg.nms_iou):
                regions.append(region)
                occupied.append((region.kind, region.bbox))
                table_boxes.append(box)

    # Signatures before text lines so sparse stroke clusters are not absorbed
    # into line morphology and then excluded.
    signature_boxes: list[PixelBox] = []
    if cfg.detect_signatures:
        for box in _detect_signatures(ink, cfg, exclude=table_boxes + photo_boxes):
            region = _make_region(RegionKind.SIGNATURE, box, width, height, page_index)
            if not _suppressed(region, occupied, cfg.nms_iou):
                regions.append(region)
                occupied.append((region.kind, region.bbox))
                signature_boxes.append(box)

    line_boxes: list[PixelBox] = []
    if cfg.detect_text_lines:
        line_boxes = _detect_text_lines(
            ink,
            cfg,
            exclude=table_boxes + signature_boxes + photo_boxes,
        )
        for box in line_boxes:
            region = _make_region(RegionKind.TEXT_LINE, box, width, height, page_index)
            if not _suppressed(region, occupied, cfg.nms_iou):
                regions.append(region)
                occupied.append((region.kind, region.bbox))

    if cfg.detect_text_blocks and line_boxes:
        for box in _cluster_text_blocks(line_boxes, cfg):
            region = _make_region(
                RegionKind.TEXT_LINE, box, width, height, page_index, prefix="block"
            )
            # Blocks may heavily overlap their lines; skip NMS against TEXT_LINE.
            if not _suppressed(
                region,
                [(k, b) for k, b in occupied if k != RegionKind.TEXT_LINE],
                cfg.nms_iou,
            ):
                regions.append(region)
                occupied.append((region.kind, region.bbox))

    return regions


def _make_region(
    kind: RegionKind,
    box: PixelBox,
    width: int,
    height: int,
    page_index: int,
    *,
    prefix: str | None = None,
) -> Region:
    bbox = pixel_to_norm_bbox(
        x0=box[0],
        y0=box[1],
        x1=box[2],
        y1=box[3],
        width=width,
        height=height,
    )
    tag = prefix if prefix is not None else kind.value
    key = f"{page_index}:{tag}:{bbox.x0:.4f}:{bbox.y0:.4f}:{bbox.x1:.4f}:{bbox.y1:.4f}"
    rid = f"{tag}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    return Region(id=rid, page_index=page_index, bbox=bbox, kind=kind)


def _suppressed(
    candidate: Region,
    occupied: list[tuple[RegionKind, BBox]],
    nms_iou: float,
) -> bool:
    from torus.layout.geometry import bbox_iou

    for kind, bbox in occupied:
        if kind != candidate.kind:
            continue
        if bbox_iou(candidate.bbox, bbox) >= nms_iou:
            return True
    return False


def _box_overlaps_any(
    box: PixelBox,
    others: list[PixelBox],
    *,
    iou: float,
) -> bool:
    if not others:
        return False
    # Use a large virtual canvas so pixel boxes normalise consistently.
    cand = pixel_to_norm_bbox(
        x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=10_000, height=10_000
    )
    from torus.layout.geometry import bbox_iou

    for other in others:
        ref = pixel_to_norm_bbox(
            x0=other[0],
            y0=other[1],
            x1=other[2],
            y1=other[3],
            width=10_000,
            height=10_000,
        )
        if bbox_iou(cand, ref) >= iou:
            return True
    return False


def _ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binary ink mask (255 = ink)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Otsu on inverted: dark ink → white.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cast(NDArray[np.uint8], binary)


def _detect_text_lines(
    ink: NDArray[np.uint8],
    cfg: LayoutClassicConfig,
    *,
    exclude: list[PixelBox],
) -> list[PixelBox]:
    height, width = ink.shape[:2]
    # Connect characters within a line.
    kx = max(3, int(round(cfg.line_close_width_ratio * width)))
    if kx % 2 == 0:
        kx += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)

    # Suppress known table / photo areas so lines don't swallow grids.
    mask = closed.copy()
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = 0

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[PixelBox] = []
    min_area = max(8, int(cfg.line_min_area_ratio * width * height))
    max_h = int(cfg.line_max_height_ratio * height)
    min_w = int(cfg.line_min_width_ratio * width)

    for i in range(1, n_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area or h > max_h or w < min_w:
            continue
        if h <= 0:
            continue
        aspect = w / float(h)
        if aspect < cfg.line_min_aspect:
            continue
        boxes.append((x, y, x + w, y + h))

    return _merge_horizontal(boxes, y_overlap=0.55, x_gap=int(0.02 * width))


def _cluster_text_blocks(
    lines: list[PixelBox],
    cfg: LayoutClassicConfig,
) -> list[PixelBox]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda b: (b[1], b[0]))
    clusters: list[list[PixelBox]] = [[ordered[0]]]
    for box in ordered[1:]:
        prev = clusters[-1][-1]
        prev_h = max(1, prev[3] - prev[1])
        gap = box[1] - prev[3]
        x_overlap = min(prev[2], box[2]) - max(prev[0], box[0])
        min_w = min(prev[2] - prev[0], box[2] - box[0])
        if gap <= cfg.block_max_line_gap_ratio * prev_h and x_overlap > 0.3 * min_w:
            clusters[-1].append(box)
        else:
            clusters.append([box])

    blocks: list[PixelBox] = []
    for group in clusters:
        if len(group) < cfg.block_min_lines:
            continue
        x0 = min(b[0] for b in group)
        y0 = min(b[1] for b in group)
        x1 = max(b[2] for b in group)
        y1 = max(b[3] for b in group)
        blocks.append((x0, y0, x1, y1))
    return blocks


def _detect_table_cells(
    ink: NDArray[np.uint8],
    cfg: LayoutClassicConfig,
) -> list[PixelBox]:
    """Recover table cells from local H/V ruling grids.

    Line projections are computed *inside* candidate table blobs so strong
    ink elsewhere on the page cannot dominate the peaks.
    """
    height, width = ink.shape[:2]
    h_len = max(12, int(cfg.table_line_min_length_ratio * width))
    v_len = max(12, int(cfg.table_line_min_length_ratio * height))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, h_close)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, v_close)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, v_kernel)

    if np.count_nonzero(horizontal) < 10 or np.count_nonzero(vertical) < 10:
        return []

    # Candidate tables = connected components of the H∪V ruling network.
    corner = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid = cv2.bitwise_or(
        cv2.dilate(horizontal, corner),
        cv2.dilate(vertical, corner),
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    cells: list[PixelBox] = []
    min_area = max(20, int(cfg.table_min_cell_area_ratio * width * height))
    max_area = int(cfg.table_max_cell_area_ratio * width * height)
    min_table_area = max(80, int(0.001 * width * height))

    for i in range(1, n_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_table_area or w < 24 or h < 24:
            continue
        # Component must contain both horizontal and vertical rulings.
        comp = labels[y : y + h, x : x + w] == i
        h_roi_full = horizontal[y : y + h, x : x + w]
        v_roi_full = vertical[y : y + h, x : x + w]
        h_count = int(np.count_nonzero(h_roi_full[comp]))
        v_count = int(np.count_nonzero(v_roi_full[comp]))
        if h_count < 8 or v_count < 8:
            continue

        x0 = max(0, x - 2)
        y0 = max(0, y - 2)
        x1 = min(width, x + w + 2)
        y1 = min(height, y + h + 2)
        h_roi = cast(NDArray[np.uint8], horizontal[y0:y1, x0:x1])
        v_roi = cast(NDArray[np.uint8], vertical[y0:y1, x0:x1])

        local_ys = _cluster_line_positions(
            _projection_peaks(h_roi, axis=1),
            tol=max(2, int(0.05 * h)),
        )
        local_xs = _cluster_line_positions(
            _projection_peaks(v_roi, axis=0),
            tol=max(2, int(0.05 * w)),
        )
        if len(local_xs) < 2 or len(local_ys) < 2:
            continue

        xs = [x0 + v for v in local_xs]
        ys = [y0 + v for v in local_ys]
        for yi in range(len(ys) - 1):
            for xi in range(len(xs) - 1):
                cx0, cx1 = xs[xi], xs[xi + 1]
                cy0, cy1 = ys[yi], ys[yi + 1]
                box = (cx0 + 1, cy0 + 1, cx1, cy1)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                cell_area = bw * bh
                if bw < 6 or bh < 6 or cell_area < min_area or cell_area > max_area:
                    continue
                cells.append(box)

    if len(cells) < cfg.table_min_cells:
        return []
    return cells


def _projection_peaks(
    mask: NDArray[np.uint8],
    *,
    axis: int,
) -> list[int]:
    """Return peak indices for line projections.

    ``axis`` is the image axis being *reduced* (summed over):
    - ``axis=1`` (sum across columns) → row / Y peaks for horizontal rulings
    - ``axis=0`` (sum across rows) → column / X peaks for vertical rulings
    """
    proj = np.sum(mask > 0, axis=axis)
    if proj.size == 0 or int(proj.max()) == 0:
        return []
    thresh = max(3, int(0.35 * float(proj.max())))
    return [int(i) for i, v in enumerate(proj) if int(v) >= thresh]


def _cluster_line_positions(positions: list[int], *, tol: int) -> list[int]:
    if not positions:
        return []
    ordered = sorted(positions)
    groups: list[list[int]] = [[ordered[0]]]
    for p in ordered[1:]:
        if p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(sum(g) / len(g))) for g in groups]


def _detect_stamps(
    rgb: NDArray[np.uint8],
    cfg: LayoutClassicConfig,
) -> list[PixelBox]:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Saturated ink (red / blue stamps) — no absolute x/y.
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Red (wraps 0/180) or blue-ish stamps; reject continuous-tone photo chroma.
    red = ((hue <= 10) | (hue >= 170)) & (sat >= cfg.stamp_min_saturation)
    blue = (hue >= 90) & (hue <= 130) & (sat >= cfg.stamp_min_saturation)
    chroma = ((red | blue) & (val >= cfg.stamp_min_value)).astype(np.uint8) * 255

    boxes: list[PixelBox] = []
    min_area = max(40, int(cfg.stamp_min_area_ratio * width * height))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        chroma, connectivity=8
    )
    for i in range(1, n_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area:
            continue
        aspect = w / float(max(h, 1))
        if aspect < 0.55 or aspect > 1.8:
            continue
        # Ring-like: outer third denser than centre third.
        cy0, cy1 = y + h // 3, y + 2 * h // 3
        cx0, cx1 = x + w // 3, x + 2 * w // 3
        centre = labels[cy0:cy1, cx0:cx1] == i
        full = labels[y : y + h, x : x + w] == i
        centre_fill = float(np.count_nonzero(centre)) / float(max(centre.size, 1))
        full_fill = float(np.count_nonzero(full)) / float(max(w * h, 1))
        if full_fill > 0.55 or centre_fill > full_fill * 1.2:
            continue
        boxes.append((x, y, x + w, y + h))

    # Hough only on the chroma mask (not page edges) to avoid floods of hits.
    if np.count_nonzero(chroma) > 0:
        # Hough cost grows steeply with image size and radius range, so search a
        # downscaled mask and map circles back to full resolution.
        scale = min(1.0, cfg.stamp_hough_max_side / float(max(width, height)))
        small = chroma
        if scale < 1.0:
            small = cast(
                NDArray[np.uint8],
                cv2.resize(
                    chroma,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                ),
            )
        min_dim = min(small.shape[:2])
        min_r = max(4, int(cfg.stamp_min_radius_ratio * min_dim))
        max_r = max(min_r + 1, int(cfg.stamp_max_radius_ratio * min_dim))
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(small, (5, 5), 0),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12, min_r * 2),
            param1=80,
            param2=cfg.stamp_hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for cx_s, cy_s, r_s in circles[0]:
                cx, cy, r = cx_s / scale, cy_s / scale, r_s / scale
                x0 = max(0, int(cx - r))
                y0 = max(0, int(cy - r))
                x1 = min(width, int(cx + r))
                y1 = min(height, int(cy + r))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                # Require chroma support inside the circle.
                support = float(np.count_nonzero(chroma[y0:y1, x0:x1])) / float(
                    max((x1 - x0) * (y1 - y0), 1)
                )
                if support < 0.02:
                    continue
                boxes.append((x0, y0, x1, y1))

    return _nms_boxes(boxes, iou=0.4)


def _detect_signatures(
    ink: NDArray[np.uint8],
    cfg: LayoutClassicConfig,
    *,
    exclude: list[PixelBox],
) -> list[PixelBox]:
    height, width = ink.shape[:2]
    mask = ink.copy()
    for x0, y0, x1, y1 in exclude:
        pad = 2
        mask[
            max(0, y0 - pad) : min(height, y1 + pad),
            max(0, x0 - pad) : min(width, x1 + pad),
        ] = 0

    # Join nearby strokes horizontally without filling the whole box.
    kx = max(7, int(0.02 * width))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    boxes: list[PixelBox] = []
    min_area = max(20, int(cfg.signature_min_area_ratio * width * height))
    max_area = int(cfg.signature_max_area_ratio * width * height)
    for i in range(1, n_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area or area > max_area:
            continue
        if h <= 0 or w <= 0:
            continue
        aspect = w / float(h)
        if aspect < cfg.signature_min_aspect:
            continue
        if h > int(cfg.signature_max_height_ratio * height):
            continue
        # Sparse fill → stroke-like, not a filled photo/stamp.
        roi = labels[y : y + h, x : x + w] == i
        fill = float(np.count_nonzero(roi)) / float(max(w * h, 1))
        if fill > cfg.signature_max_fill_ratio:
            continue
        # Reject grid-like regions (tables): many H and V rulings.
        roi_ink = ink[y : y + h, x : x + w]
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 3), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 2)))
        h_lines = cv2.morphologyEx(roi_ink, cv2.MORPH_OPEN, hk)
        v_lines = cv2.morphologyEx(roi_ink, cv2.MORPH_OPEN, vk)
        if (
            np.count_nonzero(h_lines) > 0.08 * w * h
            and np.count_nonzero(v_lines) > 0.08 * w * h
        ):
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _detect_photos(
    rgb: NDArray[np.uint8],
    gray: NDArray[np.uint8],
    cfg: LayoutClassicConfig,
) -> list[PixelBox]:
    height, width = rgb.shape[:2]
    blur = cv2.GaussianBlur(rgb, (5, 5), 0)
    f32 = blur.astype(np.float32)
    mean = np.mean(f32, axis=2)
    var = np.mean((f32 - mean[:, :, None]) ** 2, axis=2)
    # Absolute channel variance (not min-max normalised) so thresholds are stable.
    high_var = (var >= float(cfg.photo_variance_threshold)).astype(np.uint8) * 255

    edges = cv2.Canny(gray, 50, 150)
    # Suppress only dense text-edge zones, not sparse photo borders.
    edge_density = cv2.blur(edges.astype(np.float32), (15, 15))
    texty = (edge_density >= cfg.photo_text_edge_density).astype(np.uint8) * 255
    high_var[texty > 0] = 0

    k = max(7, int(0.025 * min(width, height)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blob = cv2.morphologyEx(high_var, cv2.MORPH_CLOSE, kernel)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    boxes: list[PixelBox] = []
    min_area = max(200, int(cfg.photo_min_area_ratio * width * height))
    max_area = int(cfg.photo_max_area_ratio * width * height)
    for i in range(1, n_labels):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area or area > max_area:
            continue
        aspect = w / float(max(h, 1))
        if aspect < cfg.photo_min_aspect or aspect > cfg.photo_max_aspect:
            continue
        roi_edges = edges[y : y + h, x : x + w]
        edge_ratio = float(np.count_nonzero(roi_edges)) / float(max(w * h, 1))
        if edge_ratio > cfg.photo_max_edge_ratio:
            continue
        # Continuous-tone: mean channel variance should stay high.
        roi_var = float(np.mean(var[y : y + h, x : x + w]))
        if roi_var < float(cfg.photo_variance_threshold):
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _merge_horizontal(
    boxes: list[PixelBox],
    *,
    y_overlap: float,
    x_gap: int,
) -> list[PixelBox]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[PixelBox] = [ordered[0]]
    for box in ordered[1:]:
        cur = merged[-1]
        # Vertical overlap ratio relative to the shorter box.
        oh = min(cur[3], box[3]) - max(cur[1], box[1])
        min_h = min(cur[3] - cur[1], box[3] - box[1])
        if min_h <= 0 or oh / float(min_h) < y_overlap:
            merged.append(box)
            continue
        gap = box[0] - cur[2]
        if gap > x_gap:
            merged.append(box)
            continue
        merged[-1] = (
            min(cur[0], box[0]),
            min(cur[1], box[1]),
            max(cur[2], box[2]),
            max(cur[3], box[3]),
        )
    return merged


def _nms_boxes(boxes: list[PixelBox], *, iou: float) -> list[PixelBox]:
    if not boxes:
        return []
    # Greedy by area descending.
    scored = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep: list[PixelBox] = []
    for box in scored:
        bx = pixel_to_norm_bbox(
            x0=box[0], y0=box[1], x1=box[2], y1=box[3], width=10_000, height=10_000
        )
        drop = False
        for kept in keep:
            kx = pixel_to_norm_bbox(
                x0=kept[0],
                y0=kept[1],
                x1=kept[2],
                y1=kept[3],
                width=10_000,
                height=10_000,
            )
            from torus.layout.geometry import bbox_iou

            if bbox_iou(bx, kx) >= iou:
                drop = True
                break
        if not drop:
            keep.append(box)
    return keep
