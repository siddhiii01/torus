"""Photo perspective correction (P2.3).

Finds the largest plausible document quadrilateral in a phone photo and
warps it to a frontal rectangle. Flat scans that already fill the frame are
left alone (area ratio gate).
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import PerspectiveConfig
from torus.core.models import Page


class PerspectiveResult(BaseModel):
    """Outcome of document-quad detection / optional warp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    quad_area_ratio: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    # Clockwise from top-left in source pixel coords: TL, TR, BR, BL.
    source_quad: (
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
        | None
    ) = None


def detect_perspective(
    image: Image.Image,
    config: PerspectiveConfig | None = None,
) -> PerspectiveResult:
    """Locate a document quad without mutating the image."""
    cfg = config if config is not None else PerspectiveConfig()
    if not cfg.enabled:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=0.0,
            reason="disabled",
        )

    rgb = np.asarray(image.convert("RGB"))
    quad = _find_document_quad(rgb)
    if quad is None:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=0.0,
            reason="no_quad",
        )

    height, width = rgb.shape[:2]
    area_ratio = _quad_area(quad) / float(max(width * height, 1))
    if area_ratio < cfg.min_quad_area_ratio:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=float(area_ratio),
            reason="quad_too_small",
            source_quad=_as_tuple_quad(quad),
        )
    if area_ratio > cfg.max_quad_area_ratio:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=float(area_ratio),
            reason="quad_fills_frame",
            source_quad=_as_tuple_quad(quad),
        )
    if _is_nearly_axis_aligned(quad, width, height):
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=float(area_ratio),
            reason="already_frontal",
            source_quad=_as_tuple_quad(quad),
        )
    return PerspectiveResult(
        applied=False,
        quad_area_ratio=float(area_ratio),
        reason="detected",
        source_quad=_as_tuple_quad(quad),
    )


def warp_perspective(
    image: Image.Image,
    source_quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> Image.Image:
    """Warp ``source_quad`` (TL, TR, BR, BL) to a frontal RGB rectangle."""
    rgb = np.asarray(image.convert("RGB"))
    src = np.asarray(source_quad, dtype=np.float32)
    width_top = float(np.linalg.norm(src[1] - src[0]))
    width_bottom = float(np.linalg.norm(src[2] - src[3]))
    height_left = float(np.linalg.norm(src[3] - src[0]))
    height_right = float(np.linalg.norm(src[2] - src[1]))
    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)
    dst = np.asarray(
        [
            [0.0, 0.0],
            [out_w - 1.0, 0.0],
            [out_w - 1.0, out_h - 1.0],
            [0.0, out_h - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgb,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(cast(NDArray[np.uint8], warped), mode="RGB")


def apply_perspective(
    page: Page,
    result: PerspectiveResult | None = None,
    config: PerspectiveConfig | None = None,
) -> PerspectiveResult:
    """Warp the page when a usable document quad is found."""
    cfg = config if config is not None else PerspectiveConfig()
    if not cfg.enabled:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=0.0,
            reason="disabled",
        )

    detected = result if result is not None else detect_perspective(page.image, cfg)
    if detected.reason != "detected" or detected.source_quad is None:
        return detected
    if not page.has_image:
        return PerspectiveResult(
            applied=False,
            quad_area_ratio=detected.quad_area_ratio,
            reason="no_image",
            source_quad=detected.source_quad,
        )

    warped = warp_perspective(page.image, detected.source_quad)
    page.bind_image(warped)
    page.width = warped.size[0]
    page.height = warped.size[1]
    return PerspectiveResult(
        applied=True,
        quad_area_ratio=detected.quad_area_ratio,
        reason="applied",
        source_quad=detected.source_quad,
    )


def _as_tuple_quad(
    quad: NDArray[np.float32],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    points = [(float(x), float(y)) for x, y in quad]
    return (points[0], points[1], points[2], points[3])


def _quad_area(quad: NDArray[np.float32]) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)))


def _order_quad(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order four points as TL, TR, BR, BL."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    # np.diff on (x, y) yields y - x: TR has small y-x, BL has large y-x.
    diffs = np.diff(pts, axis=1).reshape(4)
    ordered[0] = pts[int(np.argmin(sums))]  # TL
    ordered[2] = pts[int(np.argmax(sums))]  # BR
    ordered[1] = pts[int(np.argmin(diffs))]  # TR
    ordered[3] = pts[int(np.argmax(diffs))]  # BL
    return ordered


def _is_nearly_axis_aligned(
    quad: NDArray[np.float32],
    width: int,
    height: int,
) -> bool:
    """True when the quad is already a near-axis-aligned rectangle."""
    del width, height  # Signature kept for call-site clarity.
    # Edge directions must be within a few degrees of horizontal/vertical.
    for index in range(4):
        x0, y0 = quad[index]
        x1, y1 = quad[(index + 1) % 4]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 90.0
        if min(angle, 90.0 - angle) > 4.0:
            return False
    xs = quad[:, 0]
    ys = quad[:, 1]
    bbox_area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    if bbox_area <= 1.0:
        return True
    fill = _quad_area(quad) / bbox_area
    return fill >= 0.92


def _find_document_quad(rgb: NDArray[np.uint8]) -> NDArray[np.float32] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge map favours the paper boundary against a darker desk/shadow.
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # Fallback: threshold on brightness (paper vs dark surround).
        _thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    if not contours:
        return None

    height, width = gray.shape[:2]
    page_area = float(width * height)
    best: NDArray[np.float32] | None = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.1 * page_area or area > 0.99 * page_area:
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri <= 0.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area <= best_area:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        best = _order_quad(quad)
        best_area = area

    if best is None:
        # Last resort: largest contour's min-area rect as a quad.
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < 0.15 * page_area:
            return None
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        return _order_quad(box)
    return best
