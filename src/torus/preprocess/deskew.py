"""Small-angle page deskew (P2.3).

Estimates skew from the dominant ink-edge direction (min-area rect of ink
pixels, with a Hough-line fallback). Only angles inside the configured band
are applied so 90° orientation mistakes are left to the orientation stage.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import DeskewConfig
from torus.core.models import Page


class DeskewResult(BaseModel):
    """Outcome of skew detection / optional correction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    angle_deg: float
    applied: bool
    reason: str = Field(min_length=1)


def detect_skew(
    image: Image.Image,
    config: DeskewConfig | None = None,
) -> DeskewResult:
    """Estimate clockwise skew angle in degrees (positive = clock-wise)."""
    cfg = config if config is not None else DeskewConfig()
    if not cfg.enabled:
        return DeskewResult(angle_deg=0.0, applied=False, reason="disabled")

    gray = _to_gray(image)
    angle = _estimate_skew_angle(gray)
    abs_angle = abs(angle)
    if abs_angle < cfg.min_angle_deg:
        return DeskewResult(
            angle_deg=float(angle),
            applied=False,
            reason="below_min_angle",
        )
    if abs_angle > cfg.max_angle_deg:
        return DeskewResult(
            angle_deg=float(angle),
            applied=False,
            reason="above_max_angle",
        )
    return DeskewResult(angle_deg=float(angle), applied=False, reason="detected")


def deskew_image(image: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate so a clockwise skew of ``angle_deg`` becomes upright."""
    if abs(angle_deg) < 1e-6:
        return image.copy()
    # Content is rotated CW by angle_deg; Pillow rotate is CCW.
    return image.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255)).convert(
        "RGB"
    )


def apply_deskew(
    page: Page,
    result: DeskewResult | None = None,
    config: DeskewConfig | None = None,
) -> DeskewResult:
    """Deskew the page image when the detected angle is in range."""
    cfg = config if config is not None else DeskewConfig()
    if not cfg.enabled:
        return DeskewResult(angle_deg=0.0, applied=False, reason="disabled")

    detected = result if result is not None else detect_skew(page.image, cfg)
    abs_angle = abs(detected.angle_deg)
    if abs_angle < cfg.min_angle_deg:
        return DeskewResult(
            angle_deg=detected.angle_deg,
            applied=False,
            reason="below_min_angle",
        )
    if abs_angle > cfg.max_angle_deg:
        return DeskewResult(
            angle_deg=detected.angle_deg,
            applied=False,
            reason="above_max_angle",
        )
    if not page.has_image:
        return DeskewResult(
            angle_deg=detected.angle_deg,
            applied=False,
            reason="no_image",
        )

    corrected = deskew_image(page.image, detected.angle_deg)
    page.bind_image(corrected)
    page.width = corrected.size[0]
    page.height = corrected.size[1]
    return DeskewResult(
        angle_deg=detected.angle_deg,
        applied=True,
        reason="applied",
    )


def _to_gray(image: Image.Image) -> NDArray[np.uint8]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cast(NDArray[np.uint8], gray)


def _ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if gray.size == 0:
        return np.zeros_like(gray)
    inverted = np.subtract(255, gray, dtype=np.uint8)
    if int(inverted.max()) - int(inverted.min()) < 8:
        return np.zeros_like(gray)
    _threshold, mask = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cast(NDArray[np.uint8], mask)


def _normalize_skew_angle(angle_deg: float) -> float:
    """Map OpenCV / Hough angles into a small deskew band around 0."""
    angle = float(angle_deg) % 180.0
    if angle > 90.0:
        angle -= 180.0
    # Prefer the acute representation so ±45 stays within max_angle for scans.
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return float(angle)


def _estimate_skew_angle(gray: NDArray[np.uint8]) -> float:
    mask = _ink_mask(gray)
    ink_count = int(np.count_nonzero(mask))
    if ink_count < 80:
        return 0.0

    # Projection profile is the primary estimator: upright text maximises
    # row-projection variance. Rect/Hough alone false-fire on sparse forms.
    profile_angle = _angle_from_projection(mask)
    hough_angle = _angle_from_hough(gray, mask)
    if hough_angle is not None and abs(hough_angle - profile_angle) <= 2.5:
        return _normalize_skew_angle(0.5 * (profile_angle + hough_angle))
    return _normalize_skew_angle(profile_angle)


def _angle_from_projection(
    mask: NDArray[np.uint8],
    *,
    search_deg: float = 12.0,
    step_deg: float = 0.25,
) -> float:
    """Search small angles for the row-projection variance peak."""
    ink = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(ink)) < 80:
        return 0.0

    best_angle = 0.0
    best_score = -1.0
    angles = np.arange(-search_deg, search_deg + step_deg * 0.5, step_deg)
    height, width = ink.shape[:2]
    center = (width / 2.0, height / 2.0)
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            ink,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_proj = (rotated > 0).sum(axis=1).astype(np.float64)
        score = float(np.var(row_proj))
        if score > best_score:
            best_score = score
            # getRotationMatrix2D angle is CCW; the CCW rotate that uprights
            # the page equals the clockwise content skew we report.
            best_angle = float(angle)
    return best_angle


def _angle_from_min_area_rect(mask: NDArray[np.uint8]) -> float | None:
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 40:
        return None
    _center, _size, raw_angle = cv2.minAreaRect(coords)
    # OpenCV: angle of the box's longer side relative to horizontal, in [-90, 0).
    width, height = float(_size[0]), float(_size[1])
    angle = float(raw_angle)
    if width < height:
        angle = angle + 90.0
    return _normalize_skew_angle(angle)


def _angle_from_hough(
    gray: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> float | None:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, mask)
    height, width = gray.shape[:2]
    min_len = max(int(0.2 * max(width, height)), 40)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=min_len,
        maxLineGap=20,
    )
    if lines is None:
        return None

    angles: list[float] = []
    weights: list[float] = []
    flat = lines.reshape(-1, 4)
    for x1_f, y1_f, x2_f, y2_f in flat:
        x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        # Angle of the line relative to horizontal, degrees.
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Keep near-horizontal lines (text baselines / form rules).
        norm = _normalize_skew_angle(angle)
        if abs(norm) > 30.0:
            continue
        angles.append(norm)
        weights.append(length)

    if not angles:
        return None
    return float(np.average(np.asarray(angles), weights=np.asarray(weights)))
