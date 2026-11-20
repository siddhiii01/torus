"""Page quality score from blur, contrast and resolution (P2.6).

The combined score lands on ``Page.quality_score`` in ``[0, 1]`` and drives
``Page.low_quality`` when below the configured threshold. Later stages
(confidence calibration, reader routing) consume these fields; this module
only measures and records — it does not alter pixels.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import PageQualityConfig
from torus.core.models import Page


class PageQualityResult(BaseModel):
    """Measured quality components and the combined score for one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    blur_score: float | None = Field(default=None, ge=0.0, le=1.0)
    contrast_score: float | None = Field(default=None, ge=0.0, le=1.0)
    resolution_score: float | None = Field(default=None, ge=0.0, le=1.0)
    blur_laplacian: float | None = Field(default=None, ge=0.0)
    contrast: float | None = Field(default=None, ge=0.0)
    min_side_px: int | None = Field(default=None, ge=0)
    low_quality: bool = False
    reason: str = Field(min_length=1)


def measure_blur(gray: NDArray[np.uint8]) -> float:
    """Structured sharpness: Laplacian variance after mild denoise.

    A short Gaussian pass suppresses scan grain so fax noise does not look
    sharper than real ink edges.
    """
    if gray.size == 0:
        return 0.0
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(smoothed, cv2.CV_64F)
    return float(lap.var())


def measure_page_contrast(gray: NDArray[np.uint8]) -> float:
    """Grey-channel standard deviation (works with sparse ink on white)."""
    if gray.size == 0:
        return 0.0
    return float(np.std(gray.astype(np.float64)))


def measure_resolution(width: int, height: int) -> int:
    """Short-side length in pixels (proxy for usable render resolution)."""
    if width <= 0 or height <= 0:
        return 0
    return int(min(width, height))


def score_components(
    image: Image.Image,
    config: PageQualityConfig | None = None,
) -> PageQualityResult:
    """Compute blur / contrast / resolution components and a weighted score."""
    cfg = config if config is not None else PageQualityConfig()
    if not cfg.enabled:
        return PageQualityResult(applied=False, reason="disabled")

    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        return PageQualityResult(
            applied=True,
            score=0.0,
            blur_score=0.0,
            contrast_score=0.0,
            resolution_score=0.0,
            blur_laplacian=0.0,
            contrast=0.0,
            min_side_px=measure_resolution(width, height),
            low_quality=True,
            reason="too_small",
        )

    gray = _to_gray(np.asarray(rgb, dtype=np.uint8))
    blur_raw = measure_blur(gray)
    contrast_raw = measure_page_contrast(gray)
    min_side = measure_resolution(width, height)

    blur_score = _saturate(blur_raw / cfg.blur_laplacian_ref)
    contrast_score = _saturate(contrast_raw / cfg.contrast_ref)
    resolution_score = _saturate(float(min_side) / float(cfg.target_min_side_px))

    score = _weighted_mean(
        (
            (blur_score, cfg.weight_blur),
            (contrast_score, cfg.weight_contrast),
            (resolution_score, cfg.weight_resolution),
        )
    )
    low_quality = score < cfg.low_quality_threshold
    return PageQualityResult(
        applied=True,
        score=score,
        blur_score=blur_score,
        contrast_score=contrast_score,
        resolution_score=resolution_score,
        blur_laplacian=blur_raw,
        contrast=contrast_raw,
        min_side_px=min_side,
        low_quality=low_quality,
        reason="scored",
    )


def apply_page_quality(
    page: Page,
    config: PageQualityConfig | None = None,
) -> PageQualityResult:
    """Score the page image and store ``quality_score`` / ``low_quality`` on it."""
    cfg = config if config is not None else PageQualityConfig()
    if not cfg.enabled:
        return PageQualityResult(applied=False, reason="disabled")
    if not page.has_image:
        return PageQualityResult(applied=False, reason="no_image")

    result = score_components(page.image, cfg)
    if result.score is not None:
        page.quality_score = result.score
        page.low_quality = result.low_quality
    return result


def _saturate(ratio: float) -> float:
    if ratio <= 0.0:
        return 0.0
    if ratio >= 1.0:
        return 1.0
    return float(ratio)


def _weighted_mean(parts: tuple[tuple[float, float], ...]) -> float:
    total_w = 0.0
    total = 0.0
    for value, weight in parts:
        if weight <= 0.0:
            continue
        total += value * weight
        total_w += weight
    if total_w <= 0.0:
        return 0.0
    return float(total / total_w)


def _to_gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if rgb.ndim == 2:
        return rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cast(NDArray[np.uint8], gray)
