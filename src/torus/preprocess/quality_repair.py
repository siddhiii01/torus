"""Fax-like and faded page quality repair (P2.4).

Pipeline order is fixed and each step is independently switchable in config:

1. ``denoise`` — coloured non-local means (remove fax grain before thickening)
2. ``contrast_norm`` — CLAHE on the L channel (recover faded / grey scans)
3. ``stroke_thicken`` — elliptical dilation of dark strokes

Every step records before/after metrics so the effect is measurable without
logging document text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import QualityRepairConfig
from torus.core.models import Page

_STEP_DENOISE = "denoise"
_STEP_CONTRAST = "contrast_norm"
_STEP_THICKEN = "stroke_thicken"


class StepMetrics(BaseModel):
    """Measured effect of one repair step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    applied: bool
    elapsed_ms: float = Field(ge=0.0)
    contrast_before: float = Field(ge=0.0)
    contrast_after: float = Field(ge=0.0)
    ink_density_before: float = Field(ge=0.0, le=1.0)
    ink_density_after: float = Field(ge=0.0, le=1.0)
    noise_before: float = Field(ge=0.0)
    noise_after: float = Field(ge=0.0)
    stroke_width_before: float = Field(ge=0.0)
    stroke_width_after: float = Field(ge=0.0)


class QualityRepairResult(BaseModel):
    """Outcome of the quality-repair pipeline on one page image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    image_changed: bool
    steps: tuple[StepMetrics, ...] = ()
    reason: str = Field(min_length=1)


def repair_image(
    image: Image.Image,
    config: QualityRepairConfig | None = None,
) -> tuple[Image.Image, QualityRepairResult]:
    """Run configured repair steps; return repaired RGB image plus metrics."""
    cfg = config if config is not None else QualityRepairConfig()
    rgb = image.convert("RGB")
    if not cfg.enabled:
        return rgb.copy(), QualityRepairResult(
            applied=False,
            image_changed=False,
            steps=(),
            reason="disabled",
        )

    current = np.asarray(rgb, dtype=np.uint8).copy()
    steps: list[StepMetrics] = []

    if cfg.denoise.enabled:
        current, step = _run_step(
            _STEP_DENOISE,
            current,
            lambda arr: _denoise(arr, cfg),
        )
        steps.append(step)
    else:
        steps.append(_skipped_step(_STEP_DENOISE, current))

    if cfg.contrast_norm.enabled:
        current, step = _run_step(
            _STEP_CONTRAST,
            current,
            lambda arr: _contrast_norm(arr, cfg),
        )
        steps.append(step)
    else:
        steps.append(_skipped_step(_STEP_CONTRAST, current))

    if cfg.stroke_thicken.enabled:
        current, step = _run_step(
            _STEP_THICKEN,
            current,
            lambda arr: _stroke_thicken(arr, cfg),
        )
        steps.append(step)
    else:
        steps.append(_skipped_step(_STEP_THICKEN, current))

    any_applied = any(s.applied for s in steps)
    out = Image.fromarray(current, mode="RGB")
    changed = any_applied and not _arrays_equal(np.asarray(rgb), current)
    return out, QualityRepairResult(
        applied=any_applied,
        image_changed=changed,
        steps=tuple(steps),
        reason="applied" if any_applied else "all_steps_disabled",
    )


def apply_quality_repair(
    page: Page,
    config: QualityRepairConfig | None = None,
) -> QualityRepairResult:
    """Repair the page image in place when quality repair is enabled."""
    cfg = config if config is not None else QualityRepairConfig()
    if not cfg.enabled:
        return QualityRepairResult(
            applied=False,
            image_changed=False,
            steps=(),
            reason="disabled",
        )
    if not page.has_image:
        return QualityRepairResult(
            applied=False,
            image_changed=False,
            steps=(),
            reason="no_image",
        )

    repaired, result = repair_image(page.image, cfg)
    if result.image_changed:
        page.bind_image(repaired)
        page.width = repaired.size[0]
        page.height = repaired.size[1]
    return result


def measure_contrast(gray: NDArray[np.uint8]) -> float:
    """Robust contrast: 5th–95th percentile span normalised to ``[0, 1]``."""
    if gray.size == 0:
        return 0.0
    lo = float(np.percentile(gray, 5))
    hi = float(np.percentile(gray, 95))
    return max(0.0, (hi - lo) / 255.0)


def measure_ink_density(gray: NDArray[np.uint8]) -> float:
    """Fraction of pixels classified as ink (Otsu on inverted grey)."""
    mask = _ink_mask(gray)
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def measure_noise(gray: NDArray[np.uint8]) -> float:
    """High-frequency residual energy after a mild Gaussian blur."""
    if gray.size == 0:
        return 0.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(residual))


def measure_stroke_width(gray: NDArray[np.uint8]) -> float:
    """Mean distance-transform radius on ink (proxy for stroke thickness)."""
    mask = _ink_mask(gray)
    if mask.size == 0 or int(np.count_nonzero(mask)) < 20:
        return 0.0
    # Distance to nearest background; values on ink skeleton ≈ half-width.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ink_vals = dist[mask > 0]
    if ink_vals.size == 0:
        return 0.0
    return float(np.mean(ink_vals))


def _run_step(
    name: str,
    current: NDArray[np.uint8],
    transform: Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
) -> tuple[NDArray[np.uint8], StepMetrics]:
    gray_before = _to_gray_arr(current)
    before = _snapshot_metrics(gray_before)
    started = time.perf_counter()
    after_arr = transform(current)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    gray_after = _to_gray_arr(after_arr)
    after = _snapshot_metrics(gray_after)
    return after_arr, StepMetrics(
        name=name,
        applied=True,
        elapsed_ms=elapsed_ms,
        contrast_before=before["contrast"],
        contrast_after=after["contrast"],
        ink_density_before=before["ink_density"],
        ink_density_after=after["ink_density"],
        noise_before=before["noise"],
        noise_after=after["noise"],
        stroke_width_before=before["stroke_width"],
        stroke_width_after=after["stroke_width"],
    )


def _skipped_step(name: str, current: NDArray[np.uint8]) -> StepMetrics:
    gray = _to_gray_arr(current)
    snap = _snapshot_metrics(gray)
    return StepMetrics(
        name=name,
        applied=False,
        elapsed_ms=0.0,
        contrast_before=snap["contrast"],
        contrast_after=snap["contrast"],
        ink_density_before=snap["ink_density"],
        ink_density_after=snap["ink_density"],
        noise_before=snap["noise"],
        noise_after=snap["noise"],
        stroke_width_before=snap["stroke_width"],
        stroke_width_after=snap["stroke_width"],
    )


def _snapshot_metrics(gray: NDArray[np.uint8]) -> dict[str, float]:
    return {
        "contrast": measure_contrast(gray),
        "ink_density": measure_ink_density(gray),
        "noise": measure_noise(gray),
        "stroke_width": measure_stroke_width(gray),
    }


def _denoise(rgb: NDArray[np.uint8], cfg: QualityRepairConfig) -> NDArray[np.uint8]:
    denoise = cfg.denoise
    # OpenCV expects contiguous uint8 BGR for the coloured NL-means path.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        float(denoise.h),
        float(denoise.h),
        int(denoise.template_window_size),
        int(denoise.search_window_size),
    )
    out = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return cast(NDArray[np.uint8], out)


def _contrast_norm(
    rgb: NDArray[np.uint8], cfg: QualityRepairConfig
) -> NDArray[np.uint8]:
    contrast = cfg.contrast_norm
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(contrast.clip_limit),
        tileGridSize=(int(contrast.tile_grid_size), int(contrast.tile_grid_size)),
    )
    light_eq = clahe.apply(lightness)
    merged = cv2.merge((light_eq, a_ch, b_ch))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return cast(NDArray[np.uint8], out)


def _stroke_thicken(
    rgb: NDArray[np.uint8], cfg: QualityRepairConfig
) -> NDArray[np.uint8]:
    thicken = cfg.stroke_thicken
    k = int(thicken.kernel_size)
    # Dilate dark strokes in each channel so coloured ink thickens evenly.
    inverted = np.subtract(255, rgb, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    out = np.subtract(255, dilated, dtype=np.uint8)
    return cast(NDArray[np.uint8], out)


def _to_gray_arr(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if rgb.ndim == 2:
        return rgb
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


def _arrays_equal(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))
