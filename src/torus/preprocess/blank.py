"""Blank and near-blank page detection (P2.5).

Two independent signals must agree before a page is marked blank:

1. **Ink density** — fraction of Otsu ink pixels at or below ``max_ink_density``.
2. **OCR / text-presence check** — classical connected-component scan for
   glyph-sized blobs (no model import). Fewer than ``min_text_components``
   means "no readable text".

Blank pages are **recorded** on ``Page.blank``. They are never removed from
the document page list.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import BlankDetectConfig
from torus.core.models import Document, Page
from torus.preprocess.quality_repair import measure_ink_density


class BlankDetectResult(BaseModel):
    """Outcome of blank / near-blank detection for one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blank: bool
    near_blank: bool
    ink_density: float = Field(ge=0.0, le=1.0)
    text_component_count: int = Field(ge=0)
    reason: str = Field(min_length=1)


def detect_blank(
    image: Image.Image,
    config: BlankDetectConfig | None = None,
) -> BlankDetectResult:
    """Classify a page image as blank / near-blank / has content."""
    cfg = config if config is not None else BlankDetectConfig()
    if not cfg.enabled:
        return BlankDetectResult(
            blank=False,
            near_blank=False,
            ink_density=0.0,
            text_component_count=0,
            reason="disabled",
        )

    gray = _to_gray(image)
    ink_density = measure_ink_density(gray)
    text_count = count_text_like_components(gray, cfg)

    low_ink = ink_density <= cfg.max_ink_density
    no_text = text_count < cfg.min_text_components
    is_blank = low_ink and no_text

    if not is_blank:
        if not low_ink and not no_text:
            reason = "has_content"
        elif not low_ink:
            reason = "ink_above_threshold"
        else:
            reason = "text_components_present"
        return BlankDetectResult(
            blank=False,
            near_blank=False,
            ink_density=ink_density,
            text_component_count=text_count,
            reason=reason,
        )

    # Distinguish pure blank (no measurable ink) from near-blank speckles.
    near_blank = ink_density > 0.0
    return BlankDetectResult(
        blank=True,
        near_blank=near_blank,
        ink_density=ink_density,
        text_component_count=text_count,
        reason="near_blank" if near_blank else "blank",
    )


def apply_blank_detect(
    page: Page,
    config: BlankDetectConfig | None = None,
) -> BlankDetectResult:
    """Set ``page.blank`` from detection. Never drops or replaces the page."""
    cfg = config if config is not None else BlankDetectConfig()
    if not cfg.enabled:
        return BlankDetectResult(
            blank=False,
            near_blank=False,
            ink_density=0.0,
            text_component_count=0,
            reason="disabled",
        )
    if not page.has_image:
        return BlankDetectResult(
            blank=False,
            near_blank=False,
            ink_density=0.0,
            text_component_count=0,
            reason="no_image",
        )

    result = detect_blank(page.image, cfg)
    page.blank = result.blank
    return result


def flag_blank_pages(
    document: Document,
    config: BlankDetectConfig | None = None,
) -> tuple[BlankDetectResult, ...]:
    """Flag blank pages in place. Page count is unchanged (never dropped)."""
    cfg = config if config is not None else BlankDetectConfig()
    before = len(document.pages)
    results = tuple(apply_blank_detect(page, cfg) for page in document.pages)
    after = len(document.pages)
    if after != before:
        msg = f"blank detect must not drop pages: before={before} after={after}"
        raise RuntimeError(msg)
    return results


def count_text_like_components(
    gray: NDArray[np.uint8],
    config: BlankDetectConfig,
) -> int:
    """Count glyph-sized connected components (classical OCR-presence proxy)."""
    mask = _ink_mask(gray)
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return 0

    height, width = mask.shape[:2]
    page_area = float(max(height * width, 1))
    max_area = max(
        int(config.max_component_area_ratio * page_area),
        config.min_component_area,
    )
    min_h = max(1, int(config.min_component_height_ratio * height))
    max_h = max(min_h, int(config.max_component_height_ratio * height))

    # Light open removes single-pixel scan noise before labelling.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cast(NDArray[np.uint8], cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel))

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    count = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_component_area or area > max_area:
            continue
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if comp_h < min_h or comp_h > max_h:
            continue
        short = float(min(comp_w, comp_h))
        long = float(max(comp_w, comp_h))
        aspect = long / max(short, 1.0)
        if aspect > config.max_component_aspect:
            continue
        count += 1
    return count


def _to_gray(image: Image.Image) -> NDArray[np.uint8]:
    return np.asarray(image.convert("L"), dtype=np.uint8)


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
