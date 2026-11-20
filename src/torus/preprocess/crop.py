"""Border and shadow margin cropping (P2.3).

Finds the content bbox (ink and/or paper against a dark surround) and trims
empty or dark margins, keeping a configured pad. Near-blank pages are not
cropped away: sparse content skips the crop and the full page is kept.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import CropConfig
from torus.core.models import Page


class CropResult(BaseModel):
    """Outcome of content-bound cropping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in pixels
    content_ratio: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def detect_content_box(
    image: Image.Image,
    config: CropConfig | None = None,
) -> CropResult:
    """Return the padded content bbox without mutating the image."""
    cfg = config if config is not None else CropConfig()
    if not cfg.enabled:
        return CropResult(applied=False, content_ratio=0.0, reason="disabled")

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return CropResult(applied=False, content_ratio=0.0, reason="too_small")

    content = _content_mask(rgb)
    content_ratio = float(np.count_nonzero(content)) / float(max(width * height, 1))
    if content_ratio < cfg.min_content_ratio:
        return CropResult(
            applied=False,
            content_ratio=content_ratio,
            reason="sparse_content",
        )

    ys, xs = np.where(content > 0)
    if ys.size == 0:
        return CropResult(
            applied=False,
            content_ratio=0.0,
            reason="no_content",
        )

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1

    max_trim_x = int(cfg.max_trim_ratio * width)
    max_trim_y = int(cfg.max_trim_ratio * height)
    x0 = max(0, min(x0, max_trim_x))
    y0 = max(0, min(y0, max_trim_y))
    x1 = min(width, max(x1, width - max_trim_x))
    y1 = min(height, max(y1, height - max_trim_y))

    pad = cfg.margin_px
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)

    if x1 - x0 < 4 or y1 - y0 < 4:
        return CropResult(
            applied=False,
            content_ratio=content_ratio,
            reason="degenerate_box",
        )
    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        return CropResult(
            applied=False,
            box=(x0, y0, x1, y1),
            content_ratio=content_ratio,
            reason="no_margin",
        )
    return CropResult(
        applied=False,
        box=(x0, y0, x1, y1),
        content_ratio=content_ratio,
        reason="detected",
    )


def crop_image(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Crop RGB image to inclusive-exclusive ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = box
    return image.convert("RGB").crop((x0, y0, x1, y1))


def apply_crop(
    page: Page,
    result: CropResult | None = None,
    config: CropConfig | None = None,
) -> CropResult:
    """Crop border/shadow margins when a safe content box is found."""
    cfg = config if config is not None else CropConfig()
    if not cfg.enabled:
        return CropResult(applied=False, content_ratio=0.0, reason="disabled")

    detected = result if result is not None else detect_content_box(page.image, cfg)
    if detected.reason != "detected" or detected.box is None:
        return detected
    if not page.has_image:
        return CropResult(
            applied=False,
            box=detected.box,
            content_ratio=detected.content_ratio,
            reason="no_image",
        )

    x0, y0, x1, y1 = detected.box
    if x0 == 0 and y0 == 0 and x1 == page.width and y1 == page.height:
        return CropResult(
            applied=False,
            box=detected.box,
            content_ratio=detected.content_ratio,
            reason="no_margin",
        )

    cropped = crop_image(page.image, detected.box)
    page.bind_image(cropped)
    page.width = cropped.size[0]
    page.height = cropped.size[1]
    return CropResult(
        applied=True,
        box=detected.box,
        content_ratio=detected.content_ratio,
        reason="applied",
    )


def _ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    inverted = np.subtract(255, gray, dtype=np.uint8)
    if int(inverted.max()) - int(inverted.min()) < 8:
        return np.zeros_like(gray)
    _thr, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cast(NDArray[np.uint8], mask)


def _has_dark_frame(gray: NDArray[np.uint8]) -> bool:
    """True when outer strips are much darker than the page centre (photo)."""
    h, w = gray.shape[:2]
    border = max(min(h, w) // 20, 4)
    outer = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    inner = gray[border : h - border, border : w - border]
    if inner.size == 0:
        return False
    return float(outer.mean()) + 25.0 < float(inner.mean())


def _content_mask(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binary mask used to bound crop: paper against dark frame, else ink."""
    gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if _has_dark_frame(gray):
        # Keep the bright paper region; drop the dark desk/shadow surround.
        _thr, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cast(NDArray[np.uint8], bright)
    else:
        # Flat scan: trim to ink so empty white margins shrink.
        mask = _ink_mask(gray)

    if int(np.count_nonzero(mask)) < 50:
        # Fallback: anything not near-pure-white.
        mask = ((gray < 245).astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cast(
        NDArray[np.uint8],
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel),
    )
    return closed
