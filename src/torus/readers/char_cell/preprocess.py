"""Crop a comb cell to a fixed-size ink vector for the classifier.

No document knowledge — only image geometry and ink statistics.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from torus.core.models import Region
from torus.readers.crop_hash import pixel_box

GrayF32 = NDArray[np.float32]
VecF32 = NDArray[np.float32]

DEFAULT_INPUT_SIZE = 28
# Fraction of pixels that must be ink before we treat the cell as non-blank.
DEFAULT_INK_FRACTION_MIN = 0.01


def crop_region(image: Any, region: Region) -> Image.Image:
    """Crop ``image`` (PIL-like) to ``region.bbox`` as RGB."""
    size = getattr(image, "size", None)
    crop = getattr(image, "crop", None)
    if size is None or not callable(crop):
        msg = "char-cell reader needs a PIL-like image (size + crop)"
        raise TypeError(msg)
    page_w, page_h = int(size[0]), int(size[1])
    if page_w <= 0 or page_h <= 0:
        msg = "image has no pixels"
        raise ValueError(msg)
    x0, y0, x1, y1 = pixel_box((page_w, page_h), region.bbox)
    return cast(Image.Image, crop((x0, y0, x1, y1)).convert("RGB"))


def to_ink_array(image: Image.Image) -> GrayF32:
    """Return ink intensity in [0, 1] (1 = dark stroke on light paper)."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    # Forms are light paper; invert so ink is high.
    return cast(GrayF32, 1.0 - gray)


def center_and_resize(
    ink: GrayF32,
    *,
    size: int = DEFAULT_INPUT_SIZE,
) -> GrayF32:
    """Crop to ink bounding box (with padding), then resize to ``size``×``size``."""
    if ink.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    mask = ink > 0.15
    if not bool(np.any(mask)):
        return np.zeros((size, size), dtype=np.float32)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    pad = 2
    r0 = max(0, rmin - pad)
    r1 = min(ink.shape[0], rmax + pad)
    c0 = max(0, cmin - pad)
    c1 = min(ink.shape[1], cmax + pad)
    cropped = ink[r0:r1, c0:c1]
    if cropped.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    # Pillow resize keeps deps limited to what the core already uses.
    pil = Image.fromarray(
        (np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    resized = pil.resize((size, size), resample=Image.Resampling.BILINEAR)
    out = np.asarray(resized, dtype=np.float32) / 255.0
    return cast(GrayF32, out)


def feature_vector(
    image: Image.Image,
    *,
    size: int = DEFAULT_INPUT_SIZE,
) -> tuple[VecF32, float]:
    """Return (flattened size² vector, ink_fraction) for one cell crop."""
    ink = to_ink_array(image)
    ink_fraction = float(np.mean(ink > 0.15))
    grid = center_and_resize(ink, size=size)
    # Zero-mean unit-ish scale helps logistic / MLP training.
    mean = float(grid.mean())
    std = float(grid.std())
    normalised = grid - mean if std < 1e-6 else (grid - mean) / std
    return cast(VecF32, normalised.reshape(-1).astype(np.float32)), ink_fraction


def is_blank(
    ink_fraction: float, *, min_fraction: float = DEFAULT_INK_FRACTION_MIN
) -> bool:
    """True when the cell has too little ink to classify."""
    return ink_fraction < min_fraction
