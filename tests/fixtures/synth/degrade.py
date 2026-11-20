"""Deterministic image degradation (fax-like scan artefacts)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def fax_degrade(
    image: Image.Image, *, seed: int, strength: float = 0.65
) -> Image.Image:
    """Apply fax/scan-like noise, blur and contrast crush.

    ``strength`` in ``[0, 1]`` controls how aggressive the degradation is.
    Same ``(seed, strength, image)`` → same output.
    """
    if not 0.0 <= strength <= 1.0:
        msg = f"strength must be in [0, 1], got {strength}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    rgb = image.convert("RGB")
    # Mild blur mimics low-resolution fax optics.
    blur_radius = 0.4 + 1.6 * strength
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(blurred, dtype=np.float32)
    # Additive grain.
    noise_sigma = 8.0 + 28.0 * strength
    noise = rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
    arr = arr + noise
    # Crush mid-tones toward black/white (fax threshold feel).
    mid = 128.0
    contrast = 1.0 + 1.4 * strength
    arr = (arr - mid) * contrast + mid
    # Occasional horizontal streak (scan line).
    if strength > 0.2 and arr.shape[0] > 8:
        streak_y = int(rng.integers(0, arr.shape[0]))
        streak_h = max(1, int(1 + 2 * strength))
        y1 = min(arr.shape[0], streak_y + streak_h)
        streak = 40.0 + 60.0 * float(rng.random())
        arr[streak_y:y1, :, :] = streak

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def fade_degrade(
    image: Image.Image, *, seed: int, strength: float = 0.65
) -> Image.Image:
    """Pull ink and paper toward mid-grey (faded photocopy / grey scan).

    ``strength`` in ``[0, 1]`` controls how far tones collapse toward grey.
    Same ``(seed, strength, image)`` → same output.
    """
    if not 0.0 <= strength <= 1.0:
        msg = f"strength must be in [0, 1], got {strength}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mid = 140.0
    # Blend toward mid-grey; higher strength → flatter, lower contrast.
    blend = 0.35 + 0.55 * strength
    arr = arr * (1.0 - blend) + mid * blend
    # Light grain so the page is not perfectly flat (CLAHE still has work).
    noise_sigma = 2.0 + 6.0 * strength
    noise = rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
