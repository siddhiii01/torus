"""Image preprocess variants for OCR bake-offs (page reading only)."""
from __future__ import annotations

import numpy as np


def clahe_rgb(img: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on luminance."""
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge((l2, a, b))
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def upscale(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    import cv2

    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
