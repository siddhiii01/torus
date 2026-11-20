"""Image + text feature vectors for the trained page classifier.

Uses OpenCV / NumPy only (base install). Feature names are stable so saved
weights stay compatible across runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from torus.classify.text_cues import text_cue_scores
from torus.config.schemas import PageTypesConfig, PageTypeSpec
from torus.core.models import Page

# Fixed visual block (order matters for weight matrices).
VISUAL_FEATURE_NAMES: tuple[str, ...] = (
    "ink_density",
    "edge_density",
    "hline_score",
    "large_blob_ratio",
    "text_component_density",
    "blank_flag",
)


def feature_names(page_type_ids: Sequence[str]) -> list[str]:
    """Full feature name list: visuals then one text-cue score per type id."""
    names = list(VISUAL_FEATURE_NAMES)
    names.extend(f"cue:{type_id}" for type_id in page_type_ids)
    return names


def extract_visual_features(
    image: Image.Image,
    *,
    blank_flag: bool = False,
) -> NDArray[np.float64]:
    """Six-dimensional visual descriptor for one page image."""
    gray = _to_gray(image)
    h, w = gray.shape
    area = float(max(h * w, 1))

    ink = _ink_density(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / area

    # Horizontal line energy: tables / form grids raise this.
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_x = float(np.mean(np.abs(sobel_x)))
    mag_y = float(np.mean(np.abs(sobel_y)))
    hline_score = mag_x / (mag_x + mag_y + 1e-6)

    # Large dark blobs ≈ photo / stamp regions.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    large_area = 0
    text_like = 0
    min_text = max(8, int(0.00005 * area))
    max_text = int(0.02 * area)
    large_min = int(0.05 * area)
    for i in range(1, num_labels):
        comp_area = int(stats[i, cv2.CC_STAT_AREA])
        if comp_area >= large_min:
            large_area += comp_area
        if min_text <= comp_area <= max_text:
            text_like += 1
    large_blob_ratio = float(large_area) / area
    text_component_density = min(1.0, float(text_like) / 200.0)

    vec = np.array(
        [
            ink,
            edge_density,
            hline_score,
            large_blob_ratio,
            text_component_density,
            1.0 if blank_flag else 0.0,
        ],
        dtype=np.float64,
    )
    return vec


def extract_features(
    page: Page,
    page_types: PageTypesConfig | Sequence[PageTypeSpec],
    page_text: str = "",
) -> NDArray[np.float64]:
    """Concatenate visual features and per-type text-cue scores."""
    if isinstance(page_types, PageTypesConfig):
        specs = page_types.page_types
        type_ids = [s.id for s in specs]
    else:
        specs = list(page_types)
        type_ids = [s.id for s in specs]

    if page.has_image:
        image = cast(Image.Image, page.image)
        visual = extract_visual_features(image, blank_flag=page.blank)
    else:
        visual = np.zeros(len(VISUAL_FEATURE_NAMES), dtype=np.float64)
        visual[-1] = 1.0 if page.blank else 0.0

    cues = text_cue_scores(page_text, specs)
    cue_vec = np.array([cues.get(tid, 0.0) for tid in type_ids], dtype=np.float64)
    return np.concatenate([visual, cue_vec])


def _to_gray(image: Image.Image) -> NDArray[np.uint8]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cast(NDArray[np.uint8], gray)


def _ink_density(gray: NDArray[np.uint8]) -> float:
    """Fraction of Otsu ink pixels (local copy; classify must not import preprocess)."""
    if gray.size == 0:
        return 0.0
    _threshold, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return float(np.count_nonzero(mask)) / float(mask.size)
