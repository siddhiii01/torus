"""Page orientation detection (0/90/180/270).

Two independent estimators run on every page:

* **classic** — morphological horizontal structure + per-line ink asymmetry
* **classifier** — gradient energy, run-length stats, and band asymmetry

When they agree, the page image is rotated upright. When they disagree, the
pixels stay as ingested and both candidate orientations are kept so later
readers can try each view.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import OrientationConfig
from torus.core.models import Page
from torus.core.types import OrientationDegrees

_ORIENTATIONS: tuple[OrientationDegrees, ...] = (0, 90, 180, 270)


class OrientationResult(BaseModel):
    """Outcome of classic + classifier orientation detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classic: OrientationDegrees
    classifier: OrientationDegrees
    classic_score: float = Field(ge=0.0)
    classifier_score: float = Field(ge=0.0)
    agreed: bool
    candidates_for_reading: tuple[OrientationDegrees, ...] = Field(min_length=1)


def detect_orientation(
    image: Image.Image,
    config: OrientationConfig | None = None,
) -> OrientationResult:
    """Estimate page orientation with classic and classifier estimators."""
    cfg = config if config is not None else OrientationConfig()
    if not cfg.enabled:
        return OrientationResult(
            classic=0,
            classifier=0,
            classic_score=0.0,
            classifier_score=0.0,
            agreed=True,
            candidates_for_reading=(0,),
        )

    classic, classic_score = classic_orientation(image)
    classifier, classifier_score = classifier_orientation(image)
    agreed = classic == classifier
    return OrientationResult(
        classic=classic,
        classifier=classifier,
        classic_score=classic_score,
        classifier_score=classifier_score,
        agreed=agreed,
        candidates_for_reading=_candidates_for_reading(classic, classifier),
    )


def classic_orientation(image: Image.Image) -> tuple[OrientationDegrees, float]:
    """Classic detector: closed projection structure + vertical ink mass."""
    scores = {
        degrees: _classic_score_for_candidate(image, degrees)
        for degrees in _ORIENTATIONS
    }
    best = max(_ORIENTATIONS, key=lambda d: scores[d])
    return best, float(scores[best])


def classifier_orientation(image: Image.Image) -> tuple[OrientationDegrees, float]:
    """Small feature classifier over the four cardinal orientations."""
    scores = {
        degrees: _classifier_score_for_candidate(image, degrees)
        for degrees in _ORIENTATIONS
    }
    best = max(_ORIENTATIONS, key=lambda d: scores[d])
    return best, float(scores[best])


def upright_image(image: Image.Image, orientation: OrientationDegrees) -> Image.Image:
    """Undo a clockwise content rotation so the page is upright."""
    if orientation == 0:
        return image.copy()
    # Content was rotated CW by ``orientation``; Pillow rotates CCW.
    return image.rotate(orientation, expand=True, fillcolor=(255, 255, 255)).convert(
        "RGB"
    )


def apply_orientation(page: Page, result: OrientationResult) -> OrientationResult:
    """If detectors agree on a non-zero orientation, upright the page image.

    On disagreement the image is left unchanged so readers can try each
    candidate in ``result.candidates_for_reading``.
    """
    if not result.agreed:
        return result

    correction = result.classic
    if correction == 0:
        page.orientation = 0
        return result

    if not page.has_image:
        page.orientation = correction
        page.rotated = True
        return result

    corrected = upright_image(page.image, correction)
    page.bind_image(corrected)
    page.width = corrected.size[0]
    page.height = corrected.size[1]
    page.orientation = 0
    page.rotated = True
    return result


def _candidates_for_reading(
    classic: OrientationDegrees,
    classifier: OrientationDegrees,
) -> tuple[OrientationDegrees, ...]:
    if classic == classifier:
        return (classic,)
    # Stable order: classic first, then classifier.
    return (classic, classifier)


def _rotate_cw(image: Image.Image, degrees: OrientationDegrees) -> Image.Image:
    if degrees == 0:
        return image.copy()
    return image.rotate(-degrees, expand=True, fillcolor=(255, 255, 255)).convert("RGB")


def _to_gray(image: Image.Image) -> NDArray[np.uint8]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cast(NDArray[np.uint8], gray)


def _ink_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binary ink mask (255 = ink). Empty pages yield an all-zero mask."""
    if gray.size == 0:
        return np.zeros_like(gray)
    # Invert so ink is bright for Otsu on typical white pages.
    inverted = np.subtract(255, gray, dtype=np.uint8)
    if int(inverted.max()) - int(inverted.min()) < 8:
        return np.zeros_like(gray)
    _threshold, mask = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cast(NDArray[np.uint8], mask)


def _candidate_gray(
    image: Image.Image, degrees: OrientationDegrees
) -> NDArray[np.uint8]:
    """Gray image as if ``degrees`` CW were undone (putative upright)."""
    return _to_gray(upright_image(image, degrees))


def _projection_score(mask: NDArray[np.uint8]) -> float:
    """Higher when ink forms strong horizontal bands (text lines)."""
    if mask.size == 0 or int(np.count_nonzero(mask)) < 20:
        return 0.0
    ink = (mask > 0).astype(np.float64)
    row_proj = ink.sum(axis=1)
    col_proj = ink.sum(axis=0)
    row_var = float(np.var(row_proj))
    col_var = float(np.var(col_proj))
    if row_var + col_var <= 1e-9:
        return 0.0
    # Prefer horizontal structure strongly over vertical.
    return row_var / (col_var + 1.0)


def _global_vertical_mass_score(mask: NDArray[np.uint8]) -> float:
    """Positive when ink mass sits in the upper half (titles, headers, forms).

    Distinguishes 0° from 180° (and 90° from 270°) when letter-level
    ascender/descender cues are weak — e.g. all-caps printed pages.
    """
    if mask.size == 0:
        return 0.0
    ink = mask > 0
    if int(ink.sum()) < 30:
        return 0.0
    ys, _xs = np.where(ink)
    relative = float(ys.mean()) / float(max(mask.shape[0] - 1, 1))
    return 0.5 - relative


def _line_asymmetry_score(mask: NDArray[np.uint8]) -> float:
    """Positive when ink sits toward the top of text-line bands (upright Latin)."""
    if mask.size == 0:
        return 0.0
    ink = mask > 0
    if int(ink.sum()) < 30:
        return 0.0

    row_proj = ink.sum(axis=1).astype(np.float64)
    if float(row_proj.max()) <= 0.0:
        return 0.0
    threshold = 0.25 * float(row_proj.max())
    active = row_proj >= threshold
    # Segment contiguous active row bands.
    padded = np.concatenate(([False], active, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    scores: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        height = int(end - start)
        if height < 4:
            continue
        band = ink[start:end, :]
        if not band.any():
            continue
        ys, _xs = np.where(band)
        # Relative to band: 0 = top, 1 = bottom. Upright text mass is slightly
        # above centre because ascenders outweigh descenders.
        relative = ys.astype(np.float64) / float(max(height - 1, 1))
        centre_bias = 0.5 - float(relative.mean())
        scores.append(centre_bias)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _classic_score_for_candidate(
    image: Image.Image, degrees: OrientationDegrees
) -> float:
    gray = _candidate_gray(image, degrees)
    mask = _ink_mask(gray)
    # Close horizontally to emphasise text lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    closed = cast(NDArray[np.uint8], cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))
    structure = _projection_score(closed)
    mass = _global_vertical_mass_score(mask)
    asymmetry = _line_asymmetry_score(mask)
    # Structure picks the axis; mass/asymmetry break 0 vs 180.
    return structure * 10.0 + mass * 25.0 + asymmetry * 3.0


def _gradient_hv_ratio(gray: NDArray[np.uint8]) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    energy_x = float(np.mean(np.abs(grad_x)))
    energy_y = float(np.mean(np.abs(grad_y)))
    # Upright text has stronger vertical edges (stroke sides) relative to
    # horizontal edges along baselines — prefer moderate y/x balance with
    # horizontal runs; use energy_x / energy_y so horizontal strokes win.
    return energy_x / (energy_y + 1e-6)


def _run_length_score(mask: NDArray[np.uint8]) -> float:
    """Mean horizontal ink run length over vertical — higher for upright lines."""
    ink = mask > 0
    if int(ink.sum()) < 20:
        return 0.0

    def _mean_run(axis: int) -> float:
        lengths: list[int] = []
        rows = ink if axis == 1 else ink.T
        for row in rows:
            if not row.any():
                continue
            padded = np.concatenate(([False], row, [False]))
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for start, end in zip(starts, ends, strict=True):
                length = int(end - start)
                if length >= 2:
                    lengths.append(length)
        if not lengths:
            return 0.0
        return float(np.mean(lengths))

    horiz = _mean_run(1)
    vert = _mean_run(0)
    return horiz / (vert + 1e-6)


def _top_bottom_ink_ratio(mask: NDArray[np.uint8]) -> float:
    """Ink in the top third minus ink in the bottom third (normalised)."""
    if mask.size == 0:
        return 0.0
    ink = (mask > 0).astype(np.float64)
    h = ink.shape[0]
    if h < 9:
        return 0.0
    third = max(h // 3, 1)
    top = float(ink[:third, :].sum())
    bottom = float(ink[-third:, :].sum())
    total = top + bottom
    if total <= 0.0:
        return 0.0
    return (top - bottom) / total


def _classifier_score_for_candidate(
    image: Image.Image, degrees: OrientationDegrees
) -> float:
    gray = _candidate_gray(image, degrees)
    mask = _ink_mask(gray)
    if int(np.count_nonzero(mask)) < 20:
        return 0.0
    hv = _gradient_hv_ratio(gray)
    runs = _run_length_score(mask)
    structure = _projection_score(mask)
    mass = _global_vertical_mass_score(mask)
    top_bottom = _top_bottom_ink_ratio(mask)
    # Linear "tiny classifier" over hand-crafted features (no extra deps).
    # Uses top/bottom thirds instead of line asymmetry so it is independent
    # of the classic detector's closed-projection path.
    return structure * 6.0 + runs * 2.0 + hv * 0.5 + mass * 20.0 + top_bottom * 8.0
