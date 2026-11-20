"""Shared pieces for the classic OCR readers (Tesseract, PaddleOCR, docTR).

``Reading`` (core) carries only text plus optional scores. Word boxes live on
``OcrWord`` and are exposed through each reader's ``read_words()``; the core
schema is unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.models import BBox, ReaderCaps, ReadHint, Reading, Region
from torus.readers.crop_hash import pixel_box


@dataclass(frozen=True)
class OcrWord:
    """One recognised word with a page-normalised box.

    ``confidence`` and ``char_confidences`` are the engine's own scores in
    [0, 1], or ``None`` when the engine does not report them. Never derived.
    """

    text: str
    bbox: BBox
    confidence: float | None = None
    char_confidences: tuple[float, ...] | None = None
    line: int = 0


@dataclass(frozen=True)
class CropView:
    """A region crop plus what is needed to map crop pixels back to the page."""

    image: Any
    origin: tuple[int, int]
    page_size: tuple[int, int]


def crop_region(image: Any, region: Region) -> CropView:
    """Crop ``image`` (PIL) to ``region.bbox``; raise ``TypeError`` if not PIL-like."""
    size = getattr(image, "size", None)
    crop = getattr(image, "crop", None)
    if size is None or not callable(crop):
        msg = "classic OCR readers need a PIL-like image (size + crop)"
        raise TypeError(msg)
    page_w, page_h = int(size[0]), int(size[1])
    if page_w <= 0 or page_h <= 0:
        msg = "image has no pixels"
        raise ValueError(msg)
    x0, y0, x1, y1 = pixel_box((page_w, page_h), region.bbox)
    return CropView(
        image=crop((x0, y0, x1, y1)).convert("RGB"),
        origin=(x0, y0),
        page_size=(page_w, page_h),
    )


def local_box_to_bbox(
    box: tuple[float, float, float, float],
    view: CropView,
) -> BBox | None:
    """Map a crop-pixel box (x0, y0, x1, y1) to a page-normalised ``BBox``.

    Returns ``None`` for boxes with no area after clamping.
    """
    ox, oy = view.origin
    pw, ph = view.page_size

    def clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    x0 = clamp((ox + box[0]) / pw)
    y0 = clamp((oy + box[1]) / ph)
    x1 = clamp((ox + box[2]) / pw)
    y1 = clamp((oy + box[3]) / ph)
    if x0 >= x1 or y0 >= y1:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def unit_score(value: float | None, *, scale: float = 1.0) -> float | None:
    """Convert an engine score to [0, 1]; ``None`` and out-of-range stay ``None``."""
    if value is None:
        return None
    score = float(value) / scale
    if score != score or not 0.0 <= score <= 1.0:  # NaN or out of range
        return None
    return score


def words_to_text(words: Sequence[OcrWord]) -> str:
    """Join words: spaces within a line, newlines between lines."""
    lines: list[list[str]] = []
    current = None
    for word in words:
        if word.line != current:
            lines.append([])
            current = word.line
        lines[-1].append(word.text)
    return "\n".join(" ".join(parts) for parts in lines)


def raw_score(words: Sequence[OcrWord]) -> float | None:
    """Character-weighted mean of the engine's word scores.

    ``None`` unless every word has a score: a partial mean would overstate it.
    This is a raw engine score, not a calibrated confidence.
    """
    if not words or any(w.confidence is None for w in words):
        return None
    weight = sum(len(w.text) for w in words)
    if weight == 0:
        return None
    total = sum(len(w.text) * (w.confidence or 0.0) for w in words)
    return total / weight


def single_word_char_confidences(words: Sequence[OcrWord]) -> list[float] | None:
    """Per-character scores aligned 1:1 with ``Reading.text``.

    Only a single-word reading has no separator characters that would lack a
    score, so anything else returns ``None`` (use ``read_words`` for detail).
    """
    if len(words) != 1:
        return None
    word = words[0]
    if word.char_confidences is None or len(word.char_confidences) != len(word.text):
        return None
    return list(word.char_confidences)


def build_reading(
    *,
    region: Region,
    name: str,
    version: str,
    words: Sequence[OcrWord],
    started: float,
) -> Reading:
    """Assemble a ``Reading`` from words; ``started`` is a ``perf_counter`` value."""
    return Reading(
        region_id=region.id,
        reader_name=name,
        reader_version=version,
        text=words_to_text(words),
        char_confidences=single_word_char_confidences(words),
        raw_score=raw_score(words),
        time_ms=(time.perf_counter() - started) * 1000.0,
    )


class ClassicReaderBase:
    """Common identity/caps handling; subclasses implement ``read_words``."""

    default_name = "classic_ocr"

    def __init__(self, *, device: str = "cpu", spec: ReaderSpec | None = None) -> None:
        self.device = device
        self._spec = spec
        self.name = spec.name if spec is not None else self.default_name
        self._version = (spec.version if spec is not None else None) or "0"
        self._caps = spec.capabilities if spec is not None else ReaderCaps()

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> ReaderCaps:
        return self._caps

    def read_words(
        self, image: Any, region: Region, hint: ReadHint
    ) -> list[OcrWord]:  # pragma: no cover - abstract
        raise NotImplementedError

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        started = time.perf_counter()
        words = self.read_words(image, region, hint)
        return build_reading(
            region=region,
            name=self.name,
            version=self._version,
            words=words,
            started=started,
        )
