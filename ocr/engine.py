"""Read a PDF or image into words, boxes, confidence, and a page status.

Boxes are normalized 0-1, origin at the top-left of the page. The text layer
is used when the embedded text has at least MIN_CHARS characters. Otherwise
docTR runs. A scrambled text layer still counts as usable here; the review
catches reading order. This module does not log document text.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Weights stay on C: (the user profile). Do not point these at D:.
_cache = Path.home() / ".cache"
os.environ.setdefault("DOCTR_CACHE_DIR", str(_cache / "doctr"))
os.environ.setdefault("TORCH_HOME", str(_cache / "torch"))
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pymupdf
from PIL import Image

MIN_CHARS = 25
LOW_CONF = 0.60
DPI = 200
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_doctr = None


@dataclass
class Word:
    text: str
    bbox: tuple[float, float, float, float]
    conf: float | None
    line: int


@dataclass
class Field:
    field: str
    value: str


@dataclass
class PageRead:
    index: int
    width: int
    height: int
    reader: str
    status: str
    reason: str
    seconds: float
    words: list[Word] = field(default_factory=list)
    fields: list[Field] = field(default_factory=list)
    tables: list[list[dict[str, str]]] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class DocumentRead:
    pages: list[PageRead]

    def to_json(self) -> dict:
        return {"pages": [p.to_json() for p in self.pages]}


def read_document(path: Path) -> tuple[DocumentRead, list[np.ndarray]]:
    """Return the read result and the RGB page images used for review."""
    if path.suffix.lower() in IMAGE_EXTS:
        img = np.array(Image.open(path).convert("RGB"))
        page = _read_page(img, None, 0)
        return DocumentRead([page]), [img]

    images: list[np.ndarray] = []
    pages: list[PageRead] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise PermissionError("password-protected")
        for i, pdf_page in enumerate(doc):
            pix = pdf_page.get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            images.append(img)
            pages.append(_read_page(img, pdf_page, i))
    return DocumentRead(pages), images


def _read_page(img: np.ndarray, pdf_page, index: int) -> PageRead:
    import time

    height, width = img.shape[:2]
    t0 = time.perf_counter()
    if pdf_page is not None:
        embedded = pdf_page.get_text("text").strip()
        if len(embedded) >= MIN_CHARS:
            words = _finish(_text_layer_words(pdf_page))
            return _page(index, width, height, "textlayer", "ok", "", time.perf_counter() - t0, words)
    try:
        words, conf = _doctr_words(img)
    except Exception as exc:
        return _page(index, width, height, "doctr", "needs_review", type(exc).__name__, time.perf_counter() - t0, [])
    words = _finish(words)
    reason = ""
    status = "ok"
    if not words:
        status, reason = "needs_review", "empty"
    elif conf < LOW_CONF:
        status, reason = "needs_review", "low_confidence"
    return _page(index, width, height, "doctr", status, reason, time.perf_counter() - t0, words)


def _finish(words: list[Word]) -> list[Word]:
    from ocr.fields import reading_order
    return reading_order(words)


def _page(index, width, height, reader, status, reason, seconds, words) -> PageRead:
    from ocr.fields import extract
    pairs, tables = extract(words)
    return PageRead(
        index, width, height, reader, status, reason, seconds, words,
        [Field(name, value) for name, value in pairs], tables,
    )


def _text_layer_words(pdf_page) -> list[Word]:
    rect = pdf_page.rect
    raw = pdf_page.get_text("words")
    raw.sort(key=lambda w: (w[5], w[6], w[7]))
    words: list[Word] = []
    line_key: tuple[int, int] | None = None
    line = -1
    for x0, y0, x1, y1, text, block, ln, _word in raw:
        if not str(text).strip():
            continue
        key = (block, ln)
        if key != line_key:
            line += 1
            line_key = key
        words.append(Word(
            str(text),
            (x0 / rect.width, y0 / rect.height, x1 / rect.width, y1 / rect.height),
            None,
            line,
        ))
    return words


def _doctr_model():
    global _doctr
    if _doctr is None:
        from doctr.models import ocr_predictor
        _doctr = ocr_predictor(pretrained=True)
    return _doctr


def _doctr_words(img: np.ndarray) -> tuple[list[Word], float]:
    result = _doctr_model()([img])
    words: list[Word] = []
    confs: list[float] = []
    line = 0
    for block in result.pages[0].blocks:
        for ln in block.lines:
            for w in ln.words:
                (x0, y0), (x1, y1) = w.geometry
                words.append(Word(w.value, (float(x0), float(y0), float(x1), float(y1)), float(w.confidence), line))
                confs.append(float(w.confidence))
            line += 1
    conf = sum(confs) / len(confs) if confs else 0.0
    return words, conf
