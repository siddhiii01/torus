"""Embedded PDF text-layer extraction and usability checks (P2.1).

Usable means: enough characters, not garbled (CID garbage / non-printables),
and roughly in top-to-bottom / left-to-right reading order. Downstream stages
may use usable text; unusable layers are ignored (image readers take over).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import IngestConfig
from torus.core.models import BBox

# PDF extraction garbage like "(cid:12)" from broken ToUnicode maps.
_CID_GARBAGE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)

# Same-line tolerance as a fraction of page height.
_SAME_LINE_Y = 0.02
# Significant upward jump (reading-order violation) as a fraction of page height.
_UPWARD_JUMP_Y = 0.05


class EmbeddedWord(BaseModel):
    """One word from a PDF text layer (normalised bbox, top-left origin)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    bbox: BBox
    line: int = Field(ge=0)


class PageEmbeddedText(BaseModel):
    """Per-page text-layer assessment. Do not log ``words`` or raw text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_index: int = Field(ge=0)
    usable: bool
    reason: str | None = None
    char_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    words: tuple[EmbeddedWord, ...] = ()


def assess_pdf_page_text(
    pdf_page: Any,
    page_index: int,
    config: IngestConfig,
) -> PageEmbeddedText:
    """Extract words from ``pdf_page`` and decide whether the layer is usable."""
    words = _extract_words(pdf_page)
    return assess_words(words, page_index=page_index, config=config)


def assess_words(
    words: list[EmbeddedWord],
    *,
    page_index: int,
    config: IngestConfig,
) -> PageEmbeddedText:
    """Run usability checks on an already-extracted word list (unit-testable)."""
    if not words:
        return PageEmbeddedText(
            page_index=page_index,
            usable=False,
            reason="empty",
            char_count=0,
            word_count=0,
            words=(),
        )

    joined = " ".join(w.text for w in words)
    char_count = sum(1 for ch in joined if not ch.isspace())
    if char_count < config.embedded_text_min_chars:
        return PageEmbeddedText(
            page_index=page_index,
            usable=False,
            reason="too_few_chars",
            char_count=char_count,
            word_count=len(words),
            words=(),
        )

    if config.embedded_text_reject_cid_garbage and _CID_GARBAGE.search(joined):
        return PageEmbeddedText(
            page_index=page_index,
            usable=False,
            reason="cid_garbage",
            char_count=char_count,
            word_count=len(words),
            words=(),
        )

    nonprintable_ratio = _nonprintable_ratio(joined)
    if nonprintable_ratio > config.embedded_text_max_nonprintable_ratio:
        return PageEmbeddedText(
            page_index=page_index,
            usable=False,
            reason="nonprintable",
            char_count=char_count,
            word_count=len(words),
            words=(),
        )

    order_ratio = _order_violation_ratio(words)
    if order_ratio > config.embedded_text_max_order_violation_ratio:
        return PageEmbeddedText(
            page_index=page_index,
            usable=False,
            reason="bad_reading_order",
            char_count=char_count,
            word_count=len(words),
            words=(),
        )

    return PageEmbeddedText(
        page_index=page_index,
        usable=True,
        reason=None,
        char_count=char_count,
        word_count=len(words),
        words=tuple(words),
    )


def _extract_words(pdf_page: Any) -> list[EmbeddedWord]:
    """Pull words with normalised boxes from a PyMuPDF page."""
    rect = pdf_page.rect
    width = float(rect.width)
    height = float(rect.height)
    if width <= 0 or height <= 0:
        return []

    raw = pdf_page.get_text("words")
    # PyMuPDF words: x0, y0, x1, y1, word, block_no, line_no, word_no
    raw = sorted(raw, key=lambda w: (w[5], w[6], w[7]))

    words: list[EmbeddedWord] = []
    line_key: tuple[int, int] | None = None
    line = -1
    for item in raw:
        x0, y0, x1, y1, text, block, ln, _word_no = item[:8]
        cleaned = str(text).strip()
        if not cleaned:
            continue
        key = (int(block), int(ln))
        if key != line_key:
            line += 1
            line_key = key
        try:
            bbox = BBox(
                x0=_clamp01(float(x0) / width),
                y0=_clamp01(float(y0) / height),
                x1=_clamp01(float(x1) / width),
                y1=_clamp01(float(y1) / height),
            )
        except ValueError:
            # Degenerate box after clamp — skip this word.
            continue
        words.append(EmbeddedWord(text=cleaned, bbox=bbox, line=line))
    return words


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _nonprintable_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch.isprintable() or ch.isspace():
            continue
        bad += 1
    return bad / len(text)


def _order_violation_ratio(words: list[EmbeddedWord]) -> float:
    """Fraction of consecutive pairs that break top-to-bottom / LTR order."""
    if len(words) < 2:
        return 0.0
    violations = 0
    pairs = 0
    for left, right in zip(words, words[1:], strict=False):
        pairs += 1
        ay = (left.bbox.y0 + left.bbox.y1) / 2.0
        by = (right.bbox.y0 + right.bbox.y1) / 2.0
        ax = (left.bbox.x0 + left.bbox.x1) / 2.0
        bx = (right.bbox.x0 + right.bbox.x1) / 2.0
        if abs(ay - by) <= _SAME_LINE_Y:
            if ax > bx + 0.01:
                violations += 1
        elif by < ay - _UPWARD_JUMP_Y:
            violations += 1
    return violations / pairs if pairs else 0.0
