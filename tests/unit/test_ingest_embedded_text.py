"""Unit tests for embedded PDF text usability checks (P2.1)."""

from __future__ import annotations

from torus.config import IngestConfig
from torus.core import BBox
from torus.ingest import EmbeddedWord, assess_words


def _word(
    text: str,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line: int = 0,
) -> EmbeddedWord:
    return EmbeddedWord(
        text=text,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        line=line,
    )


def test_empty_words_not_usable() -> None:
    result = assess_words([], page_index=0, config=IngestConfig())
    assert result.usable is False
    assert result.reason == "empty"
    assert result.words == ()


def test_too_few_chars_not_usable() -> None:
    words = [_word("ab", x0=0.1, y0=0.1, x1=0.2, y1=0.12)]
    config = IngestConfig(embedded_text_min_chars=25)
    result = assess_words(words, page_index=1, config=config)
    assert result.usable is False
    assert result.reason == "too_few_chars"
    assert result.words == ()


def test_cid_garbage_rejected() -> None:
    words = [
        _word("(cid:12)", x0=0.1, y0=0.1, x1=0.3, y1=0.12, line=0),
        _word("(cid:99)", x0=0.1, y0=0.15, x1=0.3, y1=0.17, line=1),
        _word("moretext", x0=0.1, y0=0.20, x1=0.4, y1=0.22, line=2),
        _word("enoughchars", x0=0.1, y0=0.25, x1=0.45, y1=0.27, line=3),
    ]
    result = assess_words(words, page_index=0, config=IngestConfig())
    assert result.usable is False
    assert result.reason == "cid_garbage"


def test_bad_reading_order_rejected() -> None:
    # Consecutive words jump upward across the page.
    words = [
        _word("AAAAAAA", x0=0.1, y0=0.80, x1=0.4, y1=0.84, line=0),
        _word("BBBBBBB", x0=0.1, y0=0.10, x1=0.4, y1=0.14, line=1),
        _word("CCCCCCC", x0=0.1, y0=0.70, x1=0.4, y1=0.74, line=2),
        _word("DDDDDDD", x0=0.1, y0=0.20, x1=0.4, y1=0.24, line=3),
    ]
    config = IngestConfig(
        embedded_text_min_chars=10,
        embedded_text_max_order_violation_ratio=0.25,
    )
    result = assess_words(words, page_index=0, config=config)
    assert result.usable is False
    assert result.reason == "bad_reading_order"


def test_ordered_words_usable() -> None:
    words = [
        _word("SYNTHAAA", x0=0.1, y0=0.10, x1=0.4, y1=0.14, line=0),
        _word("SYNTHBBB", x0=0.1, y0=0.20, x1=0.4, y1=0.24, line=1),
        _word("SYNTHCCC", x0=0.1, y0=0.30, x1=0.4, y1=0.34, line=2),
        _word("SYNTHDDD", x0=0.1, y0=0.40, x1=0.4, y1=0.44, line=3),
    ]
    config = IngestConfig(embedded_text_min_chars=20)
    result = assess_words(words, page_index=2, config=config)
    assert result.usable is True
    assert result.reason is None
    assert result.word_count == 4
    assert len(result.words) == 4
    assert result.page_index == 2


def test_nonprintable_rejected() -> None:
    junk = "X" * 20 + "\x00\x01\x02\x03\x04"
    words = [_word(junk, x0=0.1, y0=0.1, x1=0.5, y1=0.15)]
    config = IngestConfig(
        embedded_text_min_chars=10,
        embedded_text_max_nonprintable_ratio=0.02,
    )
    result = assess_words(words, page_index=0, config=config)
    assert result.usable is False
    assert result.reason == "nonprintable"
