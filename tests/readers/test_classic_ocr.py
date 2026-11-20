"""Reader smoke tests for the classic OCR family (P5.2).

Each test skips when its engine is not installed. Crops are drawn in code
(synthetic); nothing here depends on a real document.
"""

from __future__ import annotations

import importlib.util

import pytest
from PIL import Image, ImageDraw, ImageFont

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.load import instantiate_reader
from torus.readers.tesseract.printed import find_binary

pytestmark = pytest.mark.model


def _printed_crop(text: str) -> Image.Image:
    image = Image.new("RGB", (480, 100), "white")
    font = ImageFont.load_default(size=44)
    ImageDraw.Draw(image).text((12, 22), text, font=font, fill="black")
    return image


def _region(kind: RegionKind = RegionKind.TEXT_LINE) -> Region:
    return Region(
        id="synth-1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=kind,
    )


def _spec(name: str, family: str, module: str, *, per_char: bool) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family=family,
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(
            printed=True, scripts=["latin"], per_char_confidence=per_char
        ),
        version="smoke",
        module=module,
    )


@pytest.mark.skipif(find_binary() is None, reason="tesseract not installed")
def test_tesseract_reads_synthetic_printed_crop() -> None:
    spec = _spec(
        "tesseract_printed",
        "tesseract",
        "torus.readers.tesseract.printed",
        per_char=True,
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    reading = reader.read(_printed_crop("HELLO 2024"), _region(), ReadHint())
    assert reading.error is None
    assert reading.text.replace(" ", "").upper() == "HELLO2024"
    assert reading.raw_score is not None and 0.5 < reading.raw_score <= 1.0
    assert reading.time_ms > 0

    words = reader.read_words(_printed_crop("HELLO 2024"), _region(), ReadHint())  # type: ignore[attr-defined]
    assert [w.text for w in words] == ["HELLO", "2024"]
    assert all(
        w.char_confidences and len(w.char_confidences) == len(w.text) for w in words
    )
    assert all(0.0 <= w.bbox.x0 < w.bbox.x1 <= 1.0 for w in words)


@pytest.mark.skipif(find_binary() is None, reason="tesseract not installed")
def test_tesseract_digit_hint_single_word_has_aligned_char_scores() -> None:
    spec = _spec(
        "tesseract_printed",
        "tesseract",
        "torus.readers.tesseract.printed",
        per_char=True,
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    hint = ReadHint(charset=Charset.DIGITS)
    reading = reader.read(_printed_crop("482913"), _region(), hint)
    assert reading.text == "482913"
    assert reading.char_confidences is not None
    assert len(reading.char_confidences) == len(reading.text)


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


@pytest.mark.skipif(not _installed("paddle", "paddleocr"), reason="paddle extra absent")
def test_paddle_reads_synthetic_printed_crop() -> None:
    spec = _spec(
        "paddle_printed", "paddle", "torus.readers.paddle.printed", per_char=False
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    reading = reader.read(_printed_crop("HELLO 2024"), _region(), ReadHint())
    assert reading.error is None
    assert reading.text.replace(" ", "").upper() == "HELLO2024"
    assert reading.char_confidences is None


@pytest.mark.skipif(not _installed("doctr", "torch"), reason="doctr extra absent")
def test_doctr_reads_synthetic_printed_crop() -> None:
    spec = _spec(
        "doctr_printed", "doctr", "torus.readers.doctr.printed", per_char=False
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    reading = reader.read(_printed_crop("HELLO 2024"), _region(), ReadHint())
    assert reading.error is None
    assert reading.text.replace(" ", "").upper() == "HELLO2024"
    assert reading.char_confidences is None
