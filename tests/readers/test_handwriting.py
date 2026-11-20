"""Tests for the handwriting line readers (P5.6): TrOCR and PARSeq.

Logic tests use injected fake decoders and need no torch. Smoke tests are marked
``model`` and skip when the stack or weights are absent. Crops are synthetic.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
from PIL import Image, ImageDraw, ImageFont

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.handwriting.common import (
    Decoded,
    allowed_characters,
    max_chars,
    sequence_probability,
    token_is_allowed,
)
from torus.readers.handwriting.parseq import ParseqReader
from torus.readers.handwriting.trocr import TrocrReader
from torus.readers.load import instantiate_reader


def _region() -> Region:
    return Region(
        id="synth-1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )


def _image() -> Image.Image:
    return Image.new("L", (200, 60), "white")


def _spec(name: str, module: str) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family="handwriting",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(handwriting=True, printed=False, scripts=["latin"]),
        version="t",
        module=module,
    )


class _FakeDecoder:
    def __init__(self, result: Decoded | Exception) -> None:
        self._result = result
        self.seen_hint: ReadHint | None = None
        self.seen_mode: str | None = None

    def decode(self, crop: Image.Image, hint: ReadHint) -> Decoded:
        self.seen_hint = hint
        self.seen_mode = crop.mode
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


# --- pure helpers ----------------------------------------------------------


def test_allowed_characters_per_charset() -> None:
    assert allowed_characters(Charset.ANY) is None
    digits = allowed_characters(Charset.DIGITS)
    assert digits is not None and "5" in digits and "a" not in digits
    letters = allowed_characters(Charset.LETTERS)
    assert letters is not None and "a" in letters and "5" not in letters
    alnum = allowed_characters(Charset.ALPHANUMERIC)
    assert alnum is not None and {"a", "Z", "5"} <= alnum


def test_token_is_allowed_ignores_surrounding_space_markers() -> None:
    allowed = allowed_characters(Charset.DIGITS)
    assert allowed is not None
    assert token_is_allowed("Ġ12", allowed)
    assert token_is_allowed("▁7", allowed)
    assert not token_is_allowed("1a", allowed)
    assert not token_is_allowed("", allowed)  # empty decode is never a free pass


def test_max_chars_prefers_tightest_bound() -> None:
    assert max_chars(ReadHint()) is None
    assert max_chars(ReadHint(expected_length=10)) == 10
    assert max_chars(ReadHint(max_length=8)) == 8
    assert max_chars(ReadHint(expected_length=10, max_length=8)) == 8


def test_sequence_probability_is_product_of_token_probs() -> None:
    assert sequence_probability([0.5, 0.5]) == pytest.approx(0.25)
    assert sequence_probability([math.exp(-0.1)]) == pytest.approx(math.exp(-0.1))


@pytest.mark.parametrize("bad", [[], [0.5, float("nan")], [0.5, 1.5], [-0.1]])
def test_sequence_probability_refuses_to_invent_a_score(bad: list[float]) -> None:
    assert sequence_probability(bad) is None


# --- reader behaviour (fake decoders) -------------------------------------


@pytest.mark.parametrize("cls", [TrocrReader, ParseqReader])
def test_reader_returns_decoded_text_score_and_identity(cls: type) -> None:
    decoder = _FakeDecoder(Decoded(text="ab12", score=0.8))
    reader = cls(spec=_spec("hw", "x"), decoder=decoder)
    hint = ReadHint(charset=Charset.ALPHANUMERIC)
    reading = reader.read(_image(), _region(), hint)
    assert (reading.text, reading.raw_score, reading.error) == ("ab12", 0.8, None)
    assert reading.reader_name == "hw" and reading.reader_version == "t"
    assert reading.char_confidences is None
    assert decoder.seen_hint == hint
    assert decoder.seen_mode == "RGB"  # greyscale pages are converted for the model
    assert reader.capabilities().handwriting is True


@pytest.mark.parametrize("cls", [TrocrReader, ParseqReader])
def test_reader_without_score_leaves_it_none(cls: type) -> None:
    reader = cls(decoder=_FakeDecoder(Decoded(text="x", score=None)))
    assert reader.read(_image(), _region(), ReadHint()).raw_score is None


@pytest.mark.parametrize("cls", [TrocrReader, ParseqReader])
def test_decoder_failure_is_a_soft_error_not_a_crash(cls: type) -> None:
    reader = cls(decoder=_FakeDecoder(RuntimeError("boom")))
    reading = reader.read(_image(), _region(), ReadHint())
    assert reading.text == "" and reading.raw_score is None
    assert reading.error is not None and "RuntimeError" in reading.error


@pytest.mark.parametrize("cls", [TrocrReader, ParseqReader])
def test_non_image_input_is_a_soft_error(cls: type) -> None:
    reader = cls(decoder=_FakeDecoder(Decoded(text="x", score=None)))
    reading = reader.read(object(), _region(), ReadHint())
    assert reading.text == "" and reading.error is not None


def test_error_message_never_contains_decoded_text() -> None:
    reader = TrocrReader(decoder=_FakeDecoder(ValueError("bad")))
    reading = reader.read(_image(), _region(), ReadHint())
    assert reading.error == "ValueError: bad"


# --- missing stack = skip, not crash --------------------------------------


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is None for m in modules)


@pytest.mark.skipif(not _missing("torch", "transformers"), reason="stack installed")
def test_trocr_without_stack_raises_reader_error_at_construction() -> None:
    with pytest.raises(ReaderError, match="vlm"):
        TrocrReader()


@pytest.mark.skipif(not _missing("torch", "doctr"), reason="stack installed")
def test_parseq_without_stack_raises_reader_error_at_construction() -> None:
    with pytest.raises(ReaderError, match="doctr"):
        ParseqReader()


def test_modules_import_without_heavy_stack() -> None:
    # Package import must never pull torch/transformers/doctr in.
    import sys

    import torus.readers.handwriting.parseq  # noqa: F401
    import torus.readers.handwriting.trocr  # noqa: F401

    if _missing("torch"):
        assert "torch" not in sys.modules


def test_load_path_reports_missing_stack_as_reader_error() -> None:
    if not _missing("torch", "transformers"):
        pytest.skip("stack installed")
    spec = _spec("trocr_handwritten", "torus.readers.handwriting.trocr")
    with pytest.raises(ReaderError):
        instantiate_reader(spec, device=DeviceKind.CPU)


# --- model smoke tests ----------------------------------------------------


def _handwriting_like_crop(text: str) -> Image.Image:
    image = Image.new("RGB", (400, 90), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text((14, 16), text, font=font, fill="black")
    return image


@pytest.mark.model
@pytest.mark.skipif(_missing("torch", "transformers"), reason="vlm extra absent")
def test_trocr_smoke_reads_synthetic_crop() -> None:
    spec = _spec("trocr_handwritten", "torus.readers.handwriting.trocr")
    try:
        reader = instantiate_reader(spec, device=DeviceKind.CPU)
    except ReaderError as exc:
        pytest.skip(str(exc))
    reading = reader.read(
        _handwriting_like_crop("48291"), _region(), ReadHint(charset=Charset.DIGITS)
    )
    if reading.error and "weights" in reading.error:
        pytest.skip(reading.error)
    assert reading.error is None
    assert set(reading.text) <= set("0123456789 ")
    assert reading.raw_score is not None and 0.0 <= reading.raw_score <= 1.0


@pytest.mark.model
@pytest.mark.skipif(_missing("torch", "doctr"), reason="doctr extra absent")
def test_parseq_smoke_reads_synthetic_crop() -> None:
    spec = _spec("parseq_handwritten", "torus.readers.handwriting.parseq")
    try:
        reader = instantiate_reader(spec, device=DeviceKind.CPU)
    except ReaderError as exc:
        pytest.skip(str(exc))
    reading = reader.read(_handwriting_like_crop("48291"), _region(), ReadHint())
    if reading.error and "weights" in reading.error:
        pytest.skip(reading.error)
    assert reading.error is None
    assert reading.raw_score is None or 0.0 <= reading.raw_score <= 1.0
