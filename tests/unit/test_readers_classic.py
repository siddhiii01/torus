"""Unit tests for the classic OCR readers (P5.2). No OCR engine is required.

Paddle and docTR are exercised through injected fake engines; the Tesseract
hOCR parser is exercised on a synthetic hOCR string.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from torus.config.loader import default_config_dir, load_readers
from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.classic.common import (
    CropView,
    OcrWord,
    local_box_to_bbox,
    raw_score,
    single_word_char_confidences,
    unit_score,
    words_to_text,
)
from torus.readers.crop_hash import crop_image, pixel_box
from torus.readers.doctr import printed as doctr_printed
from torus.readers.load import instantiate_reader
from torus.readers.paddle import printed as paddle_printed
from torus.readers.tesseract import printed as tess


def _spec(name: str, family: str, module: str) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family=family,
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, scripts=["latin"]),
        version="t1",
        module=module,
    )


def _region(x0: float = 0.0, y0: float = 0.0, x1: float = 1.0, y1: float = 1.0):
    return Region(
        id="r1",
        page_index=3,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        kind=RegionKind.FIELD_VALUE,
    )


def _word(text: str, line: int = 0, conf: float | None = 0.9, chars=None) -> OcrWord:
    return OcrWord(
        text=text,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
        confidence=conf,
        char_confidences=chars,
        line=line,
    )


# --- geometry / scores -------------------------------------------------------


def test_pixel_box_matches_crop_image() -> None:
    img = Image.new("RGB", (200, 100))
    bbox = BBox(x0=0.25, y0=0.5, x1=0.75, y1=1.0)
    assert pixel_box(img.size, bbox) == (50, 50, 150, 100)
    assert crop_image(img, bbox).size == (100, 50)


def test_local_box_maps_into_page_coordinates() -> None:
    view = CropView(image=None, origin=(50, 50), page_size=(200, 100))
    box = local_box_to_bbox((0, 0, 100, 50), view)
    assert box == BBox(x0=0.25, y0=0.5, x1=0.75, y1=1.0)


def test_local_box_clamps_and_drops_degenerate() -> None:
    view = CropView(image=None, origin=(0, 0), page_size=(100, 100))
    clamped = local_box_to_bbox((-5, -5, 500, 50), view)
    assert clamped == BBox(x0=0.0, y0=0.0, x1=1.0, y1=0.5)
    assert local_box_to_bbox((10, 10, 10, 20), view) is None


@pytest.mark.parametrize("value", [None, float("nan"), -0.1, 1.5])
def test_unit_score_rejects_unusable(value: float | None) -> None:
    assert unit_score(value) is None


def test_unit_score_scale() -> None:
    assert unit_score(87.5, scale=100.0) == pytest.approx(0.875)


def test_raw_score_is_char_weighted_and_requires_all_scores() -> None:
    words = [_word("ab", conf=0.5), _word("cdef", conf=1.0)]
    assert raw_score(words) == pytest.approx((2 * 0.5 + 4 * 1.0) / 6)
    assert raw_score([_word("ab", conf=0.5), _word("cd", conf=None)]) is None
    assert raw_score([]) is None


def test_words_to_text_joins_lines() -> None:
    words = [_word("a", 0), _word("b", 0), _word("c", 1)]
    assert words_to_text(words) == "a b\nc"


def test_char_confidences_only_for_aligned_single_word() -> None:
    assert single_word_char_confidences([_word("ab", chars=(0.9, 0.8))]) == [0.9, 0.8]
    assert single_word_char_confidences([_word("ab", chars=(0.9,))]) is None
    assert single_word_char_confidences([_word("a", chars=(0.9,)), _word("b")]) is None
    assert single_word_char_confidences([_word("ab")]) is None


# --- tesseract ---------------------------------------------------------------

_HOCR = """<div class='ocr_page'><div class='ocr_carea'><p class='ocr_par'>
<span class='ocr_line' title="bbox 0 0 100 50">
<span class='ocrx_word' title='bbox 10 10 50 40; x_wconf 90'>
<span class='ocrx_cinfo' title='x_bboxes 10 10 30 40; x_conf 99.0'>A</span>
<span class='ocrx_cinfo' title='x_bboxes 30 10 50 40; x_conf 81.0'>&amp;</span>
</span>
<span class='ocrx_word' title='bbox 60 10 90 40; x_wconf 50'>
<span class='ocrx_cinfo' title='x_bboxes 60 10 90 40; x_conf 50'>Z</span>
</span></span>
<span class='ocr_line' title="bbox 0 50 100 100">
<span class='ocrx_word' title='bbox 10 60 40 90; x_wconf 70'>no</span>
<span class='ocrx_word' title='bbox 0 0 0 0; x_wconf 70'>zero</span>
</span></p></div></div>"""


def test_parse_hocr_words_boxes_and_char_scores() -> None:
    view = CropView(image=None, origin=(0, 0), page_size=(100, 100))
    words = tess.parse_hocr(_HOCR, view)
    assert [w.text for w in words] == ["A&", "Z", "no"]
    assert [w.line for w in words] == [0, 0, 1]
    assert words[0].confidence == pytest.approx(0.9)
    assert words[0].char_confidences == pytest.approx((0.99, 0.81))
    assert words[0].bbox == BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.4)
    assert words[2].char_confidences is None  # no per-character spans reported


def test_whitelist_only_when_it_can_be_expressed() -> None:
    assert tess.whitelist_for(ReadHint(charset=Charset.DIGITS)) == "0123456789"
    assert tess.whitelist_for(ReadHint(charset=Charset.ANY)) is None
    # No script given: an ASCII whitelist could silently drop other scripts.
    assert tess.whitelist_for(ReadHint(charset=Charset.LETTERS)) is None
    latin = ReadHint(charset=Charset.ALPHANUMERIC, script="latin")
    allowed = tess.whitelist_for(latin)
    assert allowed is not None and "7" in allowed and "q" in allowed


def test_tesseract_missing_binary_is_a_reader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tess, "find_binary", lambda: None)
    spec = _spec("tesseract_printed", "tesseract", "torus.readers.tesseract.printed")
    with pytest.raises(ReaderError):
        tess.create_reader(device="cpu", spec=spec)


# --- paddle / doctr via fake engines -----------------------------------------


class _FakePaddle:
    def predict(self, pixels: Any) -> list[dict[str, Any]]:
        assert pixels.shape[2] == 3
        return [
            {
                "rec_texts": ["hello world", "  "],
                "rec_scores": [0.8, 0.9],
                "rec_boxes": [(0, 0, 50, 10), (0, 10, 50, 20)],
            }
        ]


def test_paddle_reader_maps_lines_to_words_and_page_boxes() -> None:
    spec = _spec("paddle_printed", "paddle", "torus.readers.paddle.printed")
    reader = paddle_printed.PrintedReader(spec=spec, engine=_FakePaddle())
    page = Image.new("RGB", (200, 100), "white")
    region = _region(0.5, 0.0, 1.0, 1.0)  # crop is x 100..200
    words = reader.read_words(page, region, ReadHint())
    assert [w.text for w in words] == ["hello world"]  # blank line dropped
    assert words[0].bbox == BBox(x0=0.5, y0=0.0, x1=0.75, y1=0.1)
    reading = reader.read(page, region, ReadHint())
    assert reading.text == "hello world"
    assert reading.raw_score == pytest.approx(0.8)
    assert reading.char_confidences is None  # Paddle reports none: never invented
    assert reading.reader_name == "paddle_printed"
    assert reading.reader_version == "t1"
    assert reading.error is None


def _doctr_doc(*lines: list[tuple[str, float, Any]]) -> Any:
    line_objs = [
        SimpleNamespace(
            words=[
                SimpleNamespace(value=v, confidence=c, geometry=g) for v, c, g in line
            ]
        )
        for line in lines
    ]
    return SimpleNamespace(
        pages=[SimpleNamespace(blocks=[SimpleNamespace(lines=line_objs)])]
    )


def test_doctr_reader_maps_words_lines_and_scores() -> None:
    doc = _doctr_doc(
        [("ab", 0.9, ((0.0, 0.0), (0.5, 0.5))), ("cd", 0.7, ((0.5, 0.0), (1.0, 0.5)))],
        [("ef", 0.8, ((0.0, 0.5), (1.0, 1.0)))],
    )
    spec = _spec("doctr_printed", "doctr", "torus.readers.doctr.printed")
    reader = doctr_printed.PrintedReader(spec=spec, predictor=lambda images: doc)
    reading = reader.read(Image.new("RGB", (100, 100), "white"), _region(), ReadHint())
    assert reading.text == "ab cd\nef"
    assert reading.raw_score == pytest.approx((2 * 0.9 + 2 * 0.7 + 2 * 0.8) / 6)
    assert reading.char_confidences is None


def test_paddle_and_doctr_skip_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    for module, family in (
        ("torus.readers.paddle.printed", "paddle"),
        ("torus.readers.doctr.printed", "doctr"),
    ):
        spec = _spec(f"{family}_printed", family, module)
        with pytest.raises(ReaderError, match="not installed"):
            instantiate_reader(spec, device=DeviceKind.CPU)


def test_non_image_input_is_a_soft_error_not_a_crash() -> None:
    spec = _spec("paddle_printed", "paddle", "torus.readers.paddle.printed")
    reader = paddle_printed.PrintedReader(spec=spec, engine=_FakePaddle())
    with pytest.raises(TypeError):
        reader.read(object(), _region(), ReadHint())  # runner converts to Reading.error


def test_shipped_config_lists_classic_readers_disabled_and_honest_caps() -> None:
    cfg = load_readers(default_config_dir() / "readers.yaml")
    by_name = {r.name: r for r in cfg.readers}
    for name in ("paddle_printed", "doctr_printed", "tesseract_printed"):
        assert by_name[name].enabled is False  # enabled only after the bake-off
        assert by_name[name].module is not None
    # Only Tesseract (hOCR) exposes per-character scores among the three.
    assert by_name["tesseract_printed"].capabilities.per_char_confidence is True
    assert by_name["paddle_printed"].capabilities.per_char_confidence is False
    assert by_name["doctr_printed"].capabilities.per_char_confidence is False
