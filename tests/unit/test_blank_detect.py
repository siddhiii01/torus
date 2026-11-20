"""Unit tests for blank / near-blank detection (P2.5)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.fixtures.synth import (
    generate_blank,
    generate_near_blank,
    generate_printed_form,
)
from torus.config import BlankDetectConfig, load_config
from torus.core import Document, Page
from torus.preprocess import apply_blank_detect, detect_blank, flag_blank_pages
from torus.preprocess.blank import count_text_like_components


def _image_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def test_repo_pipeline_loads_blank_detect_defaults() -> None:
    cfg = load_config()
    blank = cfg.pipeline.preprocess.blank_detect
    assert blank.enabled is True
    assert blank.max_ink_density == 0.004
    assert blank.min_text_components == 3
    assert blank.min_component_height_ratio <= blank.max_component_height_ratio


def test_height_ratio_order_rejected() -> None:
    with pytest.raises(ValueError, match="min_component_height_ratio"):
        BlankDetectConfig(
            min_component_height_ratio=0.1,
            max_component_height_ratio=0.05,
        )


def test_disabled_is_not_blank() -> None:
    image = Image.new("RGB", (200, 280), color=(255, 255, 255))
    result = detect_blank(image, BlankDetectConfig(enabled=False))
    assert result.blank is False
    assert result.reason == "disabled"


def test_pure_white_is_blank() -> None:
    image = Image.new("RGB", (400, 560), color=(255, 255, 255))
    result = detect_blank(image)
    assert result.blank is True
    assert result.near_blank is False
    assert result.ink_density == 0.0
    assert result.text_component_count == 0
    assert result.reason == "blank"


def test_text_page_is_not_blank() -> None:
    # Glyph-sized filled marks (font-independent) pass the OCR-presence gate.
    image = Image.new("RGB", (400, 560), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)
    # Height ~3% of page (within default 0.4%–8% component height band).
    boxes = [
        (40, 50, 55, 70),
        (90, 50, 105, 70),
        (140, 50, 155, 70),
        (40, 120, 55, 140),
        (90, 120, 105, 140),
        (140, 120, 155, 140),
    ]
    for box in boxes:
        draw.rectangle(box, fill=(15, 15, 15))
    result = detect_blank(image)
    assert result.blank is False
    assert result.text_component_count >= 3
    assert result.ink_density > 0.0


def test_apply_sets_page_blank_flag() -> None:
    page = Page(index=0, width=300, height=400)
    page.bind_image(Image.new("RGB", (300, 400), color=(252, 252, 252)))
    result = apply_blank_detect(page)
    assert result.blank is True
    assert page.blank is True


def test_apply_no_image_leaves_blank_false() -> None:
    page = Page(index=0, width=100, height=100)
    result = apply_blank_detect(page)
    assert result.reason == "no_image"
    assert page.blank is False


def test_flag_blank_pages_never_drops() -> None:
    blank = generate_blank(seed=1)
    form = generate_printed_form(seed=2)
    pages: list[Page] = []
    for synth in (blank, form, generate_near_blank(seed=3)):
        page = Page(
            index=synth.page_index,
            width=synth.width,
            height=synth.height,
        )
        page.bind_image(synth.image)
        pages.append(page)
    # Distinct indices so Document stays valid.
    for i, page in enumerate(pages):
        page.index = i

    doc = Document(id="synth-blank-doc", source_path="synth://blank", pages=pages)
    before = len(doc.pages)
    results = flag_blank_pages(doc)
    assert len(doc.pages) == before
    assert len(results) == before
    assert doc.pages[0].blank is True
    assert doc.pages[1].blank is False
    assert doc.pages[2].blank is True
    assert results[2].near_blank is True


def test_blank_generator_deterministic() -> None:
    a = generate_blank(seed=11, near_blank=True)
    b = generate_blank(seed=11, near_blank=True)
    assert _image_digest(a.image) == _image_digest(b.image)
    assert a.kind == "blank"
    assert a.regions == ()
    assert a.meta["near_blank"] is True


def test_count_text_like_on_empty() -> None:
    gray = np.full((120, 80), 255, dtype=np.uint8)
    assert count_text_like_components(gray, BlankDetectConfig()) == 0
