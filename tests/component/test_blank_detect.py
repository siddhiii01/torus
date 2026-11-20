"""Component tests: blank detection on synthetic pages (P2.5)."""

from __future__ import annotations

from tests.fixtures.synth import (
    generate_blank,
    generate_fax_degraded,
    generate_near_blank,
    generate_printed_form,
    generate_rotated,
    generate_skewed,
)

from torus.config.schemas import BlankDetectConfig
from torus.core import Document, Page
from torus.preprocess import apply_blank_detect, detect_blank, flag_blank_pages


def _page_from_synth(synth: object, index: int = 0) -> Page:
    page = Page(
        index=index,
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def test_blank_synth_flagged() -> None:
    synth = generate_blank(seed=21)
    result = detect_blank(synth.image)
    assert result.blank is True
    assert result.near_blank is False
    assert result.reason == "blank"


def test_near_blank_synth_flagged() -> None:
    synth = generate_near_blank(seed=22, speck_count=4)
    result = detect_blank(synth.image)
    assert result.blank is True
    assert result.near_blank is True
    assert result.reason == "near_blank"
    assert result.ink_density > 0.0
    assert result.ink_density <= BlankDetectConfig().max_ink_density


def test_printed_form_not_blank() -> None:
    synth = generate_printed_form(seed=23)
    result = detect_blank(synth.image)
    assert result.blank is False
    assert result.text_component_count >= 3


def test_rotated_and_skewed_content_not_blank() -> None:
    rotated = generate_rotated(seed=24, degrees=90)
    skewed = generate_skewed(seed=25, angle_deg=8.0)
    assert detect_blank(rotated.image).blank is False
    assert detect_blank(skewed.image).blank is False


def test_fax_degraded_content_not_blank() -> None:
    synth = generate_fax_degraded(seed=26, strength=0.6)
    result = detect_blank(synth.image)
    assert result.blank is False


def test_apply_blank_detect_records_without_dropping_image() -> None:
    synth = generate_blank(seed=27)
    page = _page_from_synth(synth)
    before_size = page.image.size
    result = apply_blank_detect(page)
    assert result.blank is True
    assert page.blank is True
    assert page.has_image is True
    assert page.image.size == before_size


def test_document_keeps_blank_pages() -> None:
    """Acceptance: blank pages are recorded, never silently dropped."""
    pages = [
        _page_from_synth(generate_blank(seed=30), index=0),
        _page_from_synth(generate_printed_form(seed=31), index=1),
        _page_from_synth(generate_near_blank(seed=32), index=2),
        _page_from_synth(generate_fax_degraded(seed=33), index=3),
    ]
    doc = Document(id="blank-pack", source_path="synth://pack", pages=pages)
    results = flag_blank_pages(doc)

    assert len(doc.pages) == 4
    assert [p.blank for p in doc.pages] == [True, False, True, False]
    assert results[0].reason == "blank"
    assert results[2].near_blank is True
    # Every page still has its image bound.
    assert all(p.has_image for p in doc.pages)


def test_disabled_config_leaves_flags_false() -> None:
    page = _page_from_synth(generate_blank(seed=40))
    result = apply_blank_detect(page, BlankDetectConfig(enabled=False))
    assert result.reason == "disabled"
    assert page.blank is False
