"""Unit tests for seeded synthetic fixture generators (P0.7)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.fixtures.synth import (
    generate,
    generate_checkbox,
    generate_comb_row,
    generate_faded,
    generate_fax_degraded,
    generate_handwriting,
    generate_printed_form,
    generate_rotated,
    generate_signature,
    generate_stamp,
)
from tests.fixtures.synth.geometry import pixel_bbox_to_norm, rotate_norm_bbox
from torus.core import BBox, RegionKind


def _image_digest(image: object) -> str:
    raw = image.tobytes()  # type: ignore[attr-defined]
    return hashlib.sha256(raw).hexdigest()


def _assert_regions_valid(page: object) -> None:
    assert page.width == page.image.size[0]  # type: ignore[attr-defined]
    assert page.height == page.image.size[1]  # type: ignore[attr-defined]
    assert page.regions  # type: ignore[attr-defined]
    for item in page.regions:  # type: ignore[attr-defined]
        bbox = item.region.bbox
        assert 0.0 <= bbox.x0 < bbox.x1 <= 1.0
        assert 0.0 <= bbox.y0 < bbox.y1 <= 1.0


@pytest.mark.parametrize(
    "factory",
    [
        generate_printed_form,
        generate_comb_row,
        generate_checkbox,
        generate_handwriting,
        generate_stamp,
        generate_signature,
        generate_rotated,
        generate_fax_degraded,
        generate_faded,
    ],
)
def test_generators_are_deterministic(factory: object) -> None:
    a = factory(seed=7)  # type: ignore[operator]
    b = factory(seed=7)  # type: ignore[operator]
    assert _image_digest(a.image) == _image_digest(b.image)
    assert a.width == b.width and a.height == b.height
    assert len(a.regions) == len(b.regions)
    for left, right in zip(a.regions, b.regions, strict=True):
        assert left.region.bbox == right.region.bbox
        assert left.text == right.text
        assert left.checked == right.checked


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_different_seeds_differ(seed: int) -> None:
    a = generate_printed_form(seed=seed)
    b = generate_printed_form(seed=seed + 1)
    assert _image_digest(a.image) != _image_digest(b.image)


def test_comb_row_ground_truth() -> None:
    page = generate_comb_row(
        seed=3, cells=6, text="ABC123", style="ticks", crossing_strokes=False
    )
    _assert_regions_valid(page)
    assert len(page.regions) == 1
    truth = page.regions[0]
    assert truth.text == "ABC123"
    assert truth.region.kind == RegionKind.COMB
    assert truth.region.label_anchor is not None
    assert truth.region.label_anchor.text == "CODE"
    assert page.meta["cells"] == 6
    assert page.meta["style"] == "ticks"
    assert page.meta["crossing_strokes"] is False
    assert len(page.meta["cell_bboxes_norm"]) == 6
    assert page.meta["cell_chars"] == list("ABC123")


def test_checkbox_checked_state() -> None:
    on = generate_checkbox(seed=1, checked=True, label="Option A", shape="square")
    off = generate_checkbox(seed=1, checked=False, label="Option A", shape="square")
    _assert_regions_valid(on)
    assert on.regions[0].checked is True
    assert on.regions[0].text == "crossed"
    assert off.regions[0].checked is False
    assert off.regions[0].text == "false"
    assert on.regions[0].region.kind == RegionKind.CHECKBOX
    assert on.meta["mark"] == "cross"
    assert any(r.region.kind == RegionKind.TEXT_LINE for r in on.regions)


def test_printed_form_has_field_values() -> None:
    page = generate_printed_form(seed=11, field_count=3)
    _assert_regions_valid(page)
    assert len(page.regions) == 3
    assert all(r.region.kind == RegionKind.FIELD_VALUE for r in page.regions)
    assert all(r.text for r in page.regions)


def test_handwriting_returns_phrase() -> None:
    page = generate_handwriting(seed=4, text="unit nine")
    _assert_regions_valid(page)
    assert page.regions[0].text == "unit nine"
    assert page.regions[0].region.kind == RegionKind.TEXT_LINE


def test_handwriting_page_has_multiple_lines() -> None:
    from tests.fixtures.synth import generate_handwriting_page

    page = generate_handwriting_page(seed=6)
    _assert_regions_valid(page)
    assert len(page.regions) == 3
    assert all(r.region.kind == RegionKind.TEXT_LINE for r in page.regions)


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_rotation_remaps_bbox(degrees: int) -> None:
    base = generate_comb_row(seed=5, cells=5, text="HELLO")
    page = generate_rotated(seed=5, degrees=degrees, base=base)  # type: ignore[arg-type]
    _assert_regions_valid(page)
    assert page.orientation == degrees
    if degrees in (90, 270):
        assert page.width == base.height
        assert page.height == base.width
    else:
        assert page.width == base.width
        assert page.height == base.height


def test_fax_degraded_preserves_geometry() -> None:
    base = generate_printed_form(seed=8, field_count=2)
    page = generate_fax_degraded(seed=8, strength=0.7, base=base)
    _assert_regions_valid(page)
    assert page.width == base.width and page.height == base.height
    assert page.regions == base.regions
    assert _image_digest(page.image) != _image_digest(base.image)


def test_blank_and_near_blank_generators() -> None:
    from tests.fixtures.synth import generate_blank, generate_near_blank

    blank = generate_blank(seed=19)
    assert blank.kind == "blank"
    assert blank.regions == ()
    assert blank.meta["near_blank"] is False
    assert blank.width > 0 and blank.height > 0

    near = generate_near_blank(seed=20, speck_count=3)
    assert near.kind == "blank"
    assert near.meta["near_blank"] is True
    assert near.meta["speck_count"] == 3
    assert _image_digest(near.image) != _image_digest(blank.image)


def test_skewed_perspective_bordered_generators() -> None:
    from tests.fixtures.synth import (
        generate_bordered,
        generate_perspective_photo,
        generate_skewed,
    )

    skewed = generate_skewed(seed=14, angle_deg=-5.0)
    assert skewed.kind == "skewed"
    assert skewed.meta["angle_deg"] == -5.0
    assert skewed.width > 0 and skewed.height > 0

    photo = generate_perspective_photo(seed=15, frame_pad=70)
    assert photo.kind == "perspective_photo"
    assert photo.width > int(photo.meta["base_width"])
    assert photo.height > int(photo.meta["base_height"])

    bordered = generate_bordered(seed=16, border=40)
    assert bordered.kind == "bordered"
    assert bordered.width == int(bordered.meta["base_width"]) + 80
    assert bordered.height == int(bordered.meta["base_height"]) + 80


def test_faded_preserves_geometry() -> None:
    base = generate_printed_form(seed=18, field_count=2)
    page = generate_faded(seed=18, strength=0.65, base=base)
    _assert_regions_valid(page)
    assert page.width == base.width and page.height == base.height
    assert page.regions == base.regions
    assert _image_digest(page.image) != _image_digest(base.image)


def test_stamp_adds_stamp_region() -> None:
    page = generate_stamp(seed=2, stamp_text="RECEIVED")
    _assert_regions_valid(page)
    stamps = [r for r in page.regions if r.region.kind == RegionKind.STAMP]
    assert len(stamps) == 1
    assert stamps[0].text == "RECEIVED"


def test_generate_dispatch() -> None:
    page = generate("comb_row", seed=9, cells=4, text="WXYZ")
    assert page.kind == "comb_row"
    assert page.regions[0].text == "WXYZ"


def test_pixel_bbox_to_norm_and_rotate_helpers() -> None:
    bbox = pixel_bbox_to_norm(x0=10, y0=20, x1=110, y1=60, width=200, height=100)
    assert bbox == BBox(x0=0.05, y0=0.2, x1=0.55, y1=0.6)
    rotated, w, h = rotate_norm_bbox(bbox, degrees=90, src_width=200, src_height=100)
    assert w == 100 and h == 200
    assert 0.0 <= rotated.x0 < rotated.x1 <= 1.0
    assert 0.0 <= rotated.y0 < rotated.y1 <= 1.0


def test_ingest_file_writers_create_bytes(tmp_path: Path) -> None:
    """S8: on-disk ingest helpers write files under a temp dir only."""
    from tests.fixtures.synth import (
        write_synth_jpeg_with_exif_orientation,
        write_synth_multipage_tiff,
        write_synth_pdf_with_text,
        write_synth_png,
    )

    png = write_synth_png(tmp_path / "a.png")
    jpg = write_synth_jpeg_with_exif_orientation(tmp_path / "b.jpg")
    tiff = write_synth_multipage_tiff(tmp_path / "c.tiff", page_count=2)
    pdf = write_synth_pdf_with_text(tmp_path / "d.pdf")
    for path in (png, jpg, tiff, pdf):
        assert path.is_file()
        assert path.stat().st_size > 0
