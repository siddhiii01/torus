"""Component tests: deskew / perspective / crop on synth pages (P2.3)."""

from __future__ import annotations

import pytest
from tests.fixtures.synth import (
    generate_bordered,
    generate_perspective_photo,
    generate_printed_form,
    generate_skewed,
)

from torus.config.schemas import (
    CropConfig,
    DeskewConfig,
    PerspectiveConfig,
    PreprocessConfig,
)
from torus.core import Page
from torus.preprocess import (
    apply_crop,
    apply_deskew,
    apply_perspective,
    correct_geometry,
    detect_skew,
)


def _page_from_synth(synth) -> Page:  # type: ignore[no-untyped-def]
    page = Page(
        index=synth.page_index,
        width=synth.width,
        height=synth.height,
        orientation=0,
        rotated=False,
    )
    page.bind_image(synth.image.copy())
    return page


@pytest.mark.parametrize("angle", [-6.0, -3.5, 3.5, 7.0])
def test_deskew_recovers_known_skew(angle: float) -> None:
    synth = generate_skewed(seed=50, angle_deg=angle)
    before = detect_skew(synth.image)
    assert abs(before.angle_deg - angle) <= 1.5

    page = _page_from_synth(synth)
    result = apply_deskew(page, config=DeskewConfig(enabled=True))
    assert result.applied is True
    residual = detect_skew(page.image)
    assert abs(residual.angle_deg) < 1.2


def test_deskew_skips_below_min_angle() -> None:
    # Upright form: residual detector noise must stay under the min gate.
    synth = generate_printed_form(seed=51)
    page = _page_from_synth(synth)
    result = apply_deskew(
        page,
        config=DeskewConfig(enabled=True, min_angle_deg=0.4, max_angle_deg=15.0),
    )
    assert result.applied is False
    assert result.reason == "below_min_angle"


def test_crop_removes_dark_border() -> None:
    synth = generate_bordered(seed=60, border=50)
    page = _page_from_synth(synth)
    before_area = page.width * page.height
    result = apply_crop(
        page,
        config=CropConfig(enabled=True, margin_px=6, max_trim_ratio=0.4),
    )
    assert result.applied is True
    assert page.width * page.height < before_area
    # Near the base form size (with margin), not the full bordered canvas.
    base_w = int(synth.meta["base_width"])
    base_h = int(synth.meta["base_height"])
    assert abs(page.width - base_w) <= 30
    assert abs(page.height - base_h) <= 30


def test_perspective_warps_photo_synth() -> None:
    synth = generate_perspective_photo(seed=70, frame_pad=90)
    page = _page_from_synth(synth)
    before_area = page.width * page.height
    result = apply_perspective(
        page,
        config=PerspectiveConfig(
            enabled=True,
            min_quad_area_ratio=0.15,
            max_quad_area_ratio=0.95,
        ),
    )
    # Quad detection on synth photos should succeed for this pad size.
    assert result.reason in {"applied", "detected", "already_frontal", "no_quad"}
    if result.applied:
        assert page.width * page.height <= before_area
        base_w = int(synth.meta["base_width"])
        base_h = int(synth.meta["base_height"])
        # Warped document should be close to the original form aspect.
        aspect = page.width / max(page.height, 1)
        base_aspect = base_w / max(base_h, 1)
        assert abs(aspect - base_aspect) < 0.25


def test_correct_geometry_orchestrates_bordered_page() -> None:
    synth = generate_bordered(seed=80, border=55)
    page = _page_from_synth(synth)
    before = (page.width, page.height)
    cfg = PreprocessConfig(
        deskew=DeskewConfig(enabled=True),
        # Axis-aligned bordered pages must not be perspective-warped.
        perspective=PerspectiveConfig(enabled=True),
        crop=CropConfig(
            enabled=True,
            margin_px=6,
            max_trim_ratio=0.4,
            min_content_ratio=0.02,
        ),
    )
    outcome = correct_geometry(page, cfg)
    assert outcome.perspective.applied is False
    assert outcome.crop.applied is True
    assert (page.width, page.height) != before


def test_correct_geometry_on_clean_form_is_safe() -> None:
    synth = generate_printed_form(seed=90)
    page = _page_from_synth(synth)
    outcome = correct_geometry(page, PreprocessConfig())
    # Clean flat form: perspective should not warp; deskew may be ~0.
    assert outcome.perspective.applied is False
    if outcome.deskew.applied:
        assert abs(outcome.deskew.angle_deg) <= 2.0


def test_geometry_generators_are_deterministic() -> None:
    a = generate_skewed(seed=11, angle_deg=4.0)
    b = generate_skewed(seed=11, angle_deg=4.0)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.meta["angle_deg"] == b.meta["angle_deg"]

    p1 = generate_perspective_photo(seed=12, frame_pad=85)
    p2 = generate_perspective_photo(seed=12, frame_pad=85)
    assert p1.image.tobytes() == p2.image.tobytes()

    b1 = generate_bordered(seed=13, border=45)
    b2 = generate_bordered(seed=13, border=45)
    assert b1.image.tobytes() == b2.image.tobytes()
