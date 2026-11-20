"""Component tests: orientation on synthetic rotated pages (P2.2)."""

from __future__ import annotations

import pytest
from tests.fixtures.synth import (
    generate_handwriting_page,
    generate_printed_form,
    generate_rotated,
)

from torus.core import Page
from torus.preprocess import apply_orientation, detect_orientation
from torus.preprocess.orientation import OrientationResult


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_orientation_detects_printed_form_rotation(degrees: int) -> None:
    synth = generate_rotated(seed=11, degrees=degrees)  # type: ignore[arg-type]
    result = detect_orientation(synth.image)
    assert result.classic == degrees
    assert result.classifier == degrees
    assert result.agreed is True
    assert result.candidates_for_reading == (degrees,)


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_orientation_detects_handwriting_rotation(degrees: int) -> None:
    base = generate_handwriting_page(seed=21)
    synth = generate_rotated(seed=21, degrees=degrees, base=base)  # type: ignore[arg-type]
    result = detect_orientation(synth.image)
    assert result.classic == degrees
    assert result.classifier == degrees
    assert result.agreed is True


@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_apply_orientation_uprights_when_agreed(degrees: int) -> None:
    synth = generate_rotated(seed=31, degrees=degrees)  # type: ignore[arg-type]
    page = Page(
        index=0,
        width=synth.width,
        height=synth.height,
        orientation=0,
        rotated=False,
    )
    page.bind_image(synth.image.copy())
    result = detect_orientation(page.image)
    assert result.agreed is True
    assert result.classic == degrees

    applied = apply_orientation(page, result)
    assert applied.agreed is True
    assert page.orientation == 0
    assert page.rotated is True
    # Corrected page should match the upright base aspect (printed form 800x1100).
    base = generate_printed_form(seed=31)
    assert page.width == base.width
    assert page.height == base.height
    assert page.image.size == (base.width, base.height)


def test_apply_orientation_skips_pixels_when_disagreed() -> None:
    synth = generate_rotated(seed=41, degrees=90)
    page = Page(
        index=0,
        width=synth.width,
        height=synth.height,
        orientation=0,
        rotated=False,
    )
    original = synth.image.copy()
    page.bind_image(original)
    disagreed = OrientationResult(
        classic=90,
        classifier=270,
        classic_score=0.9,
        classifier_score=0.8,
        agreed=False,
        candidates_for_reading=(90, 270),
    )
    apply_orientation(page, disagreed)
    assert page.orientation == 0
    assert page.rotated is False
    assert page.width == synth.width
    assert page.height == synth.height
    assert page.image.size == original.size
