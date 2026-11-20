"""Unit tests for orientation helpers (P2.2)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from torus.core.types import OrientationDegrees
from torus.preprocess.orientation import (
    OrientationResult,
    _candidates_for_reading,
    _rotate_cw,
    classic_orientation,
    classifier_orientation,
    detect_orientation,
    upright_image,
)


def _text_page(
    *,
    width: int = 400,
    height: int = 560,
    lines: tuple[str, ...] = ("ALPHA LINE", "BETA VALUE", "GAMMA CODE"),
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(image)
    y = 40
    for line in lines:
        draw.text((36, y), line, fill=(10, 10, 10))
        y += 48
    return image


def test_rotate_cw_roundtrip_sizes() -> None:
    image = Image.new("RGB", (100, 200), color=(255, 255, 255))
    rotated = _rotate_cw(image, 90)
    assert rotated.size == (200, 100)
    back = _rotate_cw(rotated, 270)
    assert back.size == (100, 200)


def test_candidates_for_reading_agreed() -> None:
    assert _candidates_for_reading(0, 0) == (0,)
    assert _candidates_for_reading(90, 90) == (90,)


def test_candidates_for_reading_disagree_keeps_both() -> None:
    assert _candidates_for_reading(0, 180) == (0, 180)
    assert _candidates_for_reading(90, 270) == (90, 270)


def test_upright_image_undoes_cw_rotation() -> None:
    upright = _text_page()
    rotated = _rotate_cw(upright, 90)
    restored = upright_image(rotated, 90)
    assert restored.size == upright.size


def test_classic_and_classifier_on_upright_text() -> None:
    image = _text_page()
    classic, _ = classic_orientation(image)
    classifier, _ = classifier_orientation(image)
    assert classic == 0
    assert classifier == 0


def test_detect_orientation_result_shape() -> None:
    image = _text_page()
    result = detect_orientation(image)
    assert isinstance(result, OrientationResult)
    assert result.classic in (0, 90, 180, 270)
    assert result.classifier in (0, 90, 180, 270)
    assert result.agreed == (result.classic == result.classifier)
    assert len(result.candidates_for_reading) >= 1
    for degrees in result.candidates_for_reading:
        assert degrees in (0, 90, 180, 270)


def test_forced_disagreement_exposes_both_candidates() -> None:
    """Merge logic keeps both orientations when detectors disagree."""
    classic: OrientationDegrees = 90
    classifier: OrientationDegrees = 270
    candidates = _candidates_for_reading(classic, classifier)
    assert candidates == (90, 270)
    result = OrientationResult(
        classic=classic,
        classifier=classifier,
        classic_score=0.8,
        classifier_score=0.7,
        agreed=False,
        candidates_for_reading=candidates,
    )
    assert result.agreed is False
    assert result.candidates_for_reading == (90, 270)
