"""Unit tests for deskew / perspective / crop helpers (P2.3)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from torus.config.schemas import CropConfig, DeskewConfig, PerspectiveConfig
from torus.preprocess.crop import detect_content_box
from torus.preprocess.deskew import deskew_image, detect_skew
from torus.preprocess.perspective import detect_perspective, warp_perspective


def _ruled_page(
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
        draw.line((30, y + 28, width - 30, y + 28), fill=(40, 40, 40), width=1)
        y += 48
    return image


def test_detect_skew_near_known_angle() -> None:
    upright = _ruled_page()
    # Positive angle_deg = clockwise content skew → Pillow rotate is CCW.
    created = upright.rotate(-5.0, expand=True, fillcolor=(255, 255, 255)).convert(
        "RGB"
    )
    result = detect_skew(created)
    assert abs(result.angle_deg - 5.0) <= 1.5


def test_detect_skew_disabled() -> None:
    image = _ruled_page()
    result = detect_skew(image, DeskewConfig(enabled=False))
    assert result.applied is False
    assert result.reason == "disabled"
    assert result.angle_deg == 0.0


def test_deskew_image_roundtrip_size() -> None:
    image = _ruled_page()
    skewed = image.rotate(-4.0, expand=True, fillcolor=(255, 255, 255)).convert("RGB")
    corrected = deskew_image(skewed, 4.0)
    assert corrected.size[1] >= corrected.size[0]


def test_detect_content_box_on_dark_border() -> None:
    inner = _ruled_page(width=200, height=280)
    bordered = Image.new("RGB", (280, 360), color=(20, 20, 22))
    bordered.paste(inner, (40, 40))
    result = detect_content_box(bordered, CropConfig(margin_px=4, max_trim_ratio=0.4))
    assert result.reason == "detected"
    assert result.box is not None
    x0, y0, x1, y1 = result.box
    assert x0 <= 44
    assert y0 <= 44
    assert x1 >= 236
    assert y1 >= 316
    assert (x1 - x0) * (y1 - y0) < 280 * 360


def test_detect_perspective_runs_on_framed_page() -> None:
    inner = _ruled_page(width=300, height=420)
    canvas = Image.new("RGB", (500, 640), color=(30, 30, 30))
    canvas.paste(inner, (100, 110))
    result = detect_perspective(canvas)
    assert result.reason in {
        "detected",
        "already_frontal",
        "quad_fills_frame",
        "no_quad",
        "quad_too_small",
    }


def test_warp_perspective_changes_size() -> None:
    image = _ruled_page(width=200, height=280)
    quad = (
        (10.0, 20.0),
        (190.0, 10.0),
        (195.0, 270.0),
        (5.0, 260.0),
    )
    warped = warp_perspective(image, quad)
    assert warped.mode == "RGB"
    assert warped.size[0] > 50
    assert warped.size[1] > 50


def test_perspective_disabled() -> None:
    image = _ruled_page()
    result = detect_perspective(image, PerspectiveConfig(enabled=False))
    assert result.reason == "disabled"
    assert result.applied is False
