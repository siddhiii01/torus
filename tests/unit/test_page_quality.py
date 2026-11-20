"""Unit tests for page quality scoring (P2.6)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from torus.config import PageQualityConfig, load_config
from torus.config.schemas import PageQualityConfig as PageQualitySchema
from torus.core import Page
from torus.preprocess.page_quality import (
    apply_page_quality,
    measure_blur,
    measure_page_contrast,
    measure_resolution,
    score_components,
)


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _sharp_text_image(size: tuple[int, int] = (320, 240)) -> Image.Image:
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for y in range(20, size[1] - 20, 18):
        draw.line((20, y, size[0] - 20, y), fill=(20, 20, 20), width=2)
        draw.text((24, y - 10), "AB12 XY98", fill=(10, 10, 10))
    return image


def test_repo_pipeline_loads_page_quality_defaults() -> None:
    cfg = load_config()
    quality = cfg.pipeline.preprocess.page_quality
    assert quality.enabled is True
    assert quality.weight_blur == 1.0
    assert quality.weight_contrast == 1.0
    assert quality.weight_resolution == 1.0
    assert quality.blur_laplacian_ref == 40.0
    assert quality.contrast_ref == 18.0
    assert quality.target_min_side_px == 1600
    assert quality.low_quality_threshold == 0.4


def test_zero_weights_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PageQualityConfig(
            weight_blur=0.0,
            weight_contrast=0.0,
            weight_resolution=0.0,
        )


def test_measure_blur_higher_on_sharp_than_blurred() -> None:
    sharp = _sharp_text_image()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3.0))
    assert measure_blur(_gray(sharp)) > measure_blur(_gray(blurred))


def test_measure_contrast_higher_on_ink_than_flat() -> None:
    sharp = _sharp_text_image()
    flat = Image.new("RGB", (320, 240), color=(128, 128, 128))
    assert measure_page_contrast(_gray(sharp)) > measure_page_contrast(_gray(flat))


def test_measure_resolution_uses_short_side() -> None:
    assert measure_resolution(2000, 1500) == 1500
    assert measure_resolution(0, 100) == 0


def test_score_components_disabled() -> None:
    image = _sharp_text_image()
    result = score_components(image, PageQualityConfig(enabled=False))
    assert result.applied is False
    assert result.reason == "disabled"
    assert result.score is None


def test_score_components_too_small_is_low_quality() -> None:
    tiny = Image.new("RGB", (1, 1), color=(255, 255, 255))
    result = score_components(tiny)
    assert result.applied is True
    assert result.score == 0.0
    assert result.low_quality is True
    assert result.reason == "too_small"


def test_blurred_page_scores_lower_than_sharp() -> None:
    sharp = _sharp_text_image((640, 480))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4.0))
    # Hold resolution constant so the drop is from blur (and some contrast).
    cfg = PageQualityConfig(
        target_min_side_px=480,
        weight_resolution=0.0,
        weight_contrast=0.0,
        weight_blur=1.0,
    )
    sharp_score = score_components(sharp, cfg)
    blur_score = score_components(blurred, cfg)
    assert sharp_score.score is not None
    assert blur_score.score is not None
    assert sharp_score.score > blur_score.score
    assert sharp_score.blur_score is not None
    assert blur_score.blur_score is not None
    assert sharp_score.blur_score > blur_score.blur_score


def test_low_resolution_lowers_resolution_score() -> None:
    large = _sharp_text_image((2000, 1800))
    small = large.resize((400, 300), Image.Resampling.BILINEAR)
    cfg = PageQualityConfig(
        target_min_side_px=1600,
        weight_blur=0.0,
        weight_contrast=0.0,
        weight_resolution=1.0,
    )
    large_result = score_components(large, cfg)
    small_result = score_components(small, cfg)
    assert large_result.resolution_score == 1.0
    assert small_result.resolution_score is not None
    assert small_result.resolution_score < 0.3
    assert large_result.score is not None
    assert small_result.score is not None
    assert large_result.score > small_result.score


def test_apply_page_quality_sets_page_fields() -> None:
    image = _sharp_text_image((1800, 1400))
    page = Page(index=0, width=1800, height=1400)
    page.bind_image(image)
    result = apply_page_quality(page, PageQualityConfig(target_min_side_px=1400))
    assert result.applied is True
    assert page.quality_score is not None
    assert page.quality_score == result.score
    assert page.low_quality is False
    assert page.low_quality == result.low_quality


def test_apply_page_quality_skips_when_disabled() -> None:
    page = Page(index=0, width=100, height=100, quality_score=None, low_quality=False)
    page.bind_image(Image.new("RGB", (100, 100), color=(200, 200, 200)))
    result = apply_page_quality(page, PageQualityConfig(enabled=False))
    assert result.applied is False
    assert page.quality_score is None
    assert page.low_quality is False


def test_apply_page_quality_no_image() -> None:
    page = Page(index=0, width=100, height=100)
    result = apply_page_quality(page)
    assert result.applied is False
    assert result.reason == "no_image"
    assert page.quality_score is None


def test_low_quality_threshold_flags_page() -> None:
    flat = Image.new("RGB", (200, 150), color=(140, 140, 140))
    page = Page(index=0, width=200, height=150)
    page.bind_image(flat)
    result = apply_page_quality(
        page,
        PageQualityConfig(
            target_min_side_px=1600,
            low_quality_threshold=0.9,
        ),
    )
    assert result.applied is True
    assert page.quality_score is not None
    assert page.quality_score < 0.9
    assert page.low_quality is True


def test_page_quality_schema_defaults_match_factory() -> None:
    cfg = PageQualitySchema()
    assert cfg.blur_laplacian_ref == 40.0
    assert cfg.contrast_ref == 18.0
    assert cfg.target_min_side_px == 1600
