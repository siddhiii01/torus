"""Component tests: page quality on sharp vs degraded synthetic pages (P2.6)."""

from __future__ import annotations

from PIL import Image, ImageFilter
from tests.fixtures.synth import generate_fax_degraded, generate_printed_form
from tests.fixtures.synth.degrade import fade_degrade, fax_degrade

from torus.config.schemas import PageQualityConfig
from torus.core import Page
from torus.preprocess import apply_page_quality, score_components


def test_clean_printed_form_scores_above_fax_degraded() -> None:
    clean = generate_printed_form(seed=11)
    fax = generate_fax_degraded(seed=11, strength=0.85, base=clean)
    cfg = PageQualityConfig(target_min_side_px=min(clean.width, clean.height))

    clean_result = score_components(clean.image, cfg)
    fax_result = score_components(fax.image, cfg)

    assert clean_result.score is not None
    assert fax_result.score is not None
    assert clean_result.score > fax_result.score
    assert clean_result.blur_score is not None
    assert fax_result.blur_score is not None
    # Fax degrade applies Gaussian blur; sharpness must drop.
    assert clean_result.blur_score > fax_result.blur_score


def test_faded_page_has_lower_contrast_score() -> None:
    base = generate_printed_form(seed=22)
    faded = fade_degrade(base.image, seed=22, strength=0.8)
    cfg = PageQualityConfig(
        target_min_side_px=min(base.width, base.height),
        weight_blur=0.0,
        weight_resolution=0.0,
        weight_contrast=1.0,
    )
    clean_result = score_components(base.image, cfg)
    faded_result = score_components(faded, cfg)
    assert clean_result.contrast_score is not None
    assert faded_result.contrast_score is not None
    assert clean_result.contrast_score > faded_result.contrast_score
    assert clean_result.score is not None
    assert faded_result.score is not None
    assert clean_result.score > faded_result.score


def test_downscaled_page_has_lower_resolution_and_overall_score() -> None:
    base = generate_printed_form(seed=33)
    # Keep content; shrink pixels so resolution component falls.
    small = base.image.resize(
        (max(80, base.width // 4), max(80, base.height // 4)),
        Image.Resampling.BILINEAR,
    )
    cfg = PageQualityConfig(target_min_side_px=min(base.width, base.height))
    full = score_components(base.image, cfg)
    low = score_components(small, cfg)
    assert full.resolution_score == 1.0
    assert low.resolution_score is not None
    assert low.resolution_score < full.resolution_score
    assert full.score is not None
    assert low.score is not None
    assert full.score > low.score


def test_apply_page_quality_on_fax_flags_low_quality_when_threshold_high() -> None:
    base = generate_printed_form(seed=44)
    fax = fax_degrade(base.image, seed=44, strength=0.9)
    # Extra blur so the combined score clearly sits in the low band.
    fax = fax.filter(ImageFilter.GaussianBlur(radius=2.5))
    page = Page(index=0, width=fax.size[0], height=fax.size[1])
    page.bind_image(fax)
    result = apply_page_quality(
        page,
        PageQualityConfig(
            target_min_side_px=min(fax.size),
            low_quality_threshold=0.85,
        ),
    )
    assert result.applied is True
    assert page.quality_score is not None
    assert page.quality_score == result.score
    assert page.low_quality is True


def test_apply_page_quality_on_clean_form_not_low_quality() -> None:
    synth = generate_printed_form(seed=55)
    page = Page(index=0, width=synth.width, height=synth.height)
    page.bind_image(synth.image)
    result = apply_page_quality(
        page,
        PageQualityConfig(
            target_min_side_px=min(synth.width, synth.height),
            low_quality_threshold=0.4,
        ),
    )
    assert result.applied is True
    assert page.quality_score is not None
    assert page.quality_score >= 0.4
    assert page.low_quality is False
