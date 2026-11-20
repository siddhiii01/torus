"""Component tests: quality repair on fax / faded synthetic pages (P2.4)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from tests.fixtures.synth import (
    generate_faded,
    generate_fax_degraded,
    generate_printed_form,
)
from tests.fixtures.synth.degrade import fade_degrade, fax_degrade

from torus.config.schemas import (
    ContrastNormConfig,
    DenoiseConfig,
    QualityRepairConfig,
    StrokeThickenConfig,
)
from torus.core import Page
from torus.preprocess import apply_quality_repair, repair_image
from torus.preprocess.quality_repair import (
    measure_contrast,
    measure_noise,
    measure_stroke_width,
)


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _step_by_name(result: object, name: str) -> object:
    for step in result.steps:  # type: ignore[attr-defined]
        if step.name == name:
            return step
    raise AssertionError(f"missing step {name!r}")


def test_denoise_reduces_noise_on_fax_page() -> None:
    synth = generate_fax_degraded(seed=101, strength=0.8)
    cfg = QualityRepairConfig(
        enabled=True,
        denoise=DenoiseConfig(enabled=True, h=12.0),
        contrast_norm=ContrastNormConfig(enabled=False),
        stroke_thicken=StrokeThickenConfig(enabled=False),
    )
    before = measure_noise(_gray(synth.image))
    repaired, result = repair_image(synth.image, cfg)
    after = measure_noise(_gray(repaired))
    step = _step_by_name(result, "denoise")
    assert step.applied is True  # type: ignore[attr-defined]
    assert after < before
    assert step.noise_after < step.noise_before  # type: ignore[attr-defined]


def test_contrast_norm_improves_faded_page() -> None:
    synth = generate_faded(seed=202, strength=0.75)
    cfg = QualityRepairConfig(
        enabled=True,
        denoise=DenoiseConfig(enabled=False),
        contrast_norm=ContrastNormConfig(enabled=True, clip_limit=3.0),
        stroke_thicken=StrokeThickenConfig(enabled=False),
    )
    before = measure_contrast(_gray(synth.image))
    repaired, result = repair_image(synth.image, cfg)
    after = measure_contrast(_gray(repaired))
    step = _step_by_name(result, "contrast_norm")
    assert step.applied is True  # type: ignore[attr-defined]
    assert after > before
    assert step.contrast_after > step.contrast_before  # type: ignore[attr-defined]


def test_stroke_thicken_increases_stroke_width() -> None:
    # Thin dark strokes on white — thickening should raise mean DT radius.
    canvas = Image.new("RGB", (200, 120), color=(255, 255, 255))
    arr = np.asarray(canvas).copy()
    arr[40:42, 20:180, :] = 20
    arr[70:72, 30:170, :] = 20
    thin = Image.fromarray(arr, mode="RGB")

    cfg = QualityRepairConfig(
        enabled=True,
        denoise=DenoiseConfig(enabled=False),
        contrast_norm=ContrastNormConfig(enabled=False),
        stroke_thicken=StrokeThickenConfig(enabled=True, kernel_size=3),
    )
    before = measure_stroke_width(_gray(thin))
    repaired, result = repair_image(thin, cfg)
    after = measure_stroke_width(_gray(repaired))
    step = _step_by_name(result, "stroke_thicken")
    assert step.applied is True  # type: ignore[attr-defined]
    assert after > before
    assert step.stroke_width_after > step.stroke_width_before  # type: ignore[attr-defined]


def test_full_repair_pipeline_on_fax_is_measured() -> None:
    base = generate_printed_form(seed=303)
    fax = fax_degrade(base.image, seed=303, strength=0.7)
    repaired, result = repair_image(fax, QualityRepairConfig())
    assert result.applied is True
    assert len(result.steps) == 3
    assert all(s.applied for s in result.steps)
    assert all(s.elapsed_ms >= 0.0 for s in result.steps)
    # Full stack should change pixels on a noisy fax page.
    assert result.image_changed is True
    assert not np.array_equal(np.asarray(repaired), np.asarray(fax))


def test_full_repair_pipeline_on_faded_is_measured() -> None:
    base = generate_printed_form(seed=404)
    faded = fade_degrade(base.image, seed=404, strength=0.7)
    repaired, result = repair_image(faded, QualityRepairConfig())
    assert result.applied is True
    assert result.image_changed is True
    contrast_step = _step_by_name(result, "contrast_norm")
    # Later steps may move global contrast; the CLAHE step itself must help.
    assert contrast_step.contrast_after > contrast_step.contrast_before  # type: ignore[attr-defined]
    denoise_step = _step_by_name(result, "denoise")
    assert denoise_step.applied is True  # type: ignore[attr-defined]
    thicken_step = _step_by_name(result, "stroke_thicken")
    assert thicken_step.applied is True  # type: ignore[attr-defined]


def test_apply_quality_repair_updates_page() -> None:
    synth = generate_fax_degraded(seed=505, strength=0.75)
    page = Page(
        index=0,
        width=synth.width,
        height=synth.height,
        orientation=0,
        rotated=False,
        low_quality=False,
    )
    original = synth.image.copy()
    page.bind_image(original)
    result = apply_quality_repair(page, QualityRepairConfig())
    assert result.applied is True
    assert result.image_changed is True
    assert not np.array_equal(np.asarray(page.image), np.asarray(original))
    assert page.width == page.image.size[0]
    assert page.height == page.image.size[1]


def test_apply_quality_repair_skips_when_disabled() -> None:
    synth = generate_fax_degraded(seed=606, strength=0.6)
    page = Page(
        index=0,
        width=synth.width,
        height=synth.height,
        orientation=0,
        rotated=False,
    )
    original = synth.image.copy()
    page.bind_image(original)
    result = apply_quality_repair(page, QualityRepairConfig(enabled=False))
    assert result.applied is False
    assert result.reason == "disabled"
    assert np.array_equal(np.asarray(page.image), np.asarray(original))


def test_faded_generator_preserves_geometry() -> None:
    base = generate_printed_form(seed=707)
    faded = generate_faded(seed=707, strength=0.6, base=base)
    assert faded.width == base.width
    assert faded.height == base.height
    assert len(faded.regions) == len(base.regions)
    for left, right in zip(faded.regions, base.regions, strict=True):
        assert left.region.bbox == right.region.bbox
