"""Unit tests for quality-repair steps and config switches (P2.4)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from torus.config import (
    ContrastNormConfig,
    DenoiseConfig,
    QualityRepairConfig,
    StrokeThickenConfig,
    load_config,
)
from torus.config.schemas import QualityRepairConfig as QualityRepairSchema
from torus.preprocess.quality_repair import (
    measure_contrast,
    measure_noise,
    measure_stroke_width,
    repair_image,
)


def _step(result: object, name: str) -> object:
    for step in result.steps:  # type: ignore[attr-defined]
        if step.name == name:
            return step
    raise AssertionError(f"missing step {name!r}")


def test_repo_pipeline_loads_quality_repair_defaults() -> None:
    cfg = load_config()
    repair = cfg.pipeline.preprocess.quality_repair
    assert repair.enabled is True
    assert repair.denoise.enabled is True
    assert repair.contrast_norm.enabled is True
    assert repair.stroke_thicken.enabled is True
    assert repair.denoise.template_window_size % 2 == 1
    assert repair.denoise.search_window_size % 2 == 1


def test_even_denoise_window_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        DenoiseConfig(template_window_size=6)


def test_master_disabled_is_identity() -> None:
    image = Image.new("RGB", (64, 48), color=(200, 200, 200))
    before = np.asarray(image)
    repaired, result = repair_image(image, QualityRepairConfig(enabled=False))
    assert result.applied is False
    assert result.image_changed is False
    assert result.reason == "disabled"
    assert result.steps == ()
    assert np.array_equal(np.asarray(repaired), before)


def test_each_step_can_be_switched_off() -> None:
    image = Image.new("RGB", (80, 60), color=(180, 180, 180))
    # Draw a dark line so thicken/contrast have content.
    arr = np.asarray(image).copy()
    arr[20:40, 10:70, :] = 40
    image = Image.fromarray(arr, mode="RGB")

    cfg = QualityRepairConfig(
        enabled=True,
        denoise=DenoiseConfig(enabled=False),
        contrast_norm=ContrastNormConfig(enabled=True),
        stroke_thicken=StrokeThickenConfig(enabled=False),
    )
    _repaired, result = repair_image(image, cfg)
    assert result.applied is True
    assert _step(result, "denoise").applied is False  # type: ignore[attr-defined]
    assert _step(result, "contrast_norm").applied is True  # type: ignore[attr-defined]
    assert _step(result, "stroke_thicken").applied is False  # type: ignore[attr-defined]


def test_all_steps_disabled_reports_reason() -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    cfg = QualityRepairConfig(
        enabled=True,
        denoise=DenoiseConfig(enabled=False),
        contrast_norm=ContrastNormConfig(enabled=False),
        stroke_thicken=StrokeThickenConfig(enabled=False),
    )
    repaired, result = repair_image(image, cfg)
    assert result.applied is False
    assert result.image_changed is False
    assert result.reason == "all_steps_disabled"
    assert np.array_equal(np.asarray(repaired), np.asarray(image))


def test_metrics_helpers_on_empty_and_flat() -> None:
    empty = np.zeros((0, 0), dtype=np.uint8)
    assert measure_contrast(empty) == 0.0
    assert measure_noise(empty) == 0.0
    assert measure_stroke_width(empty) == 0.0

    flat = np.full((20, 20), 128, dtype=np.uint8)
    assert measure_contrast(flat) == 0.0


def test_quality_repair_schema_defaults_match_factory() -> None:
    cfg = QualityRepairSchema()
    assert cfg.denoise.h == 8.0
    assert cfg.contrast_norm.clip_limit == 2.0
    assert cfg.stroke_thicken.kernel_size == 2
