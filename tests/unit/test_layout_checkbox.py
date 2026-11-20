"""Unit tests for checkbox / radio detect, fill, and label mapping (P4.4)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from torus.config import LayoutCheckboxConfig, LayoutConfig, load_config
from torus.core import BBox, Region, RegionKind
from torus.layout.checkbox import (
    CheckboxFillState,
    detect_checkbox_regions,
    detect_checkboxes,
    map_checkbox_labels,
    measure_checkbox_fill,
    merge_checkboxes_into_regions,
)


def test_layout_checkbox_config_loads() -> None:
    cfg = load_config().pipeline.layout
    assert isinstance(cfg, LayoutConfig)
    assert cfg.checkbox.enabled is True
    assert cfg.checkbox.min_side_ratio < cfg.checkbox.max_side_ratio


def test_detect_disabled_returns_empty() -> None:
    img = Image.new("RGB", (200, 80), color=(255, 255, 255))
    assert (
        detect_checkbox_regions(img, config=LayoutCheckboxConfig(enabled=False)) == []
    )


def test_measure_empty_vs_cross_on_drawn_box() -> None:
    img = Image.new("RGB", (120, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 50, 50), outline=(0, 0, 0), width=2)
    empty = Region(
        id="e",
        page_index=0,
        bbox=BBox(x0=20 / 120, y0=20 / 80, x1=50 / 120, y1=50 / 80),
        kind=RegionKind.CHECKBOX,
    )
    ratio_e, state_e = measure_checkbox_fill(img, empty)
    assert state_e is CheckboxFillState.EMPTY
    assert ratio_e < 0.1

    draw.line((24, 24, 46, 46), fill=(0, 0, 0), width=3)
    draw.line((24, 46, 46, 24), fill=(0, 0, 0), width=3)
    ratio_c, state_c = measure_checkbox_fill(img, empty)
    assert state_c is CheckboxFillState.CROSSED
    assert ratio_c > ratio_e


def test_measure_tick_on_drawn_box() -> None:
    img = Image.new("RGB", (120, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 50, 50), outline=(0, 0, 0), width=2)
    draw.line((24, 35, 32, 46), fill=(0, 0, 0), width=3)
    draw.line((32, 46, 46, 24), fill=(0, 0, 0), width=3)
    region = Region(
        id="t",
        page_index=0,
        bbox=BBox(x0=20 / 120, y0=20 / 80, x1=50 / 120, y1=50 / 80),
        kind=RegionKind.CHECKBOX,
    )
    _ratio, state = measure_checkbox_fill(img, region)
    assert state is CheckboxFillState.TICKED


def test_map_checkbox_labels_prefers_right() -> None:
    box = Region(
        id="cb",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.15, y1=0.28),
        kind=RegionKind.CHECKBOX,
    )
    right = Region(
        id="right",
        page_index=0,
        bbox=BBox(x0=0.18, y0=0.21, x1=0.4, y1=0.27),
        kind=RegionKind.TEXT_LINE,
    )
    left = Region(
        id="left",
        page_index=0,
        bbox=BBox(x0=0.01, y0=0.21, x1=0.08, y1=0.27),
        kind=RegionKind.TEXT_LINE,
    )
    from torus.layout.checkbox import CheckboxDetection

    det = CheckboxDetection(
        region=box,
        shape="square",
        fill_ratio=0.0,
        state=CheckboxFillState.EMPTY,
    )
    mapped = map_checkbox_labels([det], [left, right])
    assert mapped[0].region.label_anchor is not None
    assert mapped[0].region.label_anchor.bbox == right.bbox


def test_merge_checkboxes_dedupes() -> None:
    a = Region(
        id="a",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
        kind=RegionKind.CHECKBOX,
    )
    b = Region(
        id="b",
        page_index=0,
        bbox=BBox(x0=0.11, y0=0.11, x1=0.19, y1=0.19),
        kind=RegionKind.CHECKBOX,
    )
    line = Region(
        id="l",
        page_index=0,
        bbox=BBox(x0=0.5, y0=0.5, x1=0.7, y1=0.55),
        kind=RegionKind.TEXT_LINE,
    )
    merged = merge_checkboxes_into_regions([line], [a, b], nms_iou=0.4)
    assert sum(1 for r in merged if r.kind == RegionKind.CHECKBOX) == 1
    assert any(r.kind == RegionKind.TEXT_LINE for r in merged)


def test_detect_square_on_blank_canvas_with_box() -> None:
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 60, 72, 92), outline=(10, 10, 10), width=2)
    found = detect_checkboxes(img, config=LayoutCheckboxConfig())
    assert found, "expected at least one checkbox"
    assert any(d.shape == "square" for d in found)
    assert any(d.state is CheckboxFillState.EMPTY for d in found)


def test_diagonal_helpers_stable_on_tiny() -> None:
    # Smoke: empty tiny image should not crash detection.
    img = Image.new("RGB", (20, 20), color=(255, 255, 255))
    assert detect_checkbox_regions(img) == []
    _ = np.zeros((2, 2), dtype=np.uint8)
