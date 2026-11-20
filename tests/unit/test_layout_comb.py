"""Unit tests for comb detect / inpaint / split helpers (P4.3)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from torus.config import LayoutCombConfig, LayoutConfig, load_config
from torus.core import BBox, Region, RegionKind
from torus.layout.comb import (
    _protect_crossing_strokes,
    detect_comb_regions,
    inpaint_comb_lines,
    merge_combs_into_regions,
    split_comb_cells,
)


def test_layout_comb_config_loads() -> None:
    cfg = load_config().pipeline.layout
    assert isinstance(cfg, LayoutConfig)
    assert cfg.comb.enabled is True
    assert cfg.comb.min_cells >= 2


def test_protect_crossing_strokes_keeps_bridge() -> None:
    # Vertical line at x=5; horizontal ink bridge across it.
    ink = np.zeros((11, 11), dtype=np.uint8)
    ink[:, 5] = 255
    ink[5, :] = 255
    v_line = np.zeros((11, 11), dtype=np.uint8)
    v_line[:, 5] = 255
    protected = _protect_crossing_strokes(
        ink, v_line, radius=2, vertical_mask=v_line, horizontal_mask=None
    )
    # Crossing pixel at (5,5) must not be inpainted.
    assert protected[5, 5] == 0
    # Pure vertical-ruling pixels away from the bridge should remain.
    assert protected[1, 5] == 255
    assert protected[2, 5] == 255


def test_merge_prefers_comb_over_table_cell() -> None:
    comb = Region(
        id="c1",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.6, y1=0.35),
        kind=RegionKind.COMB,
    )
    table = Region(
        id="t1",
        page_index=0,
        bbox=BBox(x0=0.12, y0=0.21, x1=0.55, y1=0.34),
        kind=RegionKind.TABLE_CELL,
    )
    other = Region(
        id="l1",
        page_index=0,
        bbox=BBox(x0=0.7, y0=0.7, x1=0.9, y1=0.8),
        kind=RegionKind.TEXT_LINE,
    )
    merged = merge_combs_into_regions(
        [table, other], [comb], prefer_over_table_iou=0.25
    )
    kinds = {r.kind for r in merged}
    ids = {r.id for r in merged}
    assert RegionKind.COMB in kinds
    assert RegionKind.TEXT_LINE in kinds
    assert "t1" not in ids


def test_inpaint_on_blank_comb_is_noop_shape() -> None:
    img = Image.new("RGB", (200, 80), color=(255, 255, 255))
    comb = Region(
        id="empty",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8),
        kind=RegionKind.COMB,
    )
    out = inpaint_comb_lines(img, comb, config=LayoutCombConfig())
    assert out.size[0] > 0 and out.size[1] > 0


def test_split_requires_comb_region_kind_not_enforced() -> None:
    """Split works from bbox geometry even if kind is COMB."""
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    # Draw a tiny 3-cell tick comb so split has something to find.
    import cv2

    arr = np.asarray(img).copy()
    cv2.rectangle(arr, (10, 5), (90, 35), (20, 20, 20), 1)
    for x in (30, 50, 70):
        cv2.line(arr, (x, 5), (x, 35), (20, 20, 20), 1)
    img = Image.fromarray(arr)
    comb = Region(
        id="drawn",
        page_index=0,
        bbox=BBox(x0=0.05, y0=0.05, x1=0.95, y1=0.95),
        kind=RegionKind.COMB,
    )
    cells = split_comb_cells(img, comb, config=LayoutCombConfig(min_cells=2))
    assert len(cells) >= 2
    assert all(c.region.kind == RegionKind.FIELD_VALUE for c in cells)


def test_detect_disabled_returns_empty() -> None:
    img = Image.new("RGB", (200, 80), color=(255, 255, 255))
    assert detect_comb_regions(img, config=LayoutCombConfig(enabled=False)) == []
