"""Unit tests for layout detect helpers (P4.1)."""

from __future__ import annotations

from torus.config import LayoutConfig, load_config
from torus.core import BBox, Region, RegionKind
from torus.layout import locate_matches
from torus.layout.model import NullLayoutModel, load_layout_model


def test_pipeline_loads_layout_config() -> None:
    cfg = load_config().pipeline.layout
    assert isinstance(cfg, LayoutConfig)
    assert cfg.enabled is True
    assert cfg.classic.enabled is True
    assert cfg.model.enabled is False


def test_layout_model_defaults_to_null() -> None:
    backend = load_layout_model()
    assert isinstance(backend, NullLayoutModel)
    assert backend.name == "null"


def test_locate_matches_greedy_iou() -> None:
    expected = [
        Region(
            id="a",
            page_index=0,
            bbox=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
            kind=RegionKind.TEXT_LINE,
        ),
        Region(
            id="b",
            page_index=0,
            bbox=BBox(x0=0.5, y0=0.5, x1=0.7, y1=0.6),
            kind=RegionKind.STAMP,
        ),
    ]
    predicted = [
        Region(
            id="p1",
            page_index=0,
            bbox=BBox(x0=0.11, y0=0.11, x1=0.29, y1=0.19),
            kind=RegionKind.TEXT_LINE,
        ),
        Region(
            id="p2",
            page_index=0,
            bbox=BBox(x0=0.0, y0=0.0, x1=0.05, y1=0.05),
            kind=RegionKind.STAMP,
        ),
    ]
    matched, total = locate_matches(predicted, expected, iou_threshold=0.3)
    assert total == 2
    assert matched == 1
