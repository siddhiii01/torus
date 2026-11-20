"""Unit tests for layout geometry helpers (P4.1)."""

from __future__ import annotations

import pytest

from torus.core import BBox
from torus.layout.geometry import (
    bbox_area,
    bbox_intersection_over_gt,
    bbox_iou,
    pixel_to_norm_bbox,
)


def test_pixel_to_norm_bbox_unit_square() -> None:
    box = pixel_to_norm_bbox(x0=0, y0=0, x1=100, y1=200, width=100, height=200)
    assert box == BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)


def test_bbox_iou_identical_is_one() -> None:
    a = BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.4)
    assert bbox_iou(a, a) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero() -> None:
    a = BBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    b = BBox(x0=0.5, y0=0.5, x1=0.8, y1=0.8)
    assert bbox_iou(a, b) == 0.0


def test_bbox_iou_partial_overlap() -> None:
    a = BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    b = BBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75)
    # Intersection 0.25x0.25=0.0625; union=0.25+0.25-0.0625=0.4375
    assert bbox_iou(a, b) == pytest.approx(0.0625 / 0.4375)


def test_intersection_over_gt() -> None:
    gt = BBox(x0=0.0, y0=0.0, x1=0.4, y1=0.4)
    pred = BBox(x0=0.0, y0=0.0, x1=0.2, y1=0.2)
    assert bbox_intersection_over_gt(pred, gt) == pytest.approx(0.25)
    assert bbox_area(gt) == pytest.approx(0.16)
