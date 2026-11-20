"""Unit tests for the mixed layout-page synth generator (S8 / P4.1)."""

from __future__ import annotations

import hashlib

from tests.fixtures.synth import generate_layout_page
from torus.core import RegionKind


def _digest(image: object) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()  # type: ignore[attr-defined]


def test_layout_page_deterministic() -> None:
    a = generate_layout_page(seed=42)
    b = generate_layout_page(seed=42)
    assert _digest(a.image) == _digest(b.image)
    assert [r.region.id for r in a.regions] == [r.region.id for r in b.regions]


def test_layout_page_seed_changes_positions() -> None:
    a = generate_layout_page(seed=1)
    b = generate_layout_page(seed=2)
    assert _digest(a.image) != _digest(b.image)
    # At least one ground-truth bbox must differ (no fixed coordinates).
    a_boxes = {
        (
            r.region.kind,
            r.region.bbox.x0,
            r.region.bbox.y0,
            r.region.bbox.x1,
            r.region.bbox.y1,
        )
        for r in a.regions
    }
    b_boxes = {
        (
            r.region.kind,
            r.region.bbox.x0,
            r.region.bbox.y0,
            r.region.bbox.x1,
            r.region.bbox.y1,
        )
        for r in b.regions
    }
    assert a_boxes != b_boxes


def test_layout_page_covers_expected_kinds() -> None:
    page = generate_layout_page(seed=7)
    kinds = {r.region.kind for r in page.regions}
    assert RegionKind.TEXT_LINE in kinds
    assert RegionKind.TABLE_CELL in kinds
    assert RegionKind.STAMP in kinds
    assert RegionKind.SIGNATURE in kinds
    assert RegionKind.PHOTO in kinds
    for item in page.regions:
        bbox = item.region.bbox
        assert 0.0 <= bbox.x0 < bbox.x1 <= 1.0
        assert 0.0 <= bbox.y0 < bbox.y1 <= 1.0
