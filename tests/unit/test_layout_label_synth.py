"""Unit test for the label-anchor synthetic generator (S8 / P4.2)."""

from __future__ import annotations

from tests.fixtures.synth import generate, generate_label_anchor_page


def test_generate_label_anchor_deterministic() -> None:
    a = generate_label_anchor_page(seed=11)
    b = generate_label_anchor_page(seed=11)
    assert a.kind == "label_anchor"
    assert a.meta["field_count"] == b.meta["field_count"]
    assert [r.region.id for r in a.regions] == [r.region.id for r in b.regions]
    assert [r.text for r in a.regions] == [r.text for r in b.regions]


def test_generate_label_anchor_varies_with_seed() -> None:
    a = generate_label_anchor_page(seed=3)
    b = generate_label_anchor_page(seed=4)
    a_boxes = [(r.region.bbox.x0, r.region.bbox.y0) for r in a.regions]
    b_boxes = [(r.region.bbox.x0, r.region.bbox.y0) for r in b.regions]
    assert a_boxes != b_boxes


def test_generate_dispatch_label_anchor() -> None:
    page = generate("label_anchor", seed=5, field_count=2)
    assert page.kind == "label_anchor"
    assert len(page.meta["fields"]) == 2
