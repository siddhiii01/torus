"""Component tests: classic layout detection on random-layout synth (P4.1).

Proves detectors find expected kinds without fixed page coordinates by running
several seeds and matching predicted regions to ground truth via IoU.
"""

from __future__ import annotations

import pytest
from tests.fixtures.synth import generate_layout_page, generate_printed_form

from torus.config import LayoutConfig, load_config
from torus.core import Page, RegionKind
from torus.layout import detect_layout, locate_matches


def _page_from_synth(synth: object, index: int = 0) -> Page:
    page = Page(
        index=index,
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def _expected_by_kind(synth: object, kind: RegionKind) -> list:
    return [
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == kind
        and not str(item.region.id).startswith("text_block-")
    ]


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_layout_detects_kinds_on_random_seeds(seed: int) -> None:
    synth = generate_layout_page(seed=seed)
    page = _page_from_synth(synth)
    cfg = load_config().pipeline.layout
    result = detect_layout(page, cfg)

    assert "classic" in result.sources
    found = {r.kind for r in result.regions}

    # Core kinds that classic CV should find on the synth page.
    assert RegionKind.TEXT_LINE in found
    assert RegionKind.TABLE_CELL in found
    assert RegionKind.STAMP in found
    assert RegionKind.PHOTO in found
    # Signatures are thinner; require at least an attempt across seeds in
    # the aggregate test below rather than every single seed.


def test_layout_location_recall_across_seeds() -> None:
    """≥97% of primary expected regions located across random seeds."""
    cfg = load_config().pipeline.layout
    kinds = (
        RegionKind.TEXT_LINE,
        RegionKind.TABLE_CELL,
        RegionKind.STAMP,
        RegionKind.PHOTO,
        RegionKind.SIGNATURE,
    )
    matched_total = 0
    expected_total = 0
    for seed in range(10, 30):
        synth = generate_layout_page(seed=seed)
        page = _page_from_synth(synth)
        result = detect_layout(page, cfg)
        for kind in kinds:
            expected = _expected_by_kind(synth, kind)
            if not expected:
                continue
            preds = [r for r in result.regions if r.kind == kind]
            # Blocks share TEXT_LINE kind; matching is greedy by IoU.
            hit, total = locate_matches(preds, expected, iou_threshold=cfg.locate_iou)
            matched_total += hit
            expected_total += total

    assert expected_total > 0
    recall = matched_total / expected_total
    assert recall >= 0.97, (
        f"layout recall {recall:.3f} < 0.97 ({matched_total}/{expected_total})"
    )


def test_printed_form_yields_text_lines() -> None:
    synth = generate_printed_form(seed=101, field_count=3)
    page = _page_from_synth(synth)
    result = detect_layout(page, LayoutConfig())
    lines = [r for r in result.regions if r.kind == RegionKind.TEXT_LINE]
    assert len(lines) >= 2


def test_disabled_layout_returns_empty() -> None:
    synth = generate_layout_page(seed=3)
    page = _page_from_synth(synth)
    result = detect_layout(page, LayoutConfig(enabled=False))
    assert result.regions == []
    assert result.sources == ()


def test_no_fixed_coordinates_in_layout_source() -> None:
    """Guardrail: layout package must not hard-code absolute page positions."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "torus" / "layout"
    banned = ("x0 > 0.", "x0<", "page == 0", "page_index == 0")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} contains {token!r}"
