"""Component tests: comb detect / split / inpaint on random-layout synth (P4.3)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.fixtures.synth import generate_comb_row

from torus.config import load_config
from torus.core import BBox, Page, Region, RegionKind
from torus.layout import (
    bbox_iou,
    detect_comb_regions,
    detect_layout,
    inpaint_comb_lines,
    locate_matches,
    process_comb,
    split_comb_cells,
)


def _page_from_synth(synth: object, index: int = 0) -> Page:
    page = Page(
        index=index,
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def _gt_comb(synth: object) -> Region:
    return next(
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == RegionKind.COMB
    )


def _gt_cell_regions(synth: object) -> list[Region]:
    norms = synth.meta["cell_bboxes_norm"]  # type: ignore[attr-defined]
    out: list[Region] = []
    for i, (x0, y0, x1, y1) in enumerate(norms):
        out.append(
            Region(
                id=f"gt-cell-{i}",
                page_index=synth.page_index,  # type: ignore[attr-defined]
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                kind=RegionKind.FIELD_VALUE,
            )
        )
    return out


@pytest.mark.parametrize("seed", [1, 7, 14, 21, 28, 35, 42])
@pytest.mark.parametrize("style", ["boxed", "ticks"])
@pytest.mark.parametrize("crossing", [False, True])
def test_comb_detected_on_random_layout(seed: int, style: str, crossing: bool) -> None:
    synth = generate_comb_row(
        seed=seed,
        style=style,
        crossing_strokes=crossing,  # type: ignore[arg-type]
    )
    cfg = load_config().pipeline.layout.comb
    found = detect_comb_regions(synth.image, page_index=0, config=cfg)
    assert found, f"no comb detected seed={seed} style={style} crossing={crossing}"
    gt = _gt_comb(synth)
    best = max(bbox_iou(r.bbox, gt.bbox) for r in found)
    assert best >= 0.3, f"IoU {best:.3f} too low seed={seed}"


def test_comb_location_recall_across_seeds() -> None:
    """≥97% of expected comb regions located across random seeds/styles."""
    cfg = load_config().pipeline.layout
    matched = 0
    total = 0
    for seed in range(50, 90):
        for style in ("boxed", "ticks"):
            for crossing in (False, True):
                synth = generate_comb_row(
                    seed=seed,
                    style=style,  # type: ignore[arg-type]
                    crossing_strokes=crossing,
                )
                found = detect_comb_regions(synth.image, page_index=0, config=cfg.comb)
                expected = [_gt_comb(synth)]
                hit, n = locate_matches(found, expected, iou_threshold=cfg.locate_iou)
                matched += hit
                total += n
    recall = matched / total if total else 0.0
    assert recall >= 0.97, f"comb recall {recall:.3f} < 0.97 ({matched}/{total})"


@pytest.mark.parametrize("seed", [3, 11, 19, 27])
@pytest.mark.parametrize("style", ["boxed", "ticks"])
def test_cell_split_matches_ground_truth(seed: int, style: str) -> None:
    synth = generate_comb_row(
        seed=seed,
        style=style,
        crossing_strokes=False,
        cells=6,  # type: ignore[arg-type]
    )
    cfg = load_config().pipeline.layout.comb
    gt = _gt_comb(synth)
    # Use GT comb bbox so this test isolates split quality.
    cells = split_comb_cells(synth.image, gt, config=cfg)
    expected = _gt_cell_regions(synth)
    assert len(cells) == len(expected)
    hit, n = locate_matches(
        [c.region for c in cells],
        expected,
        iou_threshold=0.35,
        require_kind=False,
    )
    assert hit == n


@pytest.mark.parametrize("seed", [5, 15, 25])
def test_inpaint_preserves_crossing_stroke_ink(seed: int) -> None:
    """After inpaint, dark ink near cell centers must remain (strokes not erased)."""
    synth = generate_comb_row(
        seed=seed, style="ticks", crossing_strokes=True, cells=5, text="ABCDE"
    )
    cfg = load_config().pipeline.layout.comb
    gt = _gt_comb(synth)
    crop = inpaint_comb_lines(synth.image, gt, config=cfg)
    arr = np.asarray(crop.convert("L"))
    # At least some dark pixels remain (characters / bridges).
    dark = int(np.count_nonzero(arr < 100))
    assert dark >= 20, f"inpaint removed too much ink ({dark} dark px)"


@pytest.mark.parametrize("seed", [8, 18])
def test_process_comb_end_to_end(seed: int) -> None:
    synth = generate_comb_row(seed=seed, style="ticks", crossing_strokes=True)
    cfg = load_config().pipeline.layout.comb
    gt = _gt_comb(synth)
    result = process_comb(synth.image, gt, config=cfg)
    assert result.inpainted_crop is not None
    assert len(result.cells) == synth.meta["cells"]
    assert result.pixel_bbox is not None


def test_detect_layout_includes_comb_source() -> None:
    synth = generate_comb_row(seed=9, style="ticks", crossing_strokes=False, cells=5)
    page = _page_from_synth(synth)
    cfg = load_config().pipeline.layout
    result = detect_layout(page, cfg)
    assert "comb" in result.sources
    assert any(r.kind == RegionKind.COMB for r in result.regions)
