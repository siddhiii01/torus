"""Component tests: table detect / headers / rows / cross-page stitch (P4.5)."""

from __future__ import annotations

import pytest
from tests.fixtures.synth import generate_split_table, generate_table

from torus.config import load_config
from torus.core import Page, Region, RegionKind
from torus.layout import (
    detect_layout,
    detect_tables,
    locate_matches,
    process_tables,
    stitch_tables_across_pages,
)


def _page_from_synth(synth: object, index: int | None = None) -> Page:
    page = Page(
        index=synth.page_index if index is None else index,  # type: ignore[attr-defined]
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def _gt_cells(synth: object) -> list[Region]:
    return [
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == RegionKind.TABLE_CELL
    ]


@pytest.mark.parametrize("seed", [0, 3, 7, 11, 17, 23, 29, 37, 41, 53])
def test_table_cells_located_on_random_layout(seed: int) -> None:
    synth = generate_table(seed=seed, n_cols=3, n_body_rows=3, header_rows=1)
    cfg = load_config().pipeline.layout
    tables = detect_tables(synth.image, page_index=0, config=cfg.table)
    assert tables, f"no table detected seed={seed}"
    table = max(tables, key=lambda t: len(t.cells))
    assert table.n_cols == 3, f"cols={table.n_cols} seed={seed}"
    assert table.n_rows >= 3, f"rows={table.n_rows} seed={seed}"
    assert table.rows[0].is_header is True

    predicted = [c.region for c in table.cells]
    matched, total = locate_matches(
        predicted, _gt_cells(synth), iou_threshold=cfg.locate_iou
    )
    rate = matched / total if total else 0.0
    assert rate >= 0.97, f"locate rate {rate:.3f} ({matched}/{total}) seed={seed}"


@pytest.mark.parametrize("seed", [2, 8, 14, 20, 26, 32])
def test_process_tables_maps_rows(seed: int) -> None:
    synth = generate_table(seed=seed, n_cols=3, n_body_rows=2, header_rows=1)
    cfg = load_config().pipeline.layout.table
    tables = process_tables(synth.image, page_index=0, config=cfg)
    assert len(tables) >= 1
    table = tables[0]
    assert len(table.rows) == table.n_rows
    header = table.rows[0]
    assert header.is_header
    assert len(header.cells) == table.n_cols
    for body in table.rows[1:]:
        assert not body.is_header
        assert len(body.cells) == table.n_cols


@pytest.mark.parametrize("seed", [1, 5, 9, 13, 19, 27])
def test_detect_layout_includes_table_source(seed: int) -> None:
    synth = generate_table(seed=seed, n_cols=3, n_body_rows=2)
    page = _page_from_synth(synth)
    cfg = load_config().pipeline.layout
    result = detect_layout(page, config=cfg)
    assert "table" in result.sources or "classic" in result.sources
    cells = [r for r in result.regions if r.kind == RegionKind.TABLE_CELL]
    assert cells, f"no TABLE_CELL from detect_layout seed={seed}"
    matched, total = locate_matches(
        cells, _gt_cells(synth), iou_threshold=cfg.locate_iou
    )
    assert matched / total >= 0.9, f"{matched}/{total} seed={seed}"


@pytest.mark.parametrize("seed", [4, 12, 18, 24, 36, 48])
def test_split_table_stitched_across_pages(seed: int) -> None:
    page0, page1 = generate_split_table(
        seed=seed, n_cols=3, rows_page0=2, rows_page1=2, reprint_header=True
    )
    cfg = load_config().pipeline.layout.table
    t0 = detect_tables(page0.image, page_index=0, config=cfg)
    t1 = detect_tables(page1.image, page_index=1, config=cfg)
    assert t0, f"page0 empty seed={seed}"
    assert t1, f"page1 empty seed={seed}"

    stitched = stitch_tables_across_pages([t0, t1], config=cfg)
    first = [t for t in stitched if t.page_index == 0]
    second = [t for t in stitched if t.page_index == 1]
    assert first and second
    linked = [
        (a, b)
        for a in first
        for b in second
        if a.continues_to == b.table_id and b.continues_from == a.table_id
    ]
    assert linked, f"no cross-page link seed={seed}"
    a, b = linked[0]
    assert a.logical_id == b.logical_id
    assert a.n_cols == b.n_cols == 3
    assert b.row_offset >= 1


@pytest.mark.parametrize("seed", [6, 15, 22, 33])
def test_aggregate_locate_rate_above_phase_gate(seed: int) -> None:
    """Phase acceptance proxy: ≥97% cells located across several random seeds."""
    cfg = load_config().pipeline.layout
    matched_total = 0
    expected_total = 0
    for s in range(seed, seed + 5):
        synth = generate_table(seed=s, n_cols=3, n_body_rows=3)
        tables = detect_tables(synth.image, page_index=0, config=cfg.table)
        predicted = [c.region for t in tables for c in t.cells]
        m, t = locate_matches(predicted, _gt_cells(synth), iou_threshold=cfg.locate_iou)
        matched_total += m
        expected_total += t
    rate = matched_total / expected_total if expected_total else 0.0
    assert rate >= 0.97, f"aggregate rate {rate:.3f} ({matched_total}/{expected_total})"
