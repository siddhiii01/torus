"""Unit tests for table structure / stitch helpers (P4.5)."""

from __future__ import annotations

from torus.config import LayoutConfig, LayoutTableConfig, load_config
from torus.core import BBox, Region, RegionKind
from torus.layout.table import (
    DetectedTable,
    TableCell,
    TableRow,
    map_table_rows,
    merge_tables_into_regions,
    stitch_tables_across_pages,
    structure_cell_regions,
)


def _cell(
    rid: str, x0: float, y0: float, x1: float, y1: float, page: int = 0
) -> Region:
    return Region(
        id=rid,
        page_index=page,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        kind=RegionKind.TABLE_CELL,
    )


def test_layout_table_config_loads() -> None:
    cfg = load_config().pipeline.layout
    assert isinstance(cfg, LayoutConfig)
    assert cfg.table.enabled is True
    assert cfg.table.header_rows >= 1
    assert cfg.table.min_columns >= 2


def test_structure_assigns_header_and_body_rows() -> None:
    # 2 cols × 3 rows (1 header + 2 body), normalised grid.
    cells = [
        _cell("h0", 0.10, 0.10, 0.30, 0.18),
        _cell("h1", 0.30, 0.10, 0.50, 0.18),
        _cell("r0c0", 0.10, 0.18, 0.30, 0.26),
        _cell("r0c1", 0.30, 0.18, 0.50, 0.26),
        _cell("r1c0", 0.10, 0.26, 0.30, 0.34),
        _cell("r1c1", 0.30, 0.26, 0.50, 0.34),
    ]
    tables = structure_cell_regions(cells, page_index=0, config=LayoutTableConfig())
    assert len(tables) == 1
    table = tables[0]
    assert table.n_cols == 2
    assert table.n_rows == 3
    assert table.rows[0].is_header is True
    assert all(c.is_header for c in table.rows[0].cells)
    assert table.rows[1].is_header is False
    assert table.rows[2].is_header is False
    mapped = map_table_rows(table)
    assert len(mapped) == 3
    assert set(mapped[0].keys()) == {0, 1}


def test_two_spatially_separate_tables() -> None:
    top = [
        _cell("t00", 0.1, 0.05, 0.3, 0.12),
        _cell("t01", 0.3, 0.05, 0.5, 0.12),
        _cell("t10", 0.1, 0.12, 0.3, 0.19),
        _cell("t11", 0.3, 0.12, 0.5, 0.19),
    ]
    bottom = [
        _cell("b00", 0.1, 0.70, 0.3, 0.77),
        _cell("b01", 0.3, 0.70, 0.5, 0.77),
        _cell("b10", 0.1, 0.77, 0.3, 0.84),
        _cell("b11", 0.3, 0.77, 0.5, 0.84),
    ]
    tables = structure_cell_regions(
        top + bottom, page_index=0, config=LayoutTableConfig(table_gap_tol=1.5)
    )
    assert len(tables) == 2
    assert all(t.n_cols == 2 and t.n_rows == 2 for t in tables)


def test_merge_replaces_overlapping_classic_cells() -> None:
    classic = _cell("old", 0.10, 0.10, 0.40, 0.20)
    other = Region(
        id="line",
        page_index=0,
        bbox=BBox(x0=0.6, y0=0.6, x1=0.9, y1=0.7),
        kind=RegionKind.TEXT_LINE,
    )
    structured = _cell("new", 0.11, 0.11, 0.39, 0.19)
    merged = merge_tables_into_regions([classic, other], [structured], nms_iou=0.4)
    ids = {r.id for r in merged}
    assert "old" not in ids
    assert "new" in ids
    assert "line" in ids


def test_stitch_links_aligned_continuation() -> None:
    cfg = LayoutTableConfig(page_edge_band=0.25, col_align_tol=0.05)
    left = DetectedTable(
        table_id="t0",
        page_index=0,
        logical_id="t0",
        n_cols=2,
        n_rows=2,
        bbox=BBox(x0=0.1, y0=0.80, x1=0.5, y1=0.95),
        column_centers=(0.2, 0.4),
        cells=(
            TableCell(
                row=0,
                col=0,
                is_header=True,
                region=_cell("l00", 0.1, 0.80, 0.3, 0.87),
            ),
            TableCell(
                row=0,
                col=1,
                is_header=True,
                region=_cell("l01", 0.3, 0.80, 0.5, 0.87),
            ),
            TableCell(
                row=1,
                col=0,
                is_header=False,
                region=_cell("l10", 0.1, 0.87, 0.3, 0.95),
            ),
            TableCell(
                row=1,
                col=1,
                is_header=False,
                region=_cell("l11", 0.3, 0.87, 0.5, 0.95),
            ),
        ),
        rows=(
            TableRow(
                index=0,
                is_header=True,
                cells=(
                    TableCell(
                        row=0,
                        col=0,
                        is_header=True,
                        region=_cell("l00", 0.1, 0.80, 0.3, 0.87),
                    ),
                    TableCell(
                        row=0,
                        col=1,
                        is_header=True,
                        region=_cell("l01", 0.3, 0.80, 0.5, 0.87),
                    ),
                ),
            ),
            TableRow(
                index=1,
                is_header=False,
                cells=(
                    TableCell(
                        row=1,
                        col=0,
                        is_header=False,
                        region=_cell("l10", 0.1, 0.87, 0.3, 0.95),
                    ),
                    TableCell(
                        row=1,
                        col=1,
                        is_header=False,
                        region=_cell("l11", 0.3, 0.87, 0.5, 0.95),
                    ),
                ),
            ),
        ),
    )
    right = DetectedTable(
        table_id="t1",
        page_index=1,
        logical_id="t1",
        n_cols=2,
        n_rows=2,
        bbox=BBox(x0=0.1, y0=0.02, x1=0.5, y1=0.18),
        column_centers=(0.205, 0.405),
        cells=(
            TableCell(
                row=0,
                col=0,
                is_header=True,
                region=_cell("r00", 0.1, 0.02, 0.3, 0.10, page=1),
            ),
            TableCell(
                row=0,
                col=1,
                is_header=True,
                region=_cell("r01", 0.3, 0.02, 0.5, 0.10, page=1),
            ),
            TableCell(
                row=1,
                col=0,
                is_header=False,
                region=_cell("r10", 0.1, 0.10, 0.3, 0.18, page=1),
            ),
            TableCell(
                row=1,
                col=1,
                is_header=False,
                region=_cell("r11", 0.3, 0.10, 0.5, 0.18, page=1),
            ),
        ),
        rows=(
            TableRow(
                index=0,
                is_header=True,
                cells=(
                    TableCell(
                        row=0,
                        col=0,
                        is_header=True,
                        region=_cell("r00", 0.1, 0.02, 0.3, 0.10, page=1),
                    ),
                    TableCell(
                        row=0,
                        col=1,
                        is_header=True,
                        region=_cell("r01", 0.3, 0.02, 0.5, 0.10, page=1),
                    ),
                ),
            ),
            TableRow(
                index=1,
                is_header=False,
                cells=(
                    TableCell(
                        row=1,
                        col=0,
                        is_header=False,
                        region=_cell("r10", 0.1, 0.10, 0.3, 0.18, page=1),
                    ),
                    TableCell(
                        row=1,
                        col=1,
                        is_header=False,
                        region=_cell("r11", 0.3, 0.10, 0.5, 0.18, page=1),
                    ),
                ),
            ),
        ),
    )
    stitched = stitch_tables_across_pages([[left], [right]], config=cfg)
    assert len(stitched) == 2
    first = next(t for t in stitched if t.page_index == 0)
    second = next(t for t in stitched if t.page_index == 1)
    assert first.continues_to == second.table_id
    assert second.continues_from == first.table_id
    assert first.logical_id == second.logical_id
    assert second.row_offset == 1  # one body row on page 0


def test_stitch_rejects_misaligned_columns() -> None:
    cfg = LayoutTableConfig(page_edge_band=0.25, col_align_tol=0.02)
    left = DetectedTable(
        table_id="a",
        page_index=0,
        logical_id="a",
        n_cols=2,
        n_rows=1,
        bbox=BBox(x0=0.1, y0=0.85, x1=0.5, y1=0.95),
        column_centers=(0.2, 0.4),
        cells=(),
        rows=(),
    )
    right = DetectedTable(
        table_id="b",
        page_index=1,
        logical_id="b",
        n_cols=2,
        n_rows=1,
        bbox=BBox(x0=0.1, y0=0.02, x1=0.5, y1=0.12),
        column_centers=(0.35, 0.55),  # shifted too far
        cells=(),
        rows=(),
    )
    stitched = stitch_tables_across_pages([[left], [right]], config=cfg)
    assert all(t.continues_from is None and t.continues_to is None for t in stitched)
