"""Table cell / header / row structure and cross-page stitch (P4.5).

Cell boxes come from the classic H/V ruling detector (P4.1). This module
groups them into tables, assigns row/column indices, marks header rows, and
stitches tables that continue onto the next page by column alignment.
No absolute page coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import LayoutClassicConfig, LayoutTableConfig
from torus.core.models import BBox, Region
from torus.core.types import RegionKind
from torus.layout.classic import detect_classic_regions
from torus.layout.geometry import bbox_iou


class TableCell(BaseModel):
    """One cell inside a detected table, with grid indices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    is_header: bool = False
    region: Region


class TableRow(BaseModel):
    """One row of a table (header or body)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    is_header: bool = False
    cells: tuple[TableCell, ...] = ()


class DetectedTable(BaseModel):
    """Structured table on one page (or a continuation fragment)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    # Shared across page fragments of the same logical table.
    logical_id: str = Field(min_length=1)
    # Row index offset for body rows when this fragment continues a prior page.
    row_offset: int = Field(default=0, ge=0)
    n_cols: int = Field(ge=1)
    n_rows: int = Field(ge=1)
    bbox: BBox
    # Normalised x-centres of columns (left → right), for cross-page align.
    column_centers: tuple[float, ...] = ()
    cells: tuple[TableCell, ...] = ()
    rows: tuple[TableRow, ...] = ()
    continues_from: str | None = None
    continues_to: str | None = None


def detect_table_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutTableConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[Region]:
    """Locate ``RegionKind.TABLE_CELL`` regions (flat list)."""
    tables = detect_tables(
        image,
        page_index=page_index,
        config=config,
        classic_config=classic_config,
    )
    return [c.region for t in tables for c in t.cells]


def detect_tables(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutTableConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[DetectedTable]:
    """Detect ruled tables and map cells into header/body rows."""
    cfg = config if config is not None else LayoutTableConfig()
    if not cfg.enabled:
        return []

    classic_cfg = (
        classic_config
        if classic_config is not None
        else LayoutClassicConfig(
            detect_text_lines=False,
            detect_text_blocks=False,
            detect_stamps=False,
            detect_signatures=False,
            detect_photos=False,
            detect_tables=True,
        )
    )
    # Force table-only classic pass even if a full classic_config was passed.
    classic_cfg = classic_cfg.model_copy(
        update={
            "detect_text_lines": False,
            "detect_text_blocks": False,
            "detect_stamps": False,
            "detect_signatures": False,
            "detect_photos": False,
            "detect_tables": True,
            "enabled": True,
        }
    )

    raw = detect_classic_regions(image, page_index=page_index, config=classic_cfg)
    cell_regions = [r for r in raw if r.kind == RegionKind.TABLE_CELL]
    if len(cell_regions) < cfg.min_cells:
        return []

    return structure_cell_regions(
        cell_regions,
        page_index=page_index,
        config=cfg,
    )


def structure_cell_regions(
    cell_regions: Sequence[Region],
    *,
    page_index: int,
    config: LayoutTableConfig | None = None,
) -> list[DetectedTable]:
    """Group flat ``TABLE_CELL`` regions into structured tables."""
    cfg = config if config is not None else LayoutTableConfig()
    cells = [r for r in cell_regions if r.kind == RegionKind.TABLE_CELL]
    if len(cells) < cfg.min_cells:
        return []

    clusters = _split_into_table_clusters(cells, cfg)
    tables: list[DetectedTable] = []
    for i, cluster in enumerate(clusters):
        built = _structure_one_table(
            cluster,
            page_index=page_index,
            table_index=i,
            config=cfg,
        )
        if built is not None:
            tables.append(built)
    return tables


def map_table_rows(table: DetectedTable) -> list[dict[int, Region]]:
    """Return each body (and header) row as ``{col_index: region}``."""
    out: list[dict[int, Region]] = []
    for row in table.rows:
        mapping: dict[int, Region] = {c.col: c.region for c in row.cells}
        out.append(mapping)
    return out


def stitch_tables_across_pages(
    tables_by_page: Sequence[Sequence[DetectedTable]],
    *,
    config: LayoutTableConfig | None = None,
) -> list[DetectedTable]:
    """Link continuation fragments across consecutive pages.

    Matching uses column-count equality and normalised column-centre
    alignment. A prior fragment must sit in the bottom edge band; the next
    fragment in the top edge band. Returns a flat list with
    ``continues_from`` / ``continues_to`` and shared ``logical_id`` /
    ``row_offset`` updated.
    """
    cfg = config if config is not None else LayoutTableConfig()
    if not tables_by_page:
        return []

    # Work on copies we can rewrite.
    pages: list[list[DetectedTable]] = [list(p) for p in tables_by_page]
    for page_i in range(len(pages) - 1):
        current = pages[page_i]
        nxt = pages[page_i + 1]
        used_next: set[int] = set()
        new_current: list[DetectedTable] = []
        new_next = list(nxt)

        for left in current:
            match_j = _best_continuation(left, new_next, used_next, cfg)
            if match_j is None:
                new_current.append(left)
                continue
            right = new_next[match_j]
            used_next.add(match_j)
            logical = left.logical_id
            # Body rows on the continuation continue after left's last body row.
            body_rows = sum(1 for r in left.rows if not r.is_header)
            offset = left.row_offset + body_rows
            linked_left = left.model_copy(
                update={
                    "continues_to": right.table_id,
                    "logical_id": logical,
                }
            )
            # Keep leading geometric header rows on the continuation (reprints);
            # row_offset advances body numbering for the logical table.
            right_rows = _mark_continuation_headers(right.rows, cfg.header_rows)
            linked_right = right.model_copy(
                update={
                    "continues_from": left.table_id,
                    "logical_id": logical,
                    "row_offset": offset,
                    "rows": right_rows,
                    "cells": tuple(c for r in right_rows for c in r.cells),
                }
            )
            new_current.append(linked_left)
            new_next[match_j] = linked_right

        pages[page_i] = new_current
        pages[page_i + 1] = new_next

    flat: list[DetectedTable] = []
    for page in pages:
        flat.extend(page)
    return flat


def merge_tables_into_regions(
    existing: list[Region],
    table_cells: Sequence[Region],
    *,
    nms_iou: float = 0.4,
) -> list[Region]:
    """Replace overlapping classic ``TABLE_CELL`` regions with structured ones."""
    kept: list[Region] = []
    for r in existing:
        if r.kind != RegionKind.TABLE_CELL:
            kept.append(r)
            continue
        if any(bbox_iou(r.bbox, c.bbox) >= nms_iou for c in table_cells):
            continue
        kept.append(r)

    merged = list(kept)
    for cand in table_cells:
        if any(
            r.kind == RegionKind.TABLE_CELL and bbox_iou(r.bbox, cand.bbox) >= nms_iou
            for r in merged
        ):
            continue
        merged.append(cand)
    return merged


def process_tables(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutTableConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[DetectedTable]:
    """End-to-end single-page table detection and row mapping."""
    return detect_tables(
        image,
        page_index=page_index,
        config=config,
        classic_config=classic_config,
    )


def _mark_continuation_headers(
    rows: tuple[TableRow, ...],
    header_rows: int,
) -> tuple[TableRow, ...]:
    """Keep leading rows as header reprints; body cells lose header flag."""
    out: list[TableRow] = []
    for r in rows:
        is_header = r.index < header_rows
        cells = tuple(
            TableCell(
                row=c.row,
                col=c.col,
                is_header=is_header,
                region=c.region,
            )
            for c in r.cells
        )
        out.append(TableRow(index=r.index, is_header=is_header, cells=cells))
    return tuple(out)


def _best_continuation(
    left: DetectedTable,
    candidates: Sequence[DetectedTable],
    used: set[int],
    cfg: LayoutTableConfig,
) -> int | None:
    if left.bbox.y1 < 1.0 - cfg.page_edge_band:
        return None
    best_j: int | None = None
    best_score = float("inf")
    for j, right in enumerate(candidates):
        if j in used:
            continue
        if right.continues_from is not None:
            continue
        if right.bbox.y0 > cfg.page_edge_band:
            continue
        if not _columns_align(left, right, cfg):
            continue
        # Prefer closest column-centre L1 distance.
        score = sum(
            abs(a - b)
            for a, b in zip(left.column_centers, right.column_centers, strict=True)
        )
        if score < best_score:
            best_score = score
            best_j = j
    return best_j


def _columns_align(
    a: DetectedTable,
    b: DetectedTable,
    cfg: LayoutTableConfig,
) -> bool:
    if a.n_cols != b.n_cols:
        return False
    if len(a.column_centers) != len(b.column_centers):
        return False
    if a.n_cols < cfg.min_columns:
        return False
    return all(
        abs(ca - cb) <= cfg.col_align_tol
        for ca, cb in zip(a.column_centers, b.column_centers, strict=True)
    )


def _split_into_table_clusters(
    cells: Sequence[Region],
    cfg: LayoutTableConfig,
) -> list[list[Region]]:
    """Split cells into spatially separate tables via gap thresholding."""
    if not cells:
        return []
    heights = [_height(c.bbox) for c in cells]
    med_h = _median(heights) or 0.02
    gap_lim = cfg.table_gap_tol * med_h

    ordered = sorted(cells, key=lambda r: (r.bbox.y0, r.bbox.x0))
    # Union-find on proximity (centre distance).
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, a in enumerate(ordered):
        acx, acy = _centre(a.bbox)
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            # Early exit when vertical gap alone exceeds threshold by a lot.
            if b.bbox.y0 - a.bbox.y1 > gap_lim * 3:
                break
            bcx, bcy = _centre(b.bbox)
            # Chebyshev-ish gap between boxes (not centres).
            dx = max(0.0, max(a.bbox.x0, b.bbox.x0) - min(a.bbox.x1, b.bbox.x1))
            dy = max(0.0, max(a.bbox.y0, b.bbox.y0) - min(a.bbox.y1, b.bbox.y1))
            # Also allow near-touching via centre distance for sparse grids.
            centre_dist = max(abs(acx - bcx), abs(acy - bcy))
            if (dx <= gap_lim and dy <= gap_lim) or (
                centre_dist <= gap_lim * 2.5
                and (dx <= gap_lim * 2 or dy <= gap_lim * 2)
            ):
                union(i, j)

    groups: dict[int, list[Region]] = {}
    for i, cell in enumerate(ordered):
        root = find(i)
        groups.setdefault(root, []).append(cell)
    return list(groups.values())


def _structure_one_table(
    cells: Sequence[Region],
    *,
    page_index: int,
    table_index: int,
    config: LayoutTableConfig,
) -> DetectedTable | None:
    if len(cells) < config.min_cells:
        return None

    heights = [_height(c.bbox) for c in cells]
    widths = [_width(c.bbox) for c in cells]
    med_h = _median(heights) or 0.02
    med_w = _median(widths) or 0.02
    row_tol = config.row_cluster_tol * med_h
    col_tol = config.col_cluster_tol * med_w

    row_groups = _cluster_by_axis(cells, axis="y", tol=row_tol)
    if len(row_groups) < config.min_rows:
        return None

    # Column centres from all cells, then assign each cell a column index.
    col_centers = _cluster_centers(
        [_centre(c.bbox)[0] for c in cells],
        tol=col_tol,
    )
    if len(col_centers) < config.min_columns:
        return None

    structured_cells: list[TableCell] = []
    rows: list[TableRow] = []
    for r_idx, group in enumerate(row_groups):
        is_header = r_idx < config.header_rows
        row_cells: list[TableCell] = []
        # Sort left→right within the row.
        ordered = sorted(group, key=lambda r: r.bbox.x0)
        for region in ordered:
            cx = _centre(region.bbox)[0]
            col = _nearest_center_index(cx, col_centers)
            tc = TableCell(
                row=r_idx,
                col=col,
                is_header=is_header,
                region=region,
            )
            row_cells.append(tc)
            structured_cells.append(tc)
        rows.append(TableRow(index=r_idx, is_header=is_header, cells=tuple(row_cells)))

    if len(structured_cells) < config.min_cells:
        return None

    bbox = _union_bbox([c.region.bbox for c in structured_cells])
    tid = f"table-p{page_index}-{table_index}"
    return DetectedTable(
        table_id=tid,
        page_index=page_index,
        logical_id=tid,
        row_offset=0,
        n_cols=len(col_centers),
        n_rows=len(rows),
        bbox=bbox,
        column_centers=tuple(col_centers),
        cells=tuple(structured_cells),
        rows=tuple(rows),
    )


def _cluster_by_axis(
    cells: Sequence[Region],
    *,
    axis: str,
    tol: float,
) -> list[list[Region]]:
    if not cells:
        return []

    def centre_val(region: Region) -> float:
        cx, cy = _centre(region.bbox)
        return cy if axis == "y" else cx

    ordered = sorted(cells, key=centre_val)
    groups: list[list[Region]] = [[ordered[0]]]
    for cell in ordered[1:]:
        prev = groups[-1]
        prev_centre = sum(centre_val(c) for c in prev) / len(prev)
        if abs(centre_val(cell) - prev_centre) <= tol:
            groups[-1].append(cell)
        else:
            groups.append([cell])
    return groups


def _cluster_centers(values: Sequence[float], *, tol: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _nearest_center_index(value: float, centers: Sequence[float]) -> int:
    best_i = 0
    best_d = float("inf")
    for i, c in enumerate(centers):
        d = abs(value - c)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _centre(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def _height(bbox: BBox) -> float:
    return max(0.0, bbox.y1 - bbox.y0)


def _width(bbox: BBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0)


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0
