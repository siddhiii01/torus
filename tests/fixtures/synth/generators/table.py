"""Ruled-table generators for layout table structure tests (S8 / P4.5).

Layouts are randomised from the seed — no fixed page coordinates. Includes a
multi-page split where the same column grid continues onto the next page.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind

_HEADERS: tuple[str, ...] = ("COL A", "COL B", "COL C", "COL D")
_BODY_TOKENS: tuple[str, ...] = (
    "A1",
    "B1",
    "C1",
    "D1",
    "A2",
    "B2",
    "C2",
    "D2",
    "A3",
    "B3",
    "C3",
    "D3",
    "A4",
    "B4",
    "C4",
    "D4",
    "A5",
    "B5",
    "C5",
    "D5",
)


def generate_table(
    seed: int = 0,
    *,
    n_cols: int = 3,
    n_body_rows: int = 3,
    header_rows: int = 1,
    width: int = 720,
    height: int = 900,
    page_index: int = 0,
) -> SynthPage:
    """Draw one ruled table at a seed-dependent position with header + body."""
    rng = random.Random(seed)
    n_cols = max(2, min(n_cols, len(_HEADERS)))
    n_body_rows = max(1, n_body_rows)
    header_rows = max(0, header_rows)

    headers = list(_HEADERS[:n_cols])
    body = _make_body(rng, n_cols, n_body_rows)

    image = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = load_printed_font(size=14)

    cell_w = 64 + rng.randint(0, 24)
    cell_h = 26 + rng.randint(0, 8)
    n_rows = header_rows + n_body_rows
    box_w = n_cols * cell_w + 2
    box_h = n_rows * cell_h + 2
    margin = 20
    x0 = margin + rng.randint(0, max(0, width - box_w - 2 * margin))
    y0 = margin + rng.randint(0, max(0, height - box_h - 2 * margin))

    truths = _draw_grid(
        draw,
        font,
        origin=(x0, y0),
        cell_w=cell_w,
        cell_h=cell_h,
        headers=headers if header_rows else [],
        body=body,
        header_rows=header_rows,
        page_width=width,
        page_height=height,
        page_index=page_index,
        seed=seed,
    )

    return SynthPage(
        kind="table",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths),
        meta={
            "n_cols": n_cols,
            "n_body_rows": n_body_rows,
            "header_rows": header_rows,
            "cell_w": cell_w,
            "cell_h": cell_h,
            "origin": (x0, y0),
            "headers": headers,
            "body": body,
        },
    )


def generate_split_table(
    seed: int = 0,
    *,
    n_cols: int = 3,
    rows_page0: int = 2,
    rows_page1: int = 2,
    reprint_header: bool = True,
    width: int = 720,
    height: int = 500,
) -> tuple[SynthPage, SynthPage]:
    """Draw a table split across two pages with aligned columns.

    Page 0 places the table in the bottom band; page 1 continues in the top
    band with the same column x positions (proves cross-page stitch without
    fixed absolute coords — only relative alignment).
    """
    rng = random.Random(seed)
    n_cols = max(2, min(n_cols, len(_HEADERS)))
    rows_page0 = max(1, rows_page0)
    rows_page1 = max(1, rows_page1)

    headers = list(_HEADERS[:n_cols])
    body_all = _make_body(rng, n_cols, rows_page0 + rows_page1)
    body0 = body_all[:rows_page0]
    body1 = body_all[rows_page0:]

    cell_w = 70 + rng.randint(0, 16)
    cell_h = 28 + rng.randint(0, 6)
    margin_x = 24 + rng.randint(0, max(0, width - n_cols * cell_w - 48))

    # Page 0: bottom band.
    n_rows0 = 1 + rows_page0  # header + body
    box_h0 = n_rows0 * cell_h + 2
    y0 = height - box_h0 - (8 + rng.randint(0, 12))
    y0 = max(8, y0)

    page0_img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw0 = ImageDraw.Draw(page0_img)
    font = load_printed_font(size=14)
    truths0 = _draw_grid(
        draw0,
        font,
        origin=(margin_x, y0),
        cell_w=cell_w,
        cell_h=cell_h,
        headers=headers,
        body=body0,
        header_rows=1,
        page_width=width,
        page_height=height,
        page_index=0,
        seed=seed,
        id_prefix="p0",
    )
    page0 = SynthPage(
        kind="table",
        seed=seed,
        image=page0_img,
        page_index=0,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths0),
        meta={
            "part": "first",
            "n_cols": n_cols,
            "origin": (margin_x, y0),
            "cell_w": cell_w,
            "cell_h": cell_h,
            "headers": headers,
            "body": body0,
            "reprint_header": reprint_header,
        },
    )

    # Page 1: top band, same x origin / column width.
    header1 = headers if reprint_header else []
    header_rows1 = 1 if reprint_header else 0
    n_rows1 = header_rows1 + rows_page1
    box_h1 = n_rows1 * cell_h + 2
    y1 = 8 + rng.randint(0, 16)
    if y1 + box_h1 >= height:
        y1 = max(4, height - box_h1 - 4)

    page1_img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw1 = ImageDraw.Draw(page1_img)
    truths1 = _draw_grid(
        draw1,
        font,
        origin=(margin_x, y1),
        cell_w=cell_w,
        cell_h=cell_h,
        headers=header1,
        body=body1,
        header_rows=header_rows1,
        page_width=width,
        page_height=height,
        page_index=1,
        seed=seed,
        id_prefix="p1",
    )
    page1 = SynthPage(
        kind="table",
        seed=seed,
        image=page1_img,
        page_index=1,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths1),
        meta={
            "part": "continuation",
            "n_cols": n_cols,
            "origin": (margin_x, y1),
            "cell_w": cell_w,
            "cell_h": cell_h,
            "headers": header1,
            "body": body1,
            "reprint_header": reprint_header,
            "continues_from_seed": seed,
        },
    )
    return page0, page1


def _make_body(rng: random.Random, n_cols: int, n_rows: int) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    pool = list(_BODY_TOKENS)
    rng.shuffle(pool)
    idx = 0
    for _ in range(n_rows):
        cells: list[str] = []
        for _c in range(n_cols):
            cells.append(pool[idx % len(pool)])
            idx += 1
        rows.append(tuple(cells))
    return rows


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    font: object,
    *,
    origin: tuple[int, int],
    cell_w: int,
    cell_h: int,
    headers: list[str],
    body: list[tuple[str, ...]],
    header_rows: int,
    page_width: int,
    page_height: int,
    page_index: int,
    seed: int,
    id_prefix: str = "",
) -> list[RegionTruth]:
    ox, oy = origin
    n_cols = len(headers) if headers else (len(body[0]) if body else 0)
    if n_cols == 0:
        return []
    n_rows = header_rows + len(body)

    for r in range(n_rows + 1):
        y = oy + r * cell_h
        draw.line((ox, y, ox + n_cols * cell_w, y), fill=(35, 35, 35), width=3)
    for c in range(n_cols + 1):
        x = ox + c * cell_w
        draw.line((x, oy, x, oy + n_rows * cell_h), fill=(35, 35, 35), width=3)

    truths: list[RegionTruth] = []
    # Header cells.
    for hr in range(header_rows):
        for c, header in enumerate(headers):
            cell = (
                ox + c * cell_w,
                oy + hr * cell_h,
                ox + (c + 1) * cell_w,
                oy + (hr + 1) * cell_h,
            )
            _text_in_cell(draw, font, cell, header)
            rid = f"table_cell-{id_prefix}{seed}-h{hr}c{c}"
            truths.append(
                RegionTruth(
                    region=_cell_region(rid, page_index, cell, page_width, page_height),
                    text=header,
                )
            )

    for r, row in enumerate(body):
        for c, value in enumerate(row):
            cell = (
                ox + c * cell_w,
                oy + (header_rows + r) * cell_h,
                ox + (c + 1) * cell_w,
                oy + (header_rows + r + 1) * cell_h,
            )
            _text_in_cell(draw, font, cell, value)
            rid = f"table_cell-{id_prefix}{seed}-r{r}c{c}"
            truths.append(
                RegionTruth(
                    region=_cell_region(rid, page_index, cell, page_width, page_height),
                    text=value,
                )
            )
    return truths


def _text_in_cell(
    draw: ImageDraw.ImageDraw,
    font: object,
    cell: tuple[int, int, int, int],
    text: str,
) -> None:
    tw, th = text_size(draw, text, font)
    cx0 = cell[0] + max(2, (cell[2] - cell[0] - tw) // 2)
    cy0 = cell[1] + max(1, (cell[3] - cell[1] - th) // 2)
    draw.text((cx0, cy0), text, fill=(10, 10, 10), font=font)  # type: ignore[arg-type]


def _cell_region(
    rid: str,
    page_index: int,
    cell: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> Region:
    return Region(
        id=rid,
        page_index=page_index,
        bbox=pixel_bbox_to_norm(
            x0=cell[0],
            y0=cell[1],
            x1=cell[2],
            y1=cell[3],
            width=page_width,
            height=page_height,
        ),
        kind=RegionKind.TABLE_CELL,
    )
