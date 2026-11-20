"""Mixed-layout page generator for layout detection tests (S8 / P4.1).

Places text lines, a table, a stamp, a signature and a photo-like figure at
seed-dependent positions. No fixed page coordinates: every seed reshuffles
placement. Ground-truth regions use ``torus.core.Region`` / ``RegionKind``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind

_TEXT_LINES: tuple[str, ...] = (
    "REF CODE ALPHA",
    "QTY VALUE SAMPLE",
    "NOTE LINE ONE",
    "STATUS OPEN",
    "BATCH UNIT NINE",
)

_TABLE_HEADERS: tuple[str, ...] = ("COL A", "COL B", "COL C")
_TABLE_CELLS: tuple[tuple[str, ...], ...] = (
    ("X1", "Y1", "Z1"),
    ("X2", "Y2", "Z2"),
)


def generate_layout_page(
    seed: int = 0,
    *,
    width: int = 900,
    height: int = 1200,
    page_index: int = 0,
    include: Sequence[str] | None = None,
) -> SynthPage:
    """Draw a page with several layout kinds at randomised positions.

    ``include`` defaults to all kinds: ``text_line``, ``table``, ``stamp``,
    ``signature``, ``photo``. Positions are chosen per seed with a simple
    non-overlap packer so tests prove detectors do not rely on fixed coords.
    """
    rng = random.Random(seed)
    kinds = (
        list(include)
        if include is not None
        else ["text_line", "table", "stamp", "signature", "photo"]
    )
    rng.shuffle(kinds)

    image = Image.new("RGB", (width, height), color=(248, 248, 245))
    draw = ImageDraw.Draw(image)
    font = load_printed_font(size=18)
    small = load_printed_font(size=14)

    occupied: list[tuple[int, int, int, int]] = []
    truths: list[RegionTruth] = []
    meta: dict[str, object] = {"kinds": list(kinds)}

    for kind in kinds:
        if kind == "text_line":
            box, items = _place_text_block(
                rng, draw, font, width, height, page_index, seed, occupied
            )
            occupied.append(box)
            truths.extend(items)
            meta["text_block_box"] = box
        elif kind == "table":
            box, items = _place_table(
                rng, draw, small, width, height, page_index, seed, occupied
            )
            occupied.append(box)
            truths.extend(items)
            meta["table_box"] = box
        elif kind == "stamp":
            box, item = _place_stamp(
                rng, image, draw, width, height, page_index, seed, occupied
            )
            occupied.append(box)
            truths.append(item)
            meta["stamp_box"] = box
        elif kind == "signature":
            box, item = _place_signature(
                rng, draw, width, height, page_index, seed, occupied
            )
            occupied.append(box)
            truths.append(item)
            meta["signature_box"] = box
        elif kind == "photo":
            box, item = _place_photo(
                rng, image, width, height, page_index, seed, occupied
            )
            occupied.append(box)
            truths.append(item)
            meta["photo_box"] = box
        else:
            msg = f"unknown layout kind {kind!r}"
            raise ValueError(msg)

    return SynthPage(
        kind="layout_page",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths),
        meta=meta,
    )


def _reserve(
    rng: random.Random,
    *,
    width: int,
    height: int,
    box_w: int,
    box_h: int,
    occupied: list[tuple[int, int, int, int]],
    margin: int = 16,
    attempts: int = 80,
) -> tuple[int, int, int, int]:
    """Pick a free axis-aligned box; fall back to a least-overlap slot."""
    for _ in range(attempts):
        x0 = rng.randint(margin, max(margin, width - box_w - margin))
        y0 = rng.randint(margin, max(margin, height - box_h - margin))
        cand = (x0, y0, x0 + box_w, y0 + box_h)
        if all(not _overlaps(cand, other, pad=12) for other in occupied):
            return cand
    # Deterministic fallback: stack from top using seed jitter on x.
    y0 = margin
    for other in occupied:
        y0 = max(y0, other[3] + 12)
    x0 = margin + rng.randint(0, max(0, width - box_w - 2 * margin))
    y0 = min(y0, max(margin, height - box_h - margin))
    return (x0, y0, x0 + box_w, y0 + box_h)


def _overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _place_text_block(
    rng: random.Random,
    draw: ImageDraw.ImageDraw,
    font: object,
    width: int,
    height: int,
    page_index: int,
    seed: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], list[RegionTruth]]:
    lines = list(_TEXT_LINES)
    rng.shuffle(lines)
    n = rng.randint(2, min(4, len(lines)))
    chosen = lines[:n]

    line_h = 28
    max_tw = 0
    for text in chosen:
        tw, _ = text_size(draw, text, font)  # type: ignore[arg-type]
        max_tw = max(max_tw, tw)
    box_w = max_tw + 24
    box_h = n * line_h + 16
    outer = _reserve(
        rng, width=width, height=height, box_w=box_w, box_h=box_h, occupied=occupied
    )
    x0, y0, _, _ = outer

    truths: list[RegionTruth] = []
    for i, text in enumerate(chosen):
        ly = y0 + 8 + i * line_h
        lx = x0 + 8
        draw.text((lx, ly), text, fill=(20, 20, 20), font=font)  # type: ignore[arg-type]
        tw, th = text_size(draw, text, font)  # type: ignore[arg-type]
        bbox = pixel_bbox_to_norm(
            x0=lx - 2,
            y0=ly - 2,
            x1=lx + tw + 2,
            y1=ly + th + 2,
            width=width,
            height=height,
        )
        region = Region(
            id=f"text_line-{seed}-{i}",
            page_index=page_index,
            bbox=bbox,
            kind=RegionKind.TEXT_LINE,
        )
        truths.append(RegionTruth(region=region, text=text))

    # Block = union of lines (same kind; id marks coalesced block for eval).
    block_bbox = pixel_bbox_to_norm(
        x0=outer[0],
        y0=outer[1],
        x1=outer[2],
        y1=outer[3],
        width=width,
        height=height,
    )
    truths.append(
        RegionTruth(
            region=Region(
                id=f"text_block-{seed}",
                page_index=page_index,
                bbox=block_bbox,
                kind=RegionKind.TEXT_LINE,
            ),
            text="\n".join(chosen),
        )
    )
    return outer, truths


def _place_table(
    rng: random.Random,
    draw: ImageDraw.ImageDraw,
    font: object,
    width: int,
    height: int,
    page_index: int,
    seed: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], list[RegionTruth]]:
    cols = 3
    rows = 1 + len(_TABLE_CELLS)  # header + body
    cell_w = 70 + rng.randint(0, 20)
    cell_h = 28 + rng.randint(0, 8)
    box_w = cols * cell_w + 4
    box_h = rows * cell_h + 4
    outer = _reserve(
        rng, width=width, height=height, box_w=box_w, box_h=box_h, occupied=occupied
    )
    ox, oy = outer[0], outer[1]

    # Grid lines (width 3 so classic morphology keeps them as rulings).
    for r in range(rows + 1):
        y = oy + r * cell_h
        draw.line((ox, y, ox + cols * cell_w, y), fill=(40, 40, 40), width=3)
    for c in range(cols + 1):
        x = ox + c * cell_w
        draw.line((x, oy, x, oy + rows * cell_h), fill=(40, 40, 40), width=3)

    truths: list[RegionTruth] = []
    # Header row.
    for c, header in enumerate(_TABLE_HEADERS):
        cx0 = ox + c * cell_w + 4
        cy0 = oy + 4
        draw.text((cx0, cy0), header, fill=(15, 15, 15), font=font)  # type: ignore[arg-type]
        cell = (
            ox + c * cell_w,
            oy,
            ox + (c + 1) * cell_w,
            oy + cell_h,
        )
        truths.append(
            RegionTruth(
                region=Region(
                    id=f"table_cell-{seed}-h{c}",
                    page_index=page_index,
                    bbox=pixel_bbox_to_norm(
                        x0=cell[0],
                        y0=cell[1],
                        x1=cell[2],
                        y1=cell[3],
                        width=width,
                        height=height,
                    ),
                    kind=RegionKind.TABLE_CELL,
                ),
                text=header,
            )
        )

    for r, row in enumerate(_TABLE_CELLS):
        for c, value in enumerate(row):
            cx0 = ox + c * cell_w + 4
            cy0 = oy + (r + 1) * cell_h + 4
            draw.text((cx0, cy0), value, fill=(15, 15, 15), font=font)  # type: ignore[arg-type]
            cell = (
                ox + c * cell_w,
                oy + (r + 1) * cell_h,
                ox + (c + 1) * cell_w,
                oy + (r + 2) * cell_h,
            )
            truths.append(
                RegionTruth(
                    region=Region(
                        id=f"table_cell-{seed}-r{r}c{c}",
                        page_index=page_index,
                        bbox=pixel_bbox_to_norm(
                            x0=cell[0],
                            y0=cell[1],
                            x1=cell[2],
                            y1=cell[3],
                            width=width,
                            height=height,
                        ),
                        kind=RegionKind.TABLE_CELL,
                    ),
                    text=value,
                )
            )

    return outer, truths


def _place_stamp(
    rng: random.Random,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    page_index: int,
    seed: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], RegionTruth]:
    radius = 48 + rng.randint(0, 12)
    box_w = box_h = radius * 2 + 8
    outer = _reserve(
        rng, width=width, height=height, box_w=box_w, box_h=box_h, occupied=occupied
    )
    cx = (outer[0] + outer[2]) // 2
    cy = (outer[1] + outer[3]) // 2
    red = (190, 45, 45)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=red,
        width=4,
    )
    inner = int(radius * 0.75)
    draw.ellipse(
        (cx - inner, cy - inner, cx + inner, cy + inner),
        outline=red,
        width=2,
    )
    font = load_printed_font(size=14)
    label = "RECV"
    tw, th = text_size(draw, label, font)
    draw.text((cx - tw // 2, cy - th // 2), label, fill=red, font=font)

    bbox = pixel_bbox_to_norm(
        x0=cx - radius,
        y0=cy - radius,
        x1=cx + radius,
        y1=cy + radius,
        width=width,
        height=height,
    )
    _ = image  # kept for API symmetry with photo placer
    return outer, RegionTruth(
        region=Region(
            id=f"stamp-{seed}",
            page_index=page_index,
            bbox=bbox,
            kind=RegionKind.STAMP,
        ),
        text=label,
    )


def _place_signature(
    rng: random.Random,
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    page_index: int,
    seed: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], RegionTruth]:
    box_w = 160 + rng.randint(0, 40)
    box_h = 48 + rng.randint(0, 16)
    outer = _reserve(
        rng, width=width, height=height, box_w=box_w, box_h=box_h, occupied=occupied
    )
    x0, y0, x1, y1 = outer
    # Sparse polyline strokes (signature-like), not a filled blob.
    points: list[tuple[int, int]] = []
    n = 12 + rng.randint(0, 6)
    for i in range(n):
        px = x0 + 8 + int((x1 - x0 - 16) * i / max(n - 1, 1))
        py = y0 + box_h // 2 + int(rng.uniform(-box_h * 0.35, box_h * 0.35))
        points.append((px, py))
    draw.line(points, fill=(10, 10, 30), width=3)
    # Second overlapping stroke.
    points2 = [
        (p[0] + rng.randint(-2, 2), p[1] + rng.randint(-4, 4)) for p in points[1:-1]
    ]
    if len(points2) >= 2:
        draw.line(points2, fill=(20, 20, 40), width=2)
    # Underscore baseline makes the stroke cluster more elongated.
    draw.line(
        (x0 + 6, y1 - 6, x1 - 6, y1 - 8),
        fill=(25, 25, 40),
        width=2,
    )

    bbox = pixel_bbox_to_norm(x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height)
    return outer, RegionTruth(
        region=Region(
            id=f"signature-{seed}",
            page_index=page_index,
            bbox=bbox,
            kind=RegionKind.SIGNATURE,
        ),
        text=None,
    )


def _place_photo(
    rng: random.Random,
    image: Image.Image,
    width: int,
    height: int,
    page_index: int,
    seed: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], RegionTruth]:
    box_w = 120 + rng.randint(0, 40)
    box_h = 140 + rng.randint(0, 40)
    outer = _reserve(
        rng, width=width, height=height, box_w=box_w, box_h=box_h, occupied=occupied
    )
    x0, y0, x1, y1 = outer
    # Continuous-tone patch (high colour variance, low binary ink).
    pixels = image.load()
    assert pixels is not None
    for y in range(y0, y1):
        for x in range(x0, x1):
            # Smooth-ish gradient + noise so it looks photo-like, not text.
            t = (x - x0) / max(box_w, 1)
            u = (y - y0) / max(box_h, 1)
            r = int(80 + 100 * t + rng.randint(-12, 12))
            g = int(90 + 80 * u + rng.randint(-12, 12))
            b = int(110 + 60 * (1 - t) + rng.randint(-12, 12))
            pixels[x, y] = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
            )
    draw = ImageDraw.Draw(image)
    draw.rectangle(outer, outline=(50, 50, 50), width=2)

    bbox = pixel_bbox_to_norm(x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height)
    return outer, RegionTruth(
        region=Region(
            id=f"photo-{seed}",
            page_index=page_index,
            bbox=bbox,
            kind=RegionKind.PHOTO,
        ),
        text=None,
    )
