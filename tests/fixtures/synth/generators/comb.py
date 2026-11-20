"""Comb-row (segmented character boxes) generator."""

from __future__ import annotations

import random
from typing import Literal

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import (
    load_handwriting_font,
    load_printed_font,
    text_size,
)
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import LabelAnchor, Region, RegionKind

_DEFAULT_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

CombStyle = Literal["boxed", "ticks"]


def generate_comb_row(
    seed: int = 0,
    *,
    cells: int | None = None,
    text: str | None = None,
    width: int = 640,
    height: int = 160,
    page_index: int = 0,
    style: CombStyle | None = None,
    crossing_strokes: bool | None = None,
) -> SynthPage:
    """Draw a labelled comb row and return image + ground truth.

    Layout positions are derived from ``seed`` (no fixed page coordinates in
    production code — this is a test fixture only).

    ``style``:
      - ``boxed``: separate rectangles per cell (small gaps)
      - ``ticks``: shared top/bottom rules with vertical dividers (classic comb)

    When ``crossing_strokes`` is true, characters (and optional ink bridges)
    intentionally overlap cell dividers so inpainting must preserve strokes.
    """
    rng = random.Random(seed)
    if cells is None:
        cells = rng.randint(4, 10)
    if cells < 1:
        msg = f"cells must be >= 1, got {cells}"
        raise ValueError(msg)
    if text is None:
        text = "".join(rng.choice(_DEFAULT_CHARS) for _ in range(cells))
    elif len(text) != cells:
        msg = f"text length {len(text)} must equal cells {cells}"
        raise ValueError(msg)

    if style is None:
        style = "ticks" if rng.random() < 0.5 else "boxed"
    if crossing_strokes is None:
        crossing_strokes = rng.random() < 0.45

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = load_printed_font(size=18)
    value_font = (
        load_handwriting_font(size=28)
        if crossing_strokes
        else load_printed_font(size=22)
    )

    label = "CODE"
    label_x = 20 + rng.randint(0, 40)
    label_y = 40 + rng.randint(0, 30)
    tw, th = text_size(draw, label, font)
    draw.text((label_x, label_y), label, fill=(20, 20, 20), font=font)
    label_bbox = pixel_bbox_to_norm(
        x0=label_x,
        y0=label_y,
        x1=label_x + tw,
        y1=label_y + th,
        width=width,
        height=height,
    )

    cell_w = 32 + rng.randint(0, 10)
    cell_h = 40 + rng.randint(0, 10)
    gap = 0 if style == "ticks" else 1 + rng.randint(0, 2)
    comb_x0 = label_x + tw + 16
    comb_y0 = label_y - 8
    # Keep the comb on-page.
    max_x = width - 16 - cells * (cell_w + gap)
    if comb_x0 > max_x:
        comb_x0 = max(16, max_x)
    comb_x1 = comb_x0 + cells * (cell_w + gap) - gap
    comb_y1 = comb_y0 + cell_h

    line = (30, 30, 30)
    if style == "ticks":
        # Outer frame + vertical dividers (shared walls).
        draw.rectangle((comb_x0, comb_y0, comb_x1, comb_y1), outline=line, width=2)
        for i in range(1, cells):
            x = comb_x0 + i * cell_w
            draw.line((x, comb_y0, x, comb_y1), fill=line, width=2)
    else:
        for i in range(cells):
            x0 = comb_x0 + i * (cell_w + gap)
            y0 = comb_y0
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle((x0, y0, x1, y1), outline=line, width=2)
            draw.line((x0, y1 - 8, x0, y1), fill=line, width=2)

    cell_bboxes_px: list[tuple[int, int, int, int]] = []
    cell_bboxes_norm: list[tuple[float, float, float, float]] = []
    for i, ch in enumerate(text):
        if style == "ticks":
            x0 = comb_x0 + i * cell_w
            x1 = comb_x0 + (i + 1) * cell_w
        else:
            x0 = comb_x0 + i * (cell_w + gap)
            x1 = x0 + cell_w
        y0 = comb_y0
        y1 = comb_y0 + cell_h
        cell_bboxes_px.append((x0, y0, x1, y1))
        nb = pixel_bbox_to_norm(x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height)
        cell_bboxes_norm.append((nb.x0, nb.y0, nb.x1, nb.y1))

        cw, ch_h = text_size(draw, ch, value_font)
        # Crossing strokes: bias glyph toward the right divider so ink overlaps.
        if crossing_strokes:
            tx = x0 + max(1, cell_w - cw - 1)
            ty = y0 + max(0, (cell_h - ch_h) // 2 - 1)
        else:
            tx = x0 + (cell_w - cw) / 2
            ty = y0 + (cell_h - ch_h) / 2 - 2
        draw.text((tx, ty), ch, fill=(10, 10, 10), font=value_font)

        if crossing_strokes and i < cells - 1:
            # Short ink bridge across the shared/near divider.
            mid_x = x1 if style == "ticks" else x1 + gap // 2
            mid_y = y0 + cell_h // 2 + rng.randint(-4, 4)
            draw.line(
                (mid_x - 4, mid_y, mid_x + 4, mid_y + rng.randint(-2, 2)),
                fill=(10, 10, 10),
                width=2,
            )

    value_bbox = pixel_bbox_to_norm(
        x0=comb_x0,
        y0=comb_y0,
        x1=comb_x1,
        y1=comb_y1,
        width=width,
        height=height,
    )
    region = Region(
        id=f"comb-{seed}",
        page_index=page_index,
        bbox=value_bbox,
        kind=RegionKind.COMB,
        label_anchor=LabelAnchor(text=label, bbox=label_bbox),
    )
    return SynthPage(
        kind="comb_row",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=(RegionTruth(region=region, text=text),),
        meta={
            "cells": cells,
            "label": label,
            "style": style,
            "crossing_strokes": crossing_strokes,
            "cell_bboxes_px": cell_bboxes_px,
            "cell_bboxes_norm": cell_bboxes_norm,
            "cell_chars": list(text),
        },
    )
