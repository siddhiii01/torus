"""Single comb-cell glyph generator for the character-cell reader (P5.5 / S8)."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import (
    load_handwriting_font,
    load_printed_font,
    text_size,
)
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind

_DEFAULT_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_char_cell(
    seed: int = 0,
    *,
    character: str | None = None,
    width: int = 64,
    height: int = 64,
    page_index: int = 0,
    handwriting: bool | None = None,
) -> SynthPage:
    """Draw one character in a cell-like box; return image + ground truth.

    Layout offsets come from ``seed`` only (fixture code — not production).
    """
    rng = random.Random(seed)
    if character is None:
        character = rng.choice(_DEFAULT_CHARS)
    if len(character) != 1:
        msg = f"character must be length 1, got {character!r}"
        raise ValueError(msg)
    if handwriting is None:
        handwriting = rng.random() < 0.6

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Outer cell rule (thin), inset from the page edge. The Region bbox is the
    # *inner* content area (matches P4.3 cell crops that inset from dividers),
    # so the ruling does not dominate the classifier features.
    margin = 4 + rng.randint(0, 4)
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(40, 40, 40), width=1)
    inset = 3
    ix0, iy0 = x0 + inset, y0 + inset
    ix1, iy1 = x1 - inset, y1 - inset

    font = (
        load_handwriting_font(size=28 + rng.randint(0, 8))
        if handwriting
        else load_printed_font(size=26 + rng.randint(0, 6))
    )
    tw, th = text_size(draw, character, font)
    tx = ix0 + (ix1 - ix0 - tw) / 2 + rng.uniform(-2.0, 2.0)
    ty = iy0 + (iy1 - iy0 - th) / 2 + rng.uniform(-2.0, 2.0)
    shade = int(rng.randint(5, 45))
    draw.text((tx, ty), character, fill=(shade, shade, shade), font=font)

    bbox = pixel_bbox_to_norm(
        x0=ix0,
        y0=iy0,
        x1=ix1,
        y1=iy1,
        width=width,
        height=height,
    )
    region = Region(
        id=f"char-cell-{seed}",
        page_index=page_index,
        bbox=bbox,
        kind=RegionKind.FIELD_VALUE,
    )
    return SynthPage(
        kind="char_cell",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=(RegionTruth(region=region, text=character),),
        meta={"character": character, "handwriting": handwriting},
    )
