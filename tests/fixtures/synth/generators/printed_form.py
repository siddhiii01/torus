"""Printed form page with labelled field values (random layout per seed)."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import LabelAnchor, Region, RegionKind

# Generic placeholder content only — no real names, places, or IDs.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("REF", "AB12CD"),
    ("QTY", "42"),
    ("CODE", "ZX9K"),
    ("DATE", "01-01-2000"),
    ("NOTE", "SAMPLE"),
)


def generate_printed_form(
    seed: int = 0,
    *,
    field_count: int | None = None,
    width: int = 800,
    height: int = 1100,
    page_index: int = 0,
) -> SynthPage:
    """Draw a sparse printed form; field positions vary with ``seed``."""
    rng = random.Random(seed)
    if field_count is None:
        field_count = rng.randint(2, min(4, len(_FIELDS)))
    if not 1 <= field_count <= len(_FIELDS):
        msg = f"field_count must be in 1..{len(_FIELDS)}, got {field_count}"
        raise ValueError(msg)

    chosen = list(_FIELDS)
    rng.shuffle(chosen)
    chosen = chosen[:field_count]

    image = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(image)
    title_font = load_printed_font(size=28)
    body_font = load_printed_font(size=20)

    title = "SYNTH FORM"
    draw.text((40, 30), title, fill=(15, 15, 15), font=title_font)
    draw.line((40, 70, width - 40, 70), fill=(80, 80, 80), width=2)

    truths: list[RegionTruth] = []
    # Stagger rows; horizontal offset jittered by seed so layouts are not fixed.
    base_y = 110 + rng.randint(0, 40)
    for i, (label, value) in enumerate(chosen):
        row_y = base_y + i * (70 + rng.randint(0, 20))
        label_x = 40 + rng.randint(0, 80)
        lw, lh = text_size(draw, label, body_font)
        draw.text((label_x, row_y), f"{label}:", fill=(30, 30, 30), font=body_font)
        label_bbox = pixel_bbox_to_norm(
            x0=label_x,
            y0=row_y,
            x1=label_x + lw + 4,
            y1=row_y + lh,
            width=width,
            height=height,
        )

        value_x = label_x + lw + 28 + rng.randint(0, 40)
        vw, vh = text_size(draw, value, body_font)
        # Underline box for the value.
        pad = 6
        box = (
            value_x - pad,
            row_y - 4,
            value_x + vw + pad,
            row_y + vh + 8,
        )
        draw.rectangle(box, outline=(60, 60, 60), width=1)
        draw.text((value_x, row_y), value, fill=(10, 10, 10), font=body_font)
        value_bbox = pixel_bbox_to_norm(
            x0=box[0],
            y0=box[1],
            x1=box[2],
            y1=box[3],
            width=width,
            height=height,
        )
        region = Region(
            id=f"field-{seed}-{i}",
            page_index=page_index,
            bbox=value_bbox,
            kind=RegionKind.FIELD_VALUE,
            label_anchor=LabelAnchor(text=label, bbox=label_bbox),
        )
        truths.append(RegionTruth(region=region, text=value))

    return SynthPage(
        kind="printed_form",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths),
        meta={"field_count": field_count},
    )
