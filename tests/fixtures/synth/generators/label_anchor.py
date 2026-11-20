"""Synthetic labelled form for P4.2: random layout + synonym labels.

Positions and which synonym is drawn vary with the seed so tests prove there
are no fixed page coordinates in the anchorer.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, load_script_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind

# Generic placeholder content only — no real names, places, or IDs.
# field_key → (canonical header, synonym variants, sample value)
_FIELD_DEFS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("mobile_no", "Mobile No", ("Mobile", "Mob. No.", "Cell"), "9876543210"),
    ("address_pincode", "PIN", ("PIN Code", "Pincode", "Zip"), "560001"),
    ("person_name", "Name", ("Full Name", "Applicant Name"), "SAMPLE PERSON"),
    ("notes", "Notes", ("Remark", "Comments"), "OK"),
)

# (field_key, canonical header, synonym variants, sample value)
FieldDef = tuple[str, str, tuple[str, ...], str]

Placement = Literal["right", "below"]


def generate_label_anchor_page(
    seed: int = 0,
    *,
    field_count: int | None = None,
    width: int = 900,
    height: int = 1200,
    page_index: int = 0,
    field_defs: Sequence[FieldDef] | None = None,
    script: str | None = None,
) -> SynthPage:
    """Draw label+value pairs at seed-dependent positions and placements.

    ``field_defs`` lets a test onboard another template's columns (P12.1);
    ``script`` draws with a font for that script (P12.2) and raises
    ``LookupError`` when none is installed so callers can skip.
    """
    rng = random.Random(seed)
    defs = list(field_defs if field_defs is not None else _FIELD_DEFS)
    rng.shuffle(defs)
    if field_count is None:
        field_count = rng.randint(2, len(defs))
    if not 1 <= field_count <= len(defs):
        msg = f"field_count must be in 1..{len(defs)}, got {field_count}"
        raise ValueError(msg)
    chosen = defs[:field_count]

    image = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_printed_font(size=26)
    body_font = load_printed_font(size=18)
    if script is not None:
        script_font = load_script_font(script, size=18)
        if script_font is None:
            raise LookupError(f"no font installed for script {script!r}")
        body_font = script_font
    draw.text((36, 28), "SYNTH LABEL FORM", fill=(20, 20, 20), font=title_font)
    draw.line((36, 64, width - 36, 64), fill=(90, 90, 90), width=2)

    truths: list[RegionTruth] = []
    meta_fields: list[dict[str, object]] = []
    occupied_y: list[tuple[float, float]] = []

    for i, (field_key, header, synonyms, value) in enumerate(chosen):
        label = rng.choice((header, *synonyms))
        placement: Placement = rng.choice(("right", "below"))

        # Place rows with vertical jitter; reject heavy overlap with prior rows.
        for _attempt in range(40):
            row_y = 90 + rng.randint(0, height - 220)
            label_x = 40 + rng.randint(0, 120)
            band = (row_y / height, (row_y + 70) / height)
            if any(not (band[1] < a or band[0] > b) for a, b in occupied_y):
                continue
            occupied_y.append(band)
            break
        else:
            row_y = 100 + i * 110 + rng.randint(0, 30)
            label_x = 40 + rng.randint(0, 80)

        lw, lh = text_size(draw, f"{label}:", body_font)
        draw.text((label_x, row_y), f"{label}:", fill=(25, 25, 25), font=body_font)
        label_bbox = pixel_bbox_to_norm(
            x0=label_x,
            y0=row_y,
            x1=label_x + lw + 4,
            y1=row_y + lh,
            width=width,
            height=height,
        )
        label_region = Region(
            id=f"label-{seed}-{i}",
            page_index=page_index,
            bbox=label_bbox,
            kind=RegionKind.TEXT_LINE,
        )

        vw, vh = text_size(draw, value, body_font)
        pad = 5
        if placement == "right":
            value_x = label_x + lw + 24 + rng.randint(8, 48)
            value_y = row_y
        else:
            value_x = label_x + rng.randint(0, 20)
            value_y = row_y + lh + 10 + rng.randint(0, 12)

        box = (
            value_x - pad,
            value_y - 3,
            value_x + vw + pad,
            value_y + vh + 6,
        )
        draw.rectangle(box, outline=(70, 70, 70), width=1)
        draw.text((value_x, value_y), value, fill=(10, 10, 10), font=body_font)
        value_bbox = pixel_bbox_to_norm(
            x0=box[0],
            y0=box[1],
            x1=box[2],
            y1=box[3],
            width=width,
            height=height,
        )
        value_region = Region(
            id=f"value-{seed}-{i}",
            page_index=page_index,
            bbox=value_bbox,
            kind=RegionKind.TEXT_LINE,
        )

        # Label region carries the printed text; value region carries the value.
        truths.append(RegionTruth(region=label_region, text=label))
        truths.append(RegionTruth(region=value_region, text=value))
        meta_fields.append(
            {
                "field_key": field_key,
                "label_region_id": label_region.id,
                "value_region_id": value_region.id,
                "label_drawn": label,
                "placement": placement,
            }
        )

    return SynthPage(
        kind="label_anchor",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths),
        meta={"fields": meta_fields, "field_count": field_count},
    )
