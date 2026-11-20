"""Checkbox / radio field generators (P0.7 + P4.4).

Layouts are randomised from the seed — no fixed page coordinates.
"""

from __future__ import annotations

import random
from typing import Literal

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import LabelAnchor, Region, RegionKind

_LABELS = ("Option A", "Option B", "Yes", "No", "Agree", "Male", "Female")

CheckboxShape = Literal["square", "circle"]
CheckboxMark = Literal["empty", "tick", "cross"]
LabelSide = Literal["right", "left"]


def generate_checkbox(
    seed: int = 0,
    *,
    checked: bool | None = None,
    label: str | None = None,
    shape: CheckboxShape | None = None,
    mark: CheckboxMark | None = None,
    label_side: LabelSide | None = None,
    width: int = 480,
    height: int = 240,
    page_index: int = 0,
) -> SynthPage:
    """Draw a labelled checkbox or radio with empty / tick / cross fill.

    ``checked`` remains for older callers: True → mark (default cross),
    False → empty. Explicit ``mark`` wins when provided.
    """
    rng = random.Random(seed)
    if shape is None:
        shape = "circle" if rng.random() < 0.35 else "square"
    if mark is None:
        if checked is None:
            mark = rng.choice(("empty", "tick", "cross"))
        else:
            mark = "cross" if checked else "empty"
    if label is None:
        label = _LABELS[rng.randint(0, len(_LABELS) - 1)]
    if label_side is None:
        label_side = "left" if rng.random() < 0.2 else "right"

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = load_printed_font(size=20)

    box_size = 28 + rng.randint(0, 8)
    # Random placement — still leaves room for the label on either side.
    margin = 16
    label_reserve = 160
    if label_side == "right":
        x0 = margin + rng.randint(0, max(0, width - box_size - label_reserve - margin))
    else:
        x0 = label_reserve + rng.randint(
            0, max(0, width - box_size - label_reserve - margin)
        )
    y0 = margin + rng.randint(0, max(0, height - box_size - margin * 2))
    x1 = x0 + box_size
    y1 = y0 + box_size

    _draw_control(draw, shape, (x0, y0, x1, y1), mark)

    tw, th = text_size(draw, label, font)
    if label_side == "right":
        lx = x1 + 10 + rng.randint(0, 8)
        ly = y0 + max(0, (box_size - th) // 2)
    else:
        lx = max(margin, x0 - tw - 10 - rng.randint(0, 8))
        ly = y0 + max(0, (box_size - th) // 2)
    draw.text((lx, ly), label, fill=(20, 20, 20), font=font)

    label_bbox = pixel_bbox_to_norm(
        x0=lx, y0=ly, x1=lx + tw, y1=ly + th, width=width, height=height
    )
    box_bbox = pixel_bbox_to_norm(
        x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height
    )
    is_checked = mark != "empty"
    region = Region(
        id=f"checkbox-{seed}",
        page_index=page_index,
        bbox=box_bbox,
        kind=RegionKind.CHECKBOX,
        label_anchor=LabelAnchor(text=label, bbox=label_bbox),
    )
    label_region = Region(
        id=f"checkbox-{seed}-label",
        page_index=page_index,
        bbox=label_bbox,
        kind=RegionKind.TEXT_LINE,
    )
    return SynthPage(
        kind="checkbox",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=(
            RegionTruth(
                region=region,
                text=_mark_to_text(mark),
                checked=is_checked,
            ),
            RegionTruth(region=label_region, text=label, checked=None),
        ),
        meta={
            "label": label,
            "shape": shape,
            "mark": mark,
            "label_side": label_side,
            "fill_state": mark if mark != "tick" else "ticked",
        },
    )


def generate_checkbox_group(
    seed: int = 0,
    *,
    n_options: int | None = None,
    width: int = 520,
    height: int | None = None,
    page_index: int = 0,
    shape: CheckboxShape | None = None,
) -> SynthPage:
    """Vertical list of checkbox/radio options with randomised marks."""
    rng = random.Random(seed)
    if n_options is None:
        n_options = rng.randint(2, 4)
    if shape is None:
        shape = "circle" if rng.random() < 0.4 else "square"
    row_h = 48
    if height is None:
        height = max(120, n_options * row_h + 40)

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = load_printed_font(size=18)

    # Pick one ticked option for radios; checkboxes may have several.
    marks: list[CheckboxMark] = []
    if shape == "circle":
        chosen = rng.randint(0, n_options - 1)
        marks = ["empty"] * n_options
        marks[chosen] = "tick" if rng.random() < 0.6 else "cross"
    else:
        for _ in range(n_options):
            marks.append(rng.choice(("empty", "tick", "cross")))

    labels = list(_LABELS)
    rng.shuffle(labels)
    labels = labels[:n_options]

    truths: list[RegionTruth] = []
    # Random left margin so the column is not at a fixed x.
    col_x = 20 + rng.randint(0, 80)
    top = 16 + rng.randint(0, 20)

    for i, (lab, mark) in enumerate(zip(labels, marks, strict=True)):
        box_size = 26 + rng.randint(0, 6)
        x0 = col_x + rng.randint(0, 12)
        y0 = top + i * row_h + rng.randint(0, 8)
        x1 = x0 + box_size
        y1 = y0 + box_size
        _draw_control(draw, shape, (x0, y0, x1, y1), mark)

        tw, th = text_size(draw, lab, font)
        lx = x1 + 12 + rng.randint(0, 10)
        ly = y0 + max(0, (box_size - th) // 2)
        draw.text((lx, ly), lab, fill=(20, 20, 20), font=font)

        box_bbox = pixel_bbox_to_norm(
            x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height
        )
        label_bbox = pixel_bbox_to_norm(
            x0=lx, y0=ly, x1=lx + tw, y1=ly + th, width=width, height=height
        )
        region = Region(
            id=f"checkbox-group-{seed}-{i}",
            page_index=page_index,
            bbox=box_bbox,
            kind=RegionKind.CHECKBOX,
            label_anchor=LabelAnchor(text=lab, bbox=label_bbox),
        )
        label_region = Region(
            id=f"checkbox-group-{seed}-{i}-label",
            page_index=page_index,
            bbox=label_bbox,
            kind=RegionKind.TEXT_LINE,
        )
        truths.append(
            RegionTruth(
                region=region,
                text=_mark_to_text(mark),
                checked=mark != "empty",
            )
        )
        truths.append(RegionTruth(region=label_region, text=lab, checked=None))

    return SynthPage(
        kind="checkbox",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=tuple(truths),
        meta={
            "n_options": n_options,
            "shape": shape,
            "marks": list(marks),
            "labels": list(labels),
        },
    )


def _mark_to_text(mark: CheckboxMark) -> str:
    if mark == "empty":
        return "false"
    if mark == "tick":
        return "ticked"
    return "crossed"


def _draw_control(
    draw: ImageDraw.ImageDraw,
    shape: CheckboxShape,
    box: tuple[int, int, int, int],
    mark: CheckboxMark,
) -> None:
    x0, y0, x1, y1 = box
    if shape == "square":
        draw.rectangle((x0, y0, x1, y1), outline=(20, 20, 20), width=2)
    else:
        draw.ellipse((x0, y0, x1, y1), outline=(20, 20, 20), width=2)

    if mark == "empty":
        return

    pad = max(3, (x1 - x0) // 6)
    if mark == "cross":
        draw.line((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=(10, 10, 10), width=3)
        draw.line((x0 + pad, y1 - pad, x1 - pad, y0 + pad), fill=(10, 10, 10), width=3)
        return

    # Tick: two-segment check mark.
    mid_x = x0 + (x1 - x0) // 3
    mid_y = y1 - pad
    draw.line((x0 + pad, y0 + (y1 - y0) // 2, mid_x, mid_y), fill=(10, 10, 10), width=3)
    draw.line((mid_x, mid_y, x1 - pad, y0 + pad), fill=(10, 10, 10), width=3)
