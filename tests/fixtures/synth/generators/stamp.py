"""Stamp- and signature-over-text generators."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from tests.fixtures.synth.fonts import load_printed_font, text_size
from tests.fixtures.synth.generators.printed_form import generate_printed_form
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind


def generate_stamp(
    seed: int = 0,
    *,
    stamp_text: str = "RECEIVED",
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Overlay a translucent circular stamp that may cover field text."""
    rng = random.Random(seed)
    source = base if base is not None else generate_printed_form(seed, field_count=3)
    image = source.image.copy().convert("RGBA")

    # Stamp centre biased toward the upper field area so it often overlaps text.
    cx = int(image.width * (0.35 + 0.35 * rng.random()))
    cy = int(image.height * (0.18 + 0.25 * rng.random()))
    radius = int(55 + 25 * rng.random())

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    # Outer ring + inner ring.
    red = (180, 40, 40, 140)
    odraw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=red,
        width=4,
    )
    inner = int(radius * 0.78)
    odraw.ellipse(
        (cx - inner, cy - inner, cx + inner, cy + inner),
        outline=red,
        width=2,
    )

    font = load_printed_font(size=18)
    # Measure on a throwaway RGB draw context via temporary image.
    measure = Image.new("RGB", (1, 1))
    mdraw = ImageDraw.Draw(measure)
    tw, th = text_size(mdraw, stamp_text, font)
    # Draw stamp text slightly tilted via paste of a rotated tile.
    tile_w = tw + 20
    tile_h = th + 16
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(tile)
    tdraw.text(
        ((tile_w - tw) / 2, (tile_h - th) / 2),
        stamp_text,
        fill=red,
        font=font,
    )
    angle = -18.0 + 36.0 * rng.random()
    rotated_tile = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    tx = cx - rotated_tile.width // 2
    ty = cy - rotated_tile.height // 2
    overlay.alpha_composite(rotated_tile, (tx, ty))

    composed = Image.alpha_composite(image, overlay).convert("RGB")

    stamp_bbox = pixel_bbox_to_norm(
        x0=cx - radius,
        y0=cy - radius,
        x1=cx + radius,
        y1=cy + radius,
        width=composed.width,
        height=composed.height,
    )
    stamp_region = Region(
        id=f"stamp-{seed}",
        page_index=source.page_index if page_index is None else page_index,
        bbox=stamp_bbox,
        kind=RegionKind.STAMP,
    )
    stamp_truth = RegionTruth(region=stamp_region, text=stamp_text)

    idx = source.page_index if page_index is None else page_index
    # Preserve underlying field truth; stamp is an extra region.
    regions = tuple(
        RegionTruth(
            region=Region(
                id=t.region.id,
                page_index=idx,
                bbox=t.region.bbox,
                kind=t.region.kind,
                label_anchor=t.region.label_anchor,
            ),
            text=t.text,
            checked=t.checked,
        )
        for t in source.regions
    ) + (stamp_truth,)

    return SynthPage(
        kind="stamp",
        seed=seed,
        image=composed,
        page_index=idx,
        width=composed.width,
        height=composed.height,
        orientation=source.orientation,
        regions=regions,
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "stamp_text": stamp_text,
            "stamp_angle_deg": angle,
            # Include a deterministic angle sine so meta stays useful in tests.
            "stamp_angle_sin": math.sin(math.radians(angle)),
            "stamped": "yes",
        },
    )


def generate_signature(
    seed: int = 0,
    *,
    present: bool = True,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Overlay sparse signature strokes that may cover field text.

    When ``present`` is False the base form is returned unchanged with
    ``meta["signed"] = "no"`` and no signature region (signed yes/no columns).
    """
    rng = random.Random(seed)
    source = base if base is not None else generate_printed_form(seed, field_count=3)
    idx = source.page_index if page_index is None else page_index

    def _rebase(regions: tuple[RegionTruth, ...]) -> tuple[RegionTruth, ...]:
        return tuple(
            RegionTruth(
                region=Region(
                    id=t.region.id,
                    page_index=idx,
                    bbox=t.region.bbox,
                    kind=t.region.kind,
                    label_anchor=t.region.label_anchor,
                ),
                text=t.text,
                checked=t.checked,
            )
            for t in regions
        )

    if not present:
        return SynthPage(
            kind="signature",
            seed=seed,
            image=source.image.copy().convert("RGB"),
            page_index=idx,
            width=source.width,
            height=source.height,
            orientation=source.orientation,
            regions=_rebase(source.regions),
            meta={
                "base_kind": source.kind,
                "base_seed": source.seed,
                "signed": "no",
                "present": False,
            },
        )

    image = source.image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    # Wide, short band in the lower half (signature block), away from dense
    # field text so classic stroke clustering stays sparse.
    box_w = 170 + rng.randint(0, 50)
    box_h = 48 + rng.randint(0, 12)
    x0 = int(image.width * (0.2 + 0.35 * rng.random()))
    y0 = int(image.height * (0.55 + 0.25 * rng.random()))
    x0 = max(8, min(image.width - box_w - 8, x0))
    y0 = max(8, min(image.height - box_h - 8, y0))
    x1, y1 = x0 + box_w, y0 + box_h
    # Clear paper under the strokes so ink does not fuse with printed text.
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))

    points: list[tuple[int, int]] = []
    n = 14 + rng.randint(0, 6)
    for i in range(n):
        px = x0 + 10 + int((box_w - 20) * i / max(n - 1, 1))
        py = y0 + box_h // 2 + int(rng.uniform(-box_h * 0.32, box_h * 0.28))
        points.append((px, py))
    draw.line(points, fill=(10, 10, 30), width=3)
    points2 = [
        (p[0] + rng.randint(-2, 2), p[1] + rng.randint(-4, 4)) for p in points[1:-1]
    ]
    if len(points2) >= 2:
        draw.line(points2, fill=(20, 20, 40), width=2)
    draw.line((x0 + 8, y1 - 7, x1 - 8, y1 - 9), fill=(25, 25, 40), width=2)

    sig_bbox = pixel_bbox_to_norm(
        x0=x0, y0=y0, x1=x1, y1=y1, width=image.width, height=image.height
    )
    sig_region = Region(
        id=f"signature-{seed}",
        page_index=idx,
        bbox=sig_bbox,
        kind=RegionKind.SIGNATURE,
    )
    sig_truth = RegionTruth(region=sig_region, text=None)

    return SynthPage(
        kind="signature",
        seed=seed,
        image=image,
        page_index=idx,
        width=image.width,
        height=image.height,
        orientation=source.orientation,
        regions=_rebase(source.regions) + (sig_truth,),
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "signed": "yes",
            "present": True,
            "signature_box": (x0, y0, x1, y1),
        },
    )
