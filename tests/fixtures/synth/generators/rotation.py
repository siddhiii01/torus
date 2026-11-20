"""Rotation and fax/fade-degradation wrappers over other synth pages."""

from __future__ import annotations

import random
from typing import Literal

from tests.fixtures.synth.degrade import fade_degrade, fax_degrade
from tests.fixtures.synth.generators.printed_form import generate_printed_form
from tests.fixtures.synth.geometry import rotate_norm_bbox
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import LabelAnchor, Region

OrientationDegrees = Literal[0, 90, 180, 270]

_VALID_ROTATIONS: tuple[OrientationDegrees, ...] = (0, 90, 180, 270)


def _remap_regions(
    page: SynthPage,
    *,
    degrees: OrientationDegrees,
) -> tuple[RegionTruth, ...]:
    remapped: list[RegionTruth] = []
    for item in page.regions:
        new_bbox, _, _ = rotate_norm_bbox(
            item.region.bbox,
            degrees=degrees,
            src_width=page.width,
            src_height=page.height,
        )
        label_anchor = item.region.label_anchor
        if label_anchor is not None and label_anchor.bbox is not None:
            new_label_bbox, _, _ = rotate_norm_bbox(
                label_anchor.bbox,
                degrees=degrees,
                src_width=page.width,
                src_height=page.height,
            )
            label_anchor = LabelAnchor(text=label_anchor.text, bbox=new_label_bbox)
        region = Region(
            id=item.region.id,
            page_index=item.region.page_index,
            bbox=new_bbox,
            kind=item.region.kind,
            label_anchor=label_anchor,
        )
        remapped.append(
            RegionTruth(region=region, text=item.text, checked=item.checked)
        )
    return tuple(remapped)


def generate_rotated(
    seed: int = 0,
    *,
    degrees: OrientationDegrees | None = None,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Rotate a base synth page clockwise; remap ground-truth bboxes."""
    rng = random.Random(seed)
    if degrees is None:
        degrees = _VALID_ROTATIONS[rng.randint(0, len(_VALID_ROTATIONS) - 1)]
    if degrees not in _VALID_ROTATIONS:
        msg = f"degrees must be one of {_VALID_ROTATIONS}, got {degrees}"
        raise ValueError(msg)

    source = base if base is not None else generate_printed_form(seed)
    # Pillow rotate is counter-clockwise; negate for clockwise page rotation.
    rotated_image = source.image.rotate(
        -degrees, expand=True, fillcolor=(255, 255, 255)
    )
    rotated_image = rotated_image.convert("RGB")
    new_w, new_h = rotated_image.size
    regions = _remap_regions(source, degrees=degrees)
    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="rotated",
        seed=seed,
        image=rotated_image,
        page_index=idx,
        width=new_w,
        height=new_h,
        orientation=degrees,
        regions=regions,
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "degrees": degrees,
        },
    )


def generate_fax_degraded(
    seed: int = 0,
    *,
    strength: float | None = None,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Apply fax-like degradation; ground-truth geometry is unchanged."""
    rng = random.Random(seed)
    if strength is None:
        strength = 0.45 + 0.4 * rng.random()
    source = base if base is not None else generate_printed_form(seed)
    degraded = fax_degrade(source.image, seed=seed, strength=strength)
    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="fax_degraded",
        seed=seed,
        image=degraded,
        page_index=idx,
        width=source.width,
        height=source.height,
        orientation=source.orientation,
        regions=source.regions,
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "strength": strength,
        },
    )


def generate_faded(
    seed: int = 0,
    *,
    strength: float | None = None,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Apply faded / grey-scan degradation; ground-truth geometry is unchanged."""
    rng = random.Random(seed)
    if strength is None:
        strength = 0.5 + 0.35 * rng.random()
    source = base if base is not None else generate_printed_form(seed)
    faded = fade_degrade(source.image, seed=seed, strength=strength)
    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="faded",
        seed=seed,
        image=faded,
        page_index=idx,
        width=source.width,
        height=source.height,
        orientation=source.orientation,
        regions=source.regions,
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "strength": strength,
        },
    )
