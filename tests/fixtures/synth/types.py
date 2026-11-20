"""Ground-truth shapes returned by synthetic page generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image

from torus.core import Region

SynthKind = Literal[
    "printed_form",
    "comb_row",
    "char_cell",
    "checkbox",
    "handwriting",
    "rotated",
    "fax_degraded",
    "faded",
    "stamp",
    "signature",
    "skewed",
    "perspective_photo",
    "bordered",
    "blank",
    "layout_page",
    "label_anchor",
    "table",
]


@dataclass(frozen=True)
class RegionTruth:
    """One located region plus the expected text (or checkbox state)."""

    region: Region
    text: str | None = None
    checked: bool | None = None


@dataclass(frozen=True)
class SynthPage:
    """A generated page image with deterministic ground truth.

    Images are in-memory only. Do not write them into the repo.
    """

    kind: SynthKind
    seed: int
    image: Image.Image
    page_index: int
    width: int
    height: int
    orientation: Literal[0, 90, 180, 270]
    regions: tuple[RegionTruth, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def region_by_id(self, region_id: str) -> RegionTruth:
        for item in self.regions:
            if item.region.id == region_id:
                return item
        msg = f"no region with id {region_id!r}"
        raise KeyError(msg)
