"""Component checks: synth fixtures produce usable page-shaped outputs (P0.7).

Later preprocess/layout stages consume these generators; this test locks the
contract that a rotated / degraded / stamped page still exposes valid regions.
"""

from __future__ import annotations

import pytest
from tests.fixtures.synth import (
    generate_checkbox,
    generate_comb_row,
    generate_fax_degraded,
    generate_handwriting,
    generate_printed_form,
    generate_rotated,
    generate_stamp,
)

from torus.core import Page, RegionKind


def _as_page(synth: object) -> Page:
    page = Page(
        index=synth.page_index,  # type: ignore[attr-defined]
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
        rotated=synth.orientation != 0,  # type: ignore[attr-defined]
        low_quality=synth.kind == "fax_degraded",  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


@pytest.mark.parametrize(
    "factory,seed",
    [
        (generate_printed_form, 1),
        (generate_comb_row, 2),
        (generate_checkbox, 3),
        (generate_handwriting, 4),
        (generate_stamp, 5),
        (generate_fax_degraded, 6),
    ],
)
def test_synth_binds_to_core_page(factory: object, seed: int) -> None:
    synth = factory(seed=seed)  # type: ignore[operator]
    page = _as_page(synth)
    assert page.has_image is True
    assert page.image.size == (synth.width, synth.height)  # type: ignore[attr-defined]
    assert page.width == synth.width  # type: ignore[attr-defined]
    assert page.height == synth.height  # type: ignore[attr-defined]
    # Every value-bearing region stays inside the unit square.
    for item in synth.regions:  # type: ignore[attr-defined]
        b = item.region.bbox
        assert 0.0 <= b.x0 < b.x1 <= 1.0
        assert 0.0 <= b.y0 < b.y1 <= 1.0


def test_rotated_synth_flags_and_kinds() -> None:
    base = generate_comb_row(seed=10, cells=4, text="AB12")
    synth = generate_rotated(seed=10, degrees=90, base=base)
    page = _as_page(synth)
    assert page.rotated is True
    assert page.orientation == 90
    assert synth.regions[0].region.kind == RegionKind.COMB
    assert synth.regions[0].text == "AB12"


def test_fax_over_stamp_pipeline_chain() -> None:
    """Composition: stamp a form, then fax-degrade — still valid regions."""
    stamped = generate_stamp(seed=12)
    degraded = generate_fax_degraded(seed=12, strength=0.55, base=stamped)
    page = _as_page(degraded)
    assert page.low_quality is True
    kinds = {r.region.kind for r in degraded.regions}
    assert RegionKind.STAMP in kinds
    assert RegionKind.FIELD_VALUE in kinds
