"""Synthetic char-cell fixture generator (S8 / P5.5)."""

from __future__ import annotations

from tests.fixtures.synth import generate, generate_char_cell
from torus.core.types import RegionKind


def test_generate_char_cell_deterministic() -> None:
    a = generate_char_cell(seed=17, character="K")
    b = generate_char_cell(seed=17, character="K")
    assert a.kind == "char_cell"
    assert a.meta["character"] == "K"
    assert len(a.regions) == 1
    assert a.regions[0].text == "K"
    assert a.regions[0].region.kind is RegionKind.FIELD_VALUE
    assert a.image.tobytes() == b.image.tobytes()


def test_generate_dispatch_char_cell() -> None:
    page = generate("char_cell", seed=3, character="5")
    assert page.kind == "char_cell"
    assert page.regions[0].text == "5"
    assert page.width == page.image.size[0]
