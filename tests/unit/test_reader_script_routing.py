"""Unit tests for script detection and routing (P5.7). No models needed."""

from __future__ import annotations

from typing import Any

import pytest

from torus.config.schemas import DeviceKind, ReadersConfig, ReaderSpec
from torus.core.models import BBox, ReaderCaps, ReadHint, Reading, Region
from torus.core.types import RegionKind
from torus.readers.registry import ReaderRegistry
from torus.readers.script import (
    ProbeScriptDetector,
    guess_script,
    hint_for_script,
    route_readers,
    script_of_char,
    select_by_script,
)

# Built from code points so the test source holds no real-language text.
TAMIL = "".join(chr(c) for c in (0x0B85, 0x0B86, 0x0B87, 0x0B95))
DEVANAGARI = "".join(chr(c) for c in (0x0905, 0x0906, 0x0915, 0x0916))
ARABIC = "".join(chr(c) for c in (0x0627, 0x0628, 0x062A, 0x062B))


def _region() -> Region:
    return Region(
        id="r1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.TEXT_LINE,
    )


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("abcd", "latin"),
        (TAMIL, "tamil"),
        (DEVANAGARI, "devanagari"),
        (ARABIC, "arabic"),
    ],
)
def test_guess_script_per_block(text: str, script: str) -> None:
    guess = guess_script(text)
    assert guess.script == script and guess.share == 1.0 and guess.letters == 4


def test_digits_and_punctuation_carry_no_script() -> None:
    assert script_of_char("7") is None and script_of_char("-") is None
    assert guess_script("12 34 - 56").script is None


def test_share_is_measured_letter_fraction() -> None:
    guess = guess_script(TAMIL + "ab")  # 4 tamil, 2 latin
    assert guess.script == "tamil" and guess.share == pytest.approx(4 / 6)


def test_tie_and_too_little_text_are_unknown_not_guessed() -> None:
    assert guess_script("ab" + TAMIL[:2]).script is None  # 2 v 2
    assert guess_script("a").script is None  # below min letters
    assert guess_script("").script is None


class _Probe:
    name = "probe"

    def __init__(self, text: str = "", error: str | None = None) -> None:
        self._text, self._error = text, error

    def capabilities(self) -> ReaderCaps:
        return ReaderCaps()

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        return Reading(
            region_id=region.id,
            reader_name="probe",
            reader_version="0",
            text=self._text,
            error=self._error,
        )


def test_probe_detector_uses_reader_text() -> None:
    guess = ProbeScriptDetector(_Probe(TAMIL)).detect(None, _region())
    assert guess.script == "tamil"


@pytest.mark.parametrize("probe", [_Probe(""), _Probe(TAMIL, error="boom")])
def test_failed_or_empty_probe_is_unknown(probe: _Probe) -> None:
    assert ProbeScriptDetector(probe).detect(None, _region()).script is None


def test_select_by_script_filters_and_keeps_order() -> None:
    cands = [
        ("a", ReaderCaps(scripts=["Latin", "tamil"])),
        ("b", ReaderCaps(scripts=["latin"])),
        ("c", ReaderCaps(scripts=["tamil"])),
    ]
    route = select_by_script(cands, "TAMIL")
    assert route.script == "tamil" and route.readers == ("a", "c")


def test_unknown_script_does_not_filter_and_unsupported_is_empty() -> None:
    cands = [("a", ReaderCaps(scripts=["latin"])), ("b", ReaderCaps(scripts=[]))]
    assert select_by_script(cands, None).readers == ("a", "b")
    empty = select_by_script(cands, "arabic")
    assert empty.readers == () and "arabic" in empty.reason


def test_hint_for_script_respects_explicit_field_script() -> None:
    assert hint_for_script(ReadHint(), "Tamil").script == "tamil"
    assert hint_for_script(ReadHint(script="latin"), "tamil").script == "latin"
    base = ReadHint()
    assert hint_for_script(base, None) is base


def _spec(name: str, scripts: list[str], *, enabled: bool = True) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family="stub",
        enabled=enabled,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(scripts=scripts),
        version="0",
        module="torus.readers.stub.echo",
    )


def test_route_readers_skips_disabled_and_loads_nothing() -> None:
    config = ReadersConfig(
        readers=[
            _spec("on_tamil", ["latin", "tamil"]),
            _spec("on_latin", ["latin"]),
            _spec("off_tamil", ["tamil"], enabled=False),
        ]
    )
    registry = ReaderRegistry(config, require_ready=False)
    route = route_readers(registry, "tamil")
    assert route.readers == ("on_tamil",)
    assert all(registry.get_entry(n)._instance is None for n in registry.names())  # type: ignore[union-attr]
