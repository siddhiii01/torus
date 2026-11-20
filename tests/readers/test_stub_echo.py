"""Smoke test for the stub echo reader (no model weights)."""

from __future__ import annotations

import pytest
from PIL import Image

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import RegionKind
from torus.readers.load import instantiate_reader


@pytest.mark.model
def test_stub_echo_loads_and_reads_synthetic_crop() -> None:
    """Marked model so bake-off runners include it; stub needs no extras."""
    spec = ReaderSpec(
        name="stub_echo",
        family="stub",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, scripts=["latin"]),
        version="0",
        module="torus.readers.stub.echo",
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    image = Image.new("RGB", (64, 32), (240, 240, 240))
    region = Region(
        id="synth-1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    reading = reader.read(image, region, ReadHint())
    assert reading.region_id == "synth-1"
    assert reading.reader_name == "stub_echo"
    assert reading.error is None
    assert reading.raw_score is None
