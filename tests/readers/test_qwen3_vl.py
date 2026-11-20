"""Smoke test for the Qwen3-VL reader against a real local model server.

Skipped unless ``TORUS_LLM_URL`` is set, the server is reachable and
``TORUS_QWEN_VL_MODEL`` names a loaded Qwen3-VL tag. The crop is synthetic.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest
from PIL import Image, ImageDraw, ImageFont

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.load import instantiate_reader

pytestmark = pytest.mark.model


def _server_up() -> bool:
    base = os.environ.get("TORUS_LLM_URL", "").strip()
    if not base or not os.environ.get("TORUS_QWEN_VL_MODEL"):
        return False
    try:
        urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=3).close()
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


@pytest.mark.skipif(
    not _server_up(), reason="no local VLM server / model tag configured"
)
def test_qwen3_vl_reads_synthetic_digit_crop() -> None:
    spec = ReaderSpec(
        name="qwen3_vl_8b",
        family="general_vlm",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, handwriting=True),
        model=None,
        version="smoke",
        module="torus.readers.general_vlm.qwen3_vl_8b",
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    reader._model_id = os.environ["TORUS_QWEN_VL_MODEL"]  # type: ignore[attr-defined]

    image = Image.new("RGB", (360, 90), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text((10, 15), "482913", font=font, fill="black")
    region = Region(
        id="synth-1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    reading = reader.read(image, region, ReadHint(charset=Charset.DIGITS))
    assert reading.error is None
    assert reading.text == "482913"
    assert reading.raw_score is None or 0.0 < reading.raw_score <= 1.0
