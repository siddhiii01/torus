"""Model smoke for document-VLM adapters (needs TORUS_LLM_URL + loaded model)."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.document_vlm.glm_ocr import create_reader
from torus.readers.load import instantiate_reader


def _has_local_server() -> bool:
    return bool(os.environ.get("TORUS_LLM_URL", "").strip())


@pytest.mark.model
@pytest.mark.skipif(not _has_local_server(), reason="TORUS_LLM_URL not set")
def test_glm_ocr_loads_and_reads_synthetic_crops() -> None:
    """Smoke: load plugin and read printed + handwriting-like crops via local server."""
    spec = ReaderSpec(
        name="glm_ocr",
        family="document_vlm",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(
            scripts=["latin"],
            handwriting=True,
            printed=True,
            layout=True,
        ),
        module="torus.readers.document_vlm.glm_ocr",
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    assert reader.name == "glm_ocr"

    printed = Image.new("RGB", (120, 40), (255, 255, 255))
    handwritten = Image.new("RGB", (120, 40), (245, 245, 240))
    region = Region(
        id="synth-doc-vlm",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    for image in (printed, handwritten):
        reading = reader.read(image, region, ReadHint(charset=Charset.ANY))
        assert reading.region_id == "synth-doc-vlm"
        assert reading.reader_name == "glm_ocr"
        # Soft failure is allowed if the server has no matching model tag;
        # the contract is: typed Reading, never invented confidence.
        if reading.error is None:
            assert reading.raw_score is None or 0.0 <= reading.raw_score <= 1.0
        else:
            assert reading.text == ""
            assert reading.raw_score is None


@pytest.mark.model
def test_create_reader_factory_without_server() -> None:
    """Factory must construct even when the local server is absent."""
    spec = ReaderSpec(
        name="glm_ocr",
        family="document_vlm",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, layout=True),
        module="torus.readers.document_vlm.glm_ocr",
    )
    reader = create_reader(device="cpu", spec=spec)
    assert reader.name == "glm_ocr"
