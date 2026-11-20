"""Reader smoke for char_cell (P5.5).

Skips when weights are missing under TORUS_MODEL_DIR. Train first:

    python -m torus.readers.char_cell train --source synth --out char_cell/v1.npz
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.doctor.weights import model_dir, weights_present
from torus.readers.char_cell.train_data import synth_glyph
from torus.readers.load import instantiate_reader

pytestmark = pytest.mark.model

_MODEL = "char_cell/v1.npz"


def _weights_ready() -> bool:
    override = os.environ.get("TORUS_CHAR_CELL_WEIGHTS", "").strip()
    if override and Path(override).is_file():
        return True
    return weights_present(_MODEL, root=model_dir())


@pytest.mark.skipif(not _weights_ready(), reason="char_cell weights not installed")
def test_char_cell_loads_and_reads_synthetic_crops() -> None:
    model_path: str | None = (
        os.environ.get("TORUS_CHAR_CELL_WEIGHTS", "").strip() or None
    )
    if not model_path:
        root = model_dir()
        assert root is not None
        model_path = str(root / _MODEL)

    spec = ReaderSpec(
        name="char_cell",
        family="char_cell",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=1,
        device=DeviceKind.CPU,
        model=model_path,
        version="1",
        module="torus.readers.char_cell.reader",
        capabilities=ReaderCaps(
            scripts=["latin"],
            handwriting=True,
            printed=True,
            per_char_confidence=True,
        ),
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)

    printed = synth_glyph("7", seed=1, handwriting=False)
    region = Region(
        id="printed-cell",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    printed_reading = reader.read(
        printed, region, ReadHint(charset=Charset.DIGITS, expected_length=1)
    )
    assert printed_reading.error is None
    assert printed_reading.text != ""
    assert printed_reading.raw_score is not None
    assert printed_reading.char_confidences is not None
    assert len(printed_reading.char_confidences) == 1

    handwritten = synth_glyph("A", seed=2, handwriting=True)
    hw_reading = reader.read(
        handwritten,
        Region(
            id="hw-cell",
            page_index=0,
            bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
            kind=RegionKind.FIELD_VALUE,
        ),
        ReadHint(charset=Charset.LETTERS, expected_length=1),
    )
    assert hw_reading.error is None
    assert hw_reading.text != ""
    assert hw_reading.raw_score is not None


@pytest.mark.skipif(not _weights_ready(), reason="char_cell weights not installed")
def test_char_cell_blank_crop_is_empty() -> None:
    model_path: str | None = (
        os.environ.get("TORUS_CHAR_CELL_WEIGHTS", "").strip() or None
    )
    if not model_path:
        root = model_dir()
        assert root is not None
        model_path = str(root / _MODEL)
    spec = ReaderSpec(
        name="char_cell",
        family="char_cell",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=1,
        device=DeviceKind.CPU,
        model=model_path,
        version="1",
        module="torus.readers.char_cell.reader",
        capabilities=ReaderCaps(handwriting=True, printed=True, scripts=["latin"]),
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    blank = Image.new("RGB", (48, 48), (250, 250, 250))
    ImageDraw.Draw(blank).rectangle((2, 2, 45, 45), outline=(200, 200, 200))
    reading = reader.read(
        blank,
        Region(
            id="empty",
            page_index=0,
            bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
            kind=RegionKind.FIELD_VALUE,
        ),
        ReadHint(),
    )
    assert reading.error is None
    assert reading.text == ""
    assert reading.raw_score is None
