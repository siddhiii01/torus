"""Unit tests for the character-cell classifier (P5.5) — no model weights on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from torus.config import load_config
from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.doctor.extras import FAMILY_EXTRA
from torus.readers.char_cell.alphabet import (
    DEFAULT_CLASSES,
    DIGITS,
    allowed_indices,
    normalize_label,
)
from torus.readers.char_cell.model import (
    CharCellWeights,
    load_weights,
    save_weights,
    train_mlp,
)
from torus.readers.char_cell.preprocess import (
    feature_vector,
    is_blank,
    to_ink_array,
)
from torus.readers.char_cell.reader import CharCellReader
from torus.readers.char_cell.train_data import load_synth_samples, synth_glyph
from torus.readers.load import instantiate_reader


def test_family_needs_no_pip_extra() -> None:
    assert FAMILY_EXTRA.get("char_cell") is None


def test_readers_yaml_registers_char_cell_disabled() -> None:
    cfg = load_config()
    spec = next(r for r in cfg.readers.readers if r.name == "char_cell")
    assert spec.enabled is False
    assert spec.family == "char_cell"
    assert spec.module == "torus.readers.char_cell.reader"
    assert spec.model == "char_cell/v1.npz"
    assert spec.capabilities.handwriting is True
    assert spec.capabilities.per_char_confidence is True


def test_normalize_and_charset_mask() -> None:
    assert normalize_label("a") == "A"
    assert normalize_label("7") == "7"
    assert normalize_label("ab") is None
    assert allowed_indices(DEFAULT_CLASSES, Charset.DIGITS) == list(range(10))
    letter_idxs = allowed_indices(DEFAULT_CLASSES, Charset.LETTERS)
    assert all(DEFAULT_CLASSES[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for i in letter_idxs)
    assert len(allowed_indices(DEFAULT_CLASSES, Charset.ANY)) == len(DEFAULT_CLASSES)


def test_blank_cell_detection() -> None:
    blank = Image.new("RGB", (40, 40), (255, 255, 255))
    _, frac = feature_vector(blank)
    assert is_blank(frac)
    ink = to_ink_array(synth_glyph("5", seed=1))
    assert float(ink.mean()) > 0.0


def test_train_and_roundtrip_json(tmp_path: Path) -> None:
    classes = DIGITS
    samples = load_synth_samples(classes=classes, per_class=12, seed=3)
    weights = train_mlp(
        samples,
        classes,
        epochs=60,
        learning_rate=0.08,
        batch_size=32,
        seed=3,
    )
    path = tmp_path / "cell.json"
    save_weights(weights, path)
    loaded = load_weights(path)
    assert loaded.classes == classes

    # Hold-out style check on fresh synth glyphs of a few digits.
    hits = 0
    for label in ("0", "1", "2", "7"):
        feats, _ = feature_vector(synth_glyph(label, seed=100 + ord(label)))
        pred = loaded.predict(feats, charset=Charset.DIGITS)
        if pred.character == label:
            hits += 1
        assert 0.0 <= pred.probability <= 1.0
    assert hits >= 2  # small MLP; not every glyph, but better than chance


def test_train_and_roundtrip_npz(tmp_path: Path) -> None:
    classes = ("A", "B", "C")
    samples = load_synth_samples(classes=classes, per_class=10, seed=5)
    weights = train_mlp(samples, classes, epochs=50, seed=5)
    path = tmp_path / "cell.npz"
    save_weights(weights, path)
    loaded = load_weights(path)
    assert loaded.classes == classes
    feats, _ = feature_vector(synth_glyph("B", seed=11, handwriting=False))
    pred = loaded.predict(feats, charset=Charset.LETTERS)
    assert pred.character in classes
    assert abs(sum(pred.probabilities) - 1.0) < 1e-5


def test_charset_constrains_prediction() -> None:
    # Force a letter-preferring model, then ask for digits only.
    classes = ("A", "0")
    rng = np.random.default_rng(0)
    w = CharCellWeights(
        classes=classes,
        input_size=28,
        hidden_size=4,
        w1=rng.normal(size=(4, 784)),
        b1=np.zeros(4),
        w2=np.array([[10.0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64),
        b2=np.array([5.0, -5.0], dtype=np.float64),
    )
    x = np.zeros(784, dtype=np.float64)
    letter = w.predict(x, charset=Charset.LETTERS)
    digit = w.predict(x, charset=Charset.DIGITS)
    assert letter.character == "A"
    assert digit.character == "0"


def test_reader_with_injected_weights_reads_digit() -> None:
    classes = DIGITS
    samples = load_synth_samples(classes=classes, per_class=14, seed=9)
    weights = train_mlp(samples, classes, epochs=70, learning_rate=0.1, seed=9)
    reader = CharCellReader(device="cpu", weights=weights, spec=_spec())
    image = synth_glyph("3", seed=42, handwriting=False)
    region = Region(
        id="c1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    reading = reader.read(image, region, ReadHint(charset=Charset.DIGITS))
    assert reading.error is None
    assert reading.text in DIGITS
    assert reading.raw_score is not None
    assert reading.char_confidences == [reading.raw_score]


def test_reader_blank_returns_empty_without_invented_score() -> None:
    classes = ("0", "1")
    samples = load_synth_samples(classes=classes, per_class=4, seed=1)
    weights = train_mlp(samples, classes, epochs=20, seed=1)
    reader = CharCellReader(device="cpu", weights=weights)
    blank = Image.new("RGB", (48, 48), (255, 255, 255))
    region = Region(
        id="blank",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )
    reading = reader.read(blank, region, ReadHint())
    assert reading.error is None
    assert reading.text == ""
    assert reading.raw_score is None


def test_construct_without_weights_raises() -> None:
    with pytest.raises(ReaderError, match="weights"):
        CharCellReader(device="cpu", spec=_spec())


def test_instantiate_from_module_with_weights(tmp_path: Path) -> None:
    classes = ("X", "Y")
    samples = load_synth_samples(classes=classes, per_class=8, seed=2)
    weights = train_mlp(samples, classes, epochs=40, seed=2)
    path = tmp_path / "v1.npz"
    save_weights(weights, path)
    spec = ReaderSpec(
        name="char_cell",
        family="char_cell",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        model=str(path),
        version="1",
        module="torus.readers.char_cell.reader",
        capabilities=ReaderCaps(handwriting=True, printed=True, scripts=["latin"]),
    )
    reader = instantiate_reader(spec, device=DeviceKind.CPU)
    assert reader.name == "char_cell"
    caps = reader.capabilities()
    assert caps.handwriting is True


def _spec() -> ReaderSpec:
    return ReaderSpec(
        name="char_cell",
        family="char_cell",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=1,
        device=DeviceKind.CPU,
        model="char_cell/v1.npz",
        version="1",
        module="torus.readers.char_cell.reader",
        capabilities=ReaderCaps(
            scripts=["latin"],
            handwriting=True,
            printed=True,
            per_char_confidence=True,
        ),
    )
