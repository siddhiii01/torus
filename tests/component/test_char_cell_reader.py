"""Component: train on synth cell crops, read held-out generate_char_cell pages."""

from __future__ import annotations

from tests.fixtures.synth import generate_char_cell

from torus.core.models import ReadHint
from torus.core.types import Charset
from torus.readers.char_cell.alphabet import DIGITS, LETTERS
from torus.readers.char_cell.model import train_mlp
from torus.readers.char_cell.preprocess import crop_region, feature_vector
from torus.readers.char_cell.reader import CharCellReader


def _samples_from_generator(
    classes: tuple[str, ...],
    *,
    per_class: int,
    seed_base: int,
    handwriting: bool,
) -> list[tuple]:
    samples: list[tuple] = []
    for i, ch in enumerate(classes):
        for j in range(per_class):
            page = generate_char_cell(
                seed=seed_base + i * 100 + j,
                character=ch,
                handwriting=handwriting,
            )
            crop = crop_region(page.image, page.regions[0].region)
            features, _ = feature_vector(crop)
            samples.append((features, ch))
    return samples


def test_reader_on_synth_digit_cells() -> None:
    train = _samples_from_generator(
        DIGITS, per_class=10, seed_base=1000, handwriting=False
    )
    weights = train_mlp(
        train,
        DIGITS,
        epochs=100,
        learning_rate=0.12,
        batch_size=32,
        seed=21,
    )
    reader = CharCellReader(device="cpu", weights=weights)

    hits = 0
    for i, ch in enumerate(DIGITS):
        # Held-out seeds (not used in training above).
        page = generate_char_cell(
            seed=9000 + i,
            character=ch,
            handwriting=False,
        )
        reading = reader.read(
            page.image,
            page.regions[0].region,
            ReadHint(charset=Charset.DIGITS, expected_length=1),
        )
        assert reading.error is None
        assert reading.raw_score is not None
        if reading.text == ch:
            hits += 1
    assert hits >= 7


def test_reader_on_synth_letter_cells() -> None:
    subset = LETTERS[:8]
    train = _samples_from_generator(
        subset, per_class=10, seed_base=2000, handwriting=True
    )
    weights = train_mlp(
        train,
        subset,
        epochs=100,
        learning_rate=0.12,
        batch_size=32,
        seed=33,
    )
    reader = CharCellReader(device="cpu", weights=weights)

    hits = 0
    for i, ch in enumerate(subset):
        page = generate_char_cell(seed=9100 + i, character=ch, handwriting=True)
        reading = reader.read(
            page.image,
            page.regions[0].region,
            ReadHint(charset=Charset.LETTERS),
        )
        assert reading.error is None
        if reading.text == ch:
            hits += 1
    assert hits >= 5
