"""Training samples for the character-cell model.

Public sources only: synthetic glyph crops (always available) and optional
EMNIST via the ``datasets`` package (``pip install -e ".[train]"``).
Real customer crops are never used here (Phase 10 corrections land later).
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from torus.core.errors import ReaderError
from torus.readers.char_cell.alphabet import (
    DEFAULT_CLASSES,
    DIGITS,
    LETTERS,
    normalize_label,
)
from torus.readers.char_cell.preprocess import DEFAULT_INPUT_SIZE, feature_vector

VecF32 = NDArray[np.float32]
Sample = tuple[VecF32, str]


def synth_glyph(
    character: str,
    *,
    seed: int = 0,
    size: int = 48,
    handwriting: bool = True,
) -> Image.Image:
    """Draw one character on a white cell (public fonts / Pillow default)."""
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_size = int(size * (0.55 + rng.uniform(0, 0.2)))
    font = _load_font(font_size, handwriting=handwriting)
    # Centre with jitter so the classifier sees position noise.
    bbox = draw.textbbox((0, 0), character, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) / 2 + rng.uniform(-2.5, 2.5) - bbox[0]
    y = (size - th) / 2 + rng.uniform(-2.5, 2.5) - bbox[1]
    shade = int(rng.randint(0, 40))
    draw.text((x, y), character, fill=(shade, shade, shade), font=font)
    return image


def iter_synth_samples(
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    per_class: int = 8,
    seed: int = 0,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> Iterator[Sample]:
    """Yield (feature_vector, label) from synthetic glyphs."""
    for i, label in enumerate(classes):
        for j in range(per_class):
            sample_seed = seed + i * 10_000 + j
            handwriting = (sample_seed % 2) == 0
            image = synth_glyph(
                label,
                seed=sample_seed,
                handwriting=handwriting,
            )
            features, _ = feature_vector(image, size=input_size)
            yield features, label


def load_synth_samples(
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    per_class: int = 8,
    seed: int = 0,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> list[Sample]:
    """Materialise synthetic training samples."""
    return list(
        iter_synth_samples(
            classes=classes,
            per_class=per_class,
            seed=seed,
            input_size=input_size,
        )
    )


def load_emnist_samples(
    *,
    split: str = "train",
    max_per_class: int | None = 200,
    seed: int = 0,
    input_size: int = DEFAULT_INPUT_SIZE,
    cache_dir: Path | None = None,
) -> list[Sample]:
    """Load EMNIST ByClass samples via HuggingFace ``datasets`` (train extra).

    Digits and letters only; lowercase EMNIST labels map to uppercase classes.
    Cache stays under ``TORUS_MODEL_DIR`` / ``TORUS_DATA_DIR`` when set — never
    inside the git checkout.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ReaderError(
            "EMNIST training needs the train extra "
            '(pip install -e ".[train]") so datasets is importable'
        ) from exc

    cache = cache_dir
    if cache is None:
        cache = _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    # Lazy import stays inside this function (heavy / optional).
    ds = load_dataset(
        "emnist",
        "byclass",
        split=split,
        cache_dir=str(cache),
    )

    rng = random.Random(seed)
    # EMNIST ByClass mapping: 0-9 digits, 10-35 A-Z, 36-61 a-z.
    label_to_char = {i: str(i) for i in range(10)}
    for i, ch in enumerate(LETTERS):
        label_to_char[10 + i] = ch
        label_to_char[36 + i] = ch  # lowercase → same class

    counts: dict[str, int] = {c: 0 for c in DEFAULT_CLASSES}
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    samples: list[Sample] = []

    for idx in indices:
        if max_per_class is not None and all(
            counts[c] >= max_per_class for c in DEFAULT_CLASSES
        ):
            break
        row: dict[str, Any] = ds[int(idx)]
        label_i = int(row["label"])
        char = label_to_char.get(label_i)
        if char is None:
            continue
        if max_per_class is not None and counts[char] >= max_per_class:
            continue
        # EMNIST images are transposed relative to MNIST convention.
        arr = np.asarray(row["image"], dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.transpose(arr)
        image = Image.fromarray(arr, mode="L").convert("RGB")
        features, ink_fraction = feature_vector(image, size=input_size)
        if ink_fraction < 0.005:
            continue
        samples.append((features, char))
        counts[char] += 1

    if not samples:
        raise ReaderError("EMNIST load produced no usable samples")
    return samples


def samples_from_image_label_pairs(
    pairs: Sequence[tuple[Image.Image, str]],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> list[Sample]:
    """Convert PIL crops + labels into feature samples (tests / custom sets)."""
    out: list[Sample] = []
    for image, raw in pairs:
        label = normalize_label(raw)
        if label is None:
            raise ReaderError(f"unsupported char-cell label {raw!r}")
        features, _ = feature_vector(image, size=input_size)
        out.append((features, label))
    return out


def _default_cache_dir() -> Path:
    for env in ("TORUS_MODEL_DIR", "TORUS_DATA_DIR", "HF_HOME"):
        raw = os.environ.get(env, "").strip()
        if raw:
            return Path(raw).expanduser().resolve() / "cache" / "emnist"
    # Last resort: user cache, never the repo.
    return Path.home() / ".cache" / "torus" / "emnist"


def _load_font(
    size: int, *, handwriting: bool
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Prefer system / optional synth fonts; fall back to Pillow default."""
    # Importing synth fonts would pull tests into production; use system paths.
    candidates: list[Path] = []
    if handwriting:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\comic.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"),
            ]
        )
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


# Re-export class groups for the CLI help text.
__all__ = [
    "DIGITS",
    "LETTERS",
    "DEFAULT_CLASSES",
    "Sample",
    "iter_synth_samples",
    "load_emnist_samples",
    "load_synth_samples",
    "samples_from_image_label_pairs",
    "synth_glyph",
]
