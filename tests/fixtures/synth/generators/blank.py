"""Blank and near-blank synthetic page generators (S8 / P2.5)."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from tests.fixtures.synth.types import SynthPage


def generate_blank(
    seed: int = 0,
    *,
    width: int = 800,
    height: int = 1100,
    page_index: int = 0,
    near_blank: bool = False,
    speck_count: int | None = None,
) -> SynthPage:
    """Draw an empty page, optionally with a few seeded speckles (near-blank).

    Ground truth has no regions. Speckle positions depend only on ``seed``.
    """
    rng = random.Random(seed)
    # Slight off-white paper tone varies with seed so digests differ.
    paper = 250 - (seed % 6)
    image = Image.new("RGB", (width, height), color=(paper, paper, paper))

    if near_blank:
        if speck_count is None:
            speck_count = 2 + (seed % 3)
        if speck_count < 1:
            msg = f"speck_count must be >= 1 when near_blank, got {speck_count}"
            raise ValueError(msg)
        arr = np.asarray(image).copy()
        for _ in range(speck_count):
            x = rng.randint(8, max(9, width - 9))
            y = rng.randint(8, max(9, height - 9))
            # Tiny dark speck — below glyph size for the blank detector.
            arr[y : y + 2, x : x + 2, :] = 40
        image = Image.fromarray(arr, mode="RGB")

    return SynthPage(
        kind="blank",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=(),
        meta={
            "near_blank": near_blank,
            "speck_count": speck_count if near_blank else 0,
            "paper": paper,
        },
    )


def generate_near_blank(
    seed: int = 0,
    *,
    width: int = 800,
    height: int = 1100,
    page_index: int = 0,
    speck_count: int | None = None,
) -> SynthPage:
    """Convenience wrapper: blank page with a few speckles."""
    return generate_blank(
        seed,
        width=width,
        height=height,
        page_index=page_index,
        near_blank=True,
        speck_count=speck_count,
    )
