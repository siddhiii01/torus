"""Lightweight label embeddings (char n-grams). No heavy ML dependency."""

from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable

import numpy as np

from torus.layout.textnorm import normalize_label


@runtime_checkable
class LabelEmbedder(Protocol):
    """Maps short label strings to vectors for cosine matching."""

    def embed(self, text: str) -> np.ndarray: ...

    def similarity(self, a: str, b: str) -> float: ...


class CharNgramEmbedder:
    """Hashed character n-gram bag with L2-normalised cosine similarity.

    Stable across processes (CRC32). Good enough for short form labels
    ("Mobile", "Mob No", "Cell Phone") without downloading a model.
    """

    def __init__(self, *, n: int = 3, dim: int = 256) -> None:
        if n < 1:
            msg = f"n must be >= 1, got {n}"
            raise ValueError(msg)
        if dim < 8:
            msg = f"dim must be >= 8, got {dim}"
            raise ValueError(msg)
        self.n = n
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        normed = normalize_label(text)
        vec = np.zeros(self.dim, dtype=np.float64)
        if not normed:
            return vec
        padded = f" {normed} "
        if len(padded) < self.n:
            grams = [padded]
        else:
            grams = [padded[i : i + self.n] for i in range(len(padded) - self.n + 1)]
        for gram in grams:
            idx = zlib.crc32(gram.encode("utf-8")) % self.dim
            vec[idx] += 1.0
        length = float(np.linalg.norm(vec))
        if length > 0.0:
            vec /= length
        return vec

    def similarity(self, a: str, b: str) -> float:
        va = self.embed(a)
        vb = self.embed(b)
        if not np.any(va) or not np.any(vb):
            return 0.0
        return float(np.dot(va, vb))
