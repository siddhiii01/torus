"""Character-cell reader: one handwritten glyph per comb cell (P5.5)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReaderCaps, ReadHint, Reading, Region
from torus.doctor.weights import model_dir
from torus.readers.char_cell.model import CharCellWeights, load_weights
from torus.readers.char_cell.preprocess import (
    DEFAULT_INK_FRACTION_MIN,
    crop_region,
    feature_vector,
    is_blank,
)


class CharCellReader:
    """Small MLP classifier for a single comb-cell crop.

    Heavy deps are not used at inference (NumPy + Pillow only). Weights live
    under ``TORUS_MODEL_DIR`` (see ``spec.model``). Missing weights raise
    ``ReaderError`` at construct time so the registry soft-skips.
    """

    default_name = "char_cell"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        weights: CharCellWeights | None = None,
        ink_fraction_min: float = DEFAULT_INK_FRACTION_MIN,
    ) -> None:
        self.device = device
        self._spec = spec
        self.name = spec.name if spec is not None else self.default_name
        self._version = (spec.version if spec is not None else None) or "0"
        self._caps = (
            spec.capabilities
            if spec is not None
            else ReaderCaps(
                scripts=["latin"],
                handwriting=True,
                printed=True,
                layout=False,
                per_char_confidence=True,
            )
        )
        self._ink_fraction_min = ink_fraction_min
        if weights is not None:
            self._weights = weights
        else:
            path = _resolve_weights_path(spec.model if spec is not None else None)
            if path is None:
                raise ReaderError(
                    f"reader {self.name!r} needs weights "
                    "(set readers.yaml model= and TORUS_MODEL_DIR)"
                )
            self._weights = load_weights(path)

    def capabilities(self) -> ReaderCaps:
        return self._caps

    @property
    def version(self) -> str:
        return self._version

    @property
    def weights(self) -> CharCellWeights:
        return self._weights

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        started = time.perf_counter()
        try:
            crop = crop_region(image, region)
            features, ink_fraction = feature_vector(
                crop,
                size=self._weights.input_size,
            )
            if is_blank(ink_fraction, min_fraction=self._ink_fraction_min):
                return Reading(
                    region_id=region.id,
                    reader_name=self.name,
                    reader_version=self._version,
                    text="",
                    char_confidences=None,
                    raw_score=None,
                    time_ms=(time.perf_counter() - started) * 1000.0,
                )
            pred = self._weights.predict(features, charset=hint.charset)
            # expected_length > 1 would mean a multi-cell field; this reader
            # still returns one character (callers vote per cell).
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version=self._version,
                text=pred.character,
                char_confidences=[pred.probability],
                raw_score=pred.probability,
                time_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 — soft failure → Reading.error
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version=self._version,
                text="",
                raw_score=None,
                time_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )


def create_reader(*, device: str, spec: ReaderSpec, **kwargs: Any) -> CharCellReader:
    """Factory used by ``torus.readers.load``."""
    return CharCellReader(device=device, spec=spec, **kwargs)


def _resolve_weights_path(model: str | None) -> Path | None:
    if model is None or not model.strip():
        return None
    model = model.strip()
    candidate = Path(model)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    # Prefer TORUS_MODEL_DIR; also allow an explicit env override for tests.
    override = os.environ.get("TORUS_CHAR_CELL_WEIGHTS", "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None
    root = model_dir()
    if root is None:
        return None
    path = root / model
    return path if path.is_file() else None
