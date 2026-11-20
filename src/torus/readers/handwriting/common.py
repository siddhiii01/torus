"""Shared pieces for the handwriting line readers (TrOCR, PARSeq).

Both are attention decoders for a cropped line or field. They are independent
voters next to the VLM and classic families, so each reports only the score its
own model produced. Nothing here derives or calibrates a confidence.
"""

from __future__ import annotations

import math
import string
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReaderCaps, ReadHint, Reading, Region
from torus.core.types import Charset
from torus.doctor.weights import model_dir
from torus.readers.crop_hash import crop_image

# Tokenizer markers for a leading space (BPE and SentencePiece); not characters.
_SPACE_MARKERS = "Ġ▁ "


@dataclass(frozen=True)
class Decoded:
    """One decoder result. ``score`` is the model's own, in [0, 1], else ``None``."""

    text: str
    score: float | None = None


class LineDecoder(Protocol):
    """The model-facing half of a reader; injectable so tests need no weights."""

    def decode(self, crop: Any, hint: ReadHint) -> Decoded: ...


def allowed_characters(charset: Charset) -> frozenset[str] | None:
    """Characters a field of this charset may contain; ``None`` = unconstrained."""
    if charset is Charset.DIGITS:
        return frozenset(string.digits)
    if charset is Charset.LETTERS:
        return frozenset(string.ascii_letters)
    if charset is Charset.ALPHANUMERIC:
        return frozenset(string.ascii_letters + string.digits)
    return None


def token_is_allowed(token_text: str, allowed: frozenset[str]) -> bool:
    """True when every character of a decoded token is in ``allowed``.

    A token that decodes to nothing is refused, so special or empty tokens are
    never allowed by accident; callers whitelist end-of-sequence explicitly.
    """
    body = token_text.strip(_SPACE_MARKERS)
    return bool(body) and all(ch in allowed for ch in body)


def max_chars(hint: ReadHint) -> int | None:
    """Tightest character-count bound the hint gives, if any."""
    bounds = [b for b in (hint.expected_length, hint.max_length) if b is not None]
    return min(bounds) if bounds else None


def sequence_probability(token_probs: Sequence[float]) -> float | None:
    """Probability of the decoded sequence: the product of its token probabilities.

    ``None`` for an empty or invalid input; a partial product would overstate
    the score, and the model gave no other number to fall back on.
    """
    if not token_probs:
        return None
    log_total = 0.0
    for prob in token_probs:
        if prob != prob or not 0.0 < prob <= 1.0:  # NaN, zero or out of range
            return None
        log_total += math.log(prob)
    return math.exp(log_total)


def resolve_model_ref(model: str | None, default: str) -> str:
    """A local weights path when one exists, else the configured/default id.

    Relative paths resolve under ``TORUS_MODEL_DIR``. Callers load with
    ``local_files_only`` so the pipeline never reaches the network.
    """
    ref = (model or "").strip() or default
    candidate = Path(ref)
    if candidate.is_absolute():
        return str(candidate)
    root = model_dir()
    if root is not None and (root / ref).exists():
        return str(root / ref)
    return ref


def require_modules(*modules: str, extra: str) -> None:
    """Raise ``ReaderError`` (so the registry skips the reader) if a stack is absent."""
    import importlib.util

    for module in modules:
        if importlib.util.find_spec(module) is None:
            msg = f"{module} is not installed (pip install -e '.[{extra}]')"
            raise ReaderError(msg)


class HandwritingReaderBase:
    """Identity, caps, crop handling and soft failure; subclasses supply a decoder."""

    default_name = "handwriting"
    default_model = ""

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        decoder: LineDecoder | None = None,
    ) -> None:
        self.device = device
        self._spec = spec
        self.name = spec.name if spec is not None else self.default_name
        self._version = (spec.version if spec is not None else None) or "0"
        self._caps = spec.capabilities if spec is not None else ReaderCaps()
        self._decoder = decoder
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> ReaderCaps:
        return self._caps

    def _build_decoder(self) -> LineDecoder:  # pragma: no cover - abstract
        raise NotImplementedError

    def _get_decoder(self) -> LineDecoder:
        with self._lock:
            if self._decoder is None:
                self._decoder = self._build_decoder()
            return self._decoder

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        started = time.perf_counter()
        text, score, error = "", None, None
        try:
            decoded = self._get_decoder().decode(_as_rgb(image, region), hint)
            text, score = decoded.text.strip(), decoded.score
        except Exception as exc:  # noqa: BLE001 - soft fail per Reader protocol
            # Type and message only: a decoder error must not echo crop content.
            error = f"{type(exc).__name__}: {exc}"
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version=self._version,
            text=text,
            char_confidences=None,
            raw_score=score,
            time_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )


def _as_rgb(image: Any, region: Region) -> Any:
    crop = crop_image(image, region.bbox)
    convert = getattr(crop, "convert", None)
    if not callable(convert):
        msg = "handwriting readers need a PIL-like image"
        raise ReaderError(msg)
    return convert("RGB")
