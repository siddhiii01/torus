"""PARSeq short-crop reader, through docTR's recognition predictor.

``doctr`` / ``torch`` load on first read only. docTR returns one score per
crop and offers no decoding constraint, so ``ReadHint`` is not applied here
(same as the other docTR-based reader) and ``char_confidences`` stays ``None``.
The score is docTR's own, passed through unchanged.
"""

from __future__ import annotations

from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.models import ReadHint
from torus.readers.classic.common import unit_score
from torus.readers.handwriting.common import (
    Decoded,
    HandwritingReaderBase,
    LineDecoder,
    require_modules,
)

DEFAULT_ARCH = "parseq"


class _ParseqDecoder:
    def __init__(self, arch: str, device: str) -> None:
        from doctr.models import recognition_predictor

        predictor = recognition_predictor(arch, pretrained=True)
        import torch

        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        self._predictor = predictor.cuda() if use_cuda else predictor

    def decode(self, crop: Any, hint: ReadHint) -> Decoded:
        # No constraint hook in docTR's recogniser; see module docstring.
        _ = hint
        import numpy as np

        results = self._predictor([np.asarray(crop)])
        text, score = results[0]
        return Decoded(text=str(text), score=unit_score(score))


class ParseqReader(HandwritingReaderBase):
    """Reader protocol implementation for PARSeq on short handwritten crops."""

    default_name = "parseq_handwritten"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        decoder: LineDecoder | None = None,
    ) -> None:
        super().__init__(device=device, spec=spec, decoder=decoder)
        if decoder is None:
            require_modules("torch", "doctr", extra="doctr")

    def _build_decoder(self) -> LineDecoder:
        return _ParseqDecoder(DEFAULT_ARCH, self.device)


def create_reader(*, device: str, spec: ReaderSpec) -> ParseqReader:
    """Factory used by ``torus.readers.load``."""
    return ParseqReader(device=device, spec=spec)
