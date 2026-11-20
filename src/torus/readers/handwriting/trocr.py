"""TrOCR handwriting line reader.

``torch`` / ``transformers`` load on first read only. Decoding is constrained
by the field's ``ReadHint`` (charset mask and a length cap). ``raw_score`` is
the sequence probability from the decoder's own token scores; it is a raw model
score, not a calibrated confidence. Weights load with ``local_files_only``.
"""

from __future__ import annotations

from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReadHint
from torus.readers.handwriting.common import (
    Decoded,
    HandwritingReaderBase,
    LineDecoder,
    allowed_characters,
    max_chars,
    require_modules,
    resolve_model_ref,
    sequence_probability,
    token_is_allowed,
)

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
# Cap when the hint gives no length; a handwritten field crop is short.
FALLBACK_MAX_TOKENS = 64


class _TrocrDecoder:
    """Loads processor + model once; decodes one crop per call."""

    def __init__(self, model_ref: str, device: str) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        try:
            self._processor = TrOCRProcessor.from_pretrained(
                model_ref, local_files_only=True
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                model_ref, local_files_only=True
            )
        except OSError as exc:
            msg = (
                f"TrOCR weights not found under TORUS_MODEL_DIR ({type(exc).__name__})"
            )
            raise ReaderError(msg) from exc

        import torch

        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        self._device = "cuda" if use_cuda else "cpu"
        self._model = model.eval().to(self._device)
        self._masks: dict[frozenset[str], Any] = {}

    def _mask(self, allowed: frozenset[str]) -> Any:
        """Additive logits mask: 0 for allowed tokens and EOS, -inf elsewhere."""
        cached = self._masks.get(allowed)
        if cached is not None:
            return cached
        import torch

        tokenizer = self._processor.tokenizer
        size = int(self._model.config.decoder.vocab_size)
        mask = torch.full((size,), float("-inf"))
        for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(range(size))):
            if token is not None and token_is_allowed(token, allowed):
                mask[token_id] = 0.0
        eos = tokenizer.eos_token_id
        if eos is not None:
            mask[int(eos)] = 0.0
        mask = mask.to(self._device)
        self._masks[allowed] = mask
        return mask

    def decode(self, crop: Any, hint: ReadHint) -> Decoded:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        allowed = allowed_characters(hint.charset)
        processors = LogitsProcessorList()
        if allowed is not None:
            mask = self._mask(allowed)

            # transformers is untyped here (ignore_missing_imports), so Any base.
            class _CharsetMask(LogitsProcessor):  # type: ignore[misc]
                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    return scores + mask[: scores.shape[-1]]

            processors.append(_CharsetMask())

        pixel_values = self._processor(images=crop, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = self._model.generate(
                pixel_values.to(self._device),
                max_new_tokens=max_chars(hint) or FALLBACK_MAX_TOKENS,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=processors,
            )
            transition = self._model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
        text = self._processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        probs = [float(p) for p in transition[0].exp().tolist()]
        return Decoded(text=text, score=sequence_probability(probs))


class TrocrReader(HandwritingReaderBase):
    """Reader protocol implementation for TrOCR handwritten line/field crops."""

    default_name = "trocr_handwritten"
    default_model = DEFAULT_MODEL

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        decoder: LineDecoder | None = None,
    ) -> None:
        super().__init__(device=device, spec=spec, decoder=decoder)
        if decoder is None:
            require_modules("torch", "transformers", extra="vlm")

    def _build_decoder(self) -> LineDecoder:
        ref = resolve_model_ref(self._spec.model if self._spec else None, DEFAULT_MODEL)
        return _TrocrDecoder(ref, self.device)


def create_reader(*, device: str, spec: ReaderSpec) -> TrocrReader:
    """Factory used by ``torus.readers.load``."""
    return TrocrReader(device=device, spec=spec)
