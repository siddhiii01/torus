"""Qwen3-VL field-crop reader over a local model server (vLLM / Ollama ``/v1``).

``Reading.raw_score`` is the model's sequence probability from the server's
token log-probs, or ``None`` when the server does not return them. No
per-character scores are produced (log-probs are per token).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from PIL import Image

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReaderCaps, ReadHint, Reading, Region
from torus.readers.crop_hash import crop_image
from torus.readers.document_vlm.local_client import resolve_endpoint_url
from torus.readers.general_vlm.client import Completion, LogprobChatClient
from torus.readers.general_vlm.prompt import build_messages

DEFAULT_MODEL = "qwen3-vl:8b"


class ChatClient(Protocol):
    def complete(
        self, messages: Sequence[dict[str, str]], *, image: Image.Image
    ) -> Completion: ...


class Qwen3VlReader:
    """Reader protocol implementation; one model size per registered entry."""

    default_name = "qwen3_vl"
    endpoint_env = "TORUS_LLM_URL"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        client: ChatClient | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.device = device
        self.name = spec.name if spec is not None else self.default_name
        self._version = (spec.version if spec is not None else None) or "0"
        self._caps = spec.capabilities if spec is not None else ReaderCaps()
        self._model_id = model or (spec.model if spec else None) or DEFAULT_MODEL
        self._timeout_s = timeout_s
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model_id

    def capabilities(self) -> ReaderCaps:
        return self._caps

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        started = time.perf_counter()
        text, score, error = "", None, None
        try:
            crop = _as_rgb(crop_image(image, region.bbox))
            completion = self._require_client().complete(
                build_messages(hint), image=crop
            )
            text, score = completion.text.strip(), completion.sequence_prob
        except ReaderError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - soft fail per Reader protocol
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

    def _require_client(self) -> ChatClient:
        if self._client is None:
            base = resolve_endpoint_url(endpoint_env=self.endpoint_env)
            if base is None:
                msg = f"{self.endpoint_env} is unset; Qwen3-VL needs a local server"
                raise ReaderError(msg)
            self._client = LogprobChatClient(
                base, model=self._model_id, timeout_s=self._timeout_s
            )
        return self._client


def _as_rgb(image: Any) -> Image.Image:
    convert = getattr(image, "convert", None)
    if not callable(convert):
        raise ReaderError("Qwen3-VL reader needs a PIL-like image")
    rgb: Image.Image = convert("RGB")
    return rgb


def make_create_reader(
    default_model: str,
) -> Callable[..., Qwen3VlReader]:
    """Factory bound to a server model tag used when ``spec.model`` is null."""

    def create_reader(
        *, device: str, spec: ReaderSpec, client: ChatClient | None = None
    ) -> Qwen3VlReader:
        return Qwen3VlReader(
            device=device,
            spec=spec,
            client=client,
            model=spec.model or default_model,
        )

    return create_reader


create_reader = make_create_reader(DEFAULT_MODEL)
