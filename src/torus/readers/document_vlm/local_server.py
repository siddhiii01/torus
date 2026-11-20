"""Solid document-VLM path: local Ollama / vLLM via ``TORUS_LLM_URL``."""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReaderCaps, ReadHint, Reading, Region
from torus.readers.crop_hash import crop_image
from torus.readers.document_vlm.common import (
    LayoutBlock,
    build_ocr_messages,
    normalise_ocr_text,
    parse_layout_blocks,
)
from torus.readers.document_vlm.local_client import (
    DocumentVlmClient,
    LocalDocumentVlmClient,
    resolve_endpoint_url,
)

# Default server model tag when ``ReaderSpec.model`` is null (weights check passes).
DEFAULT_MODEL = "document-vlm"


class LocalServerDocumentVlmReader:
    """Reader protocol implementation for OCR-specialised VLMs on a local server.

    ``raw_score`` is set only when the server returns token log-probs. Otherwise
    it stays ``None`` — never invented.
    """

    default_name = "document_vlm_local"
    default_model = DEFAULT_MODEL
    endpoint_env = "TORUS_LLM_URL"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        client: DocumentVlmClient | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.device = device
        self._spec = spec
        if spec is not None:
            self.name = spec.name
            self._version = spec.version or "0"
            self._caps = spec.capabilities
            model_id = model or spec.model or self.default_model
        else:
            self.name = self.default_name
            self._version = "0"
            self._caps = ReaderCaps(
                scripts=["latin"],
                handwriting=True,
                printed=True,
                layout=True,
                per_char_confidence=False,
            )
            model_id = model or self.default_model

        self._model_id = model_id
        self._timeout_s = timeout_s
        self._client = client
        self._last_layout: list[LayoutBlock] | None = None

    @property
    def version(self) -> str:
        return self._version

    @property
    def model_id(self) -> str:
        return self._model_id

    def capabilities(self) -> ReaderCaps:
        return self._caps

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        started = time.perf_counter()
        self._last_layout = None
        try:
            client = self._require_client()
            crop = self._crop_or_page(image, region)
            page_level = _is_page_level(region)
            messages = build_ocr_messages(hint, page_level=page_level)
            completion = client.complete(messages, image=crop)
            text = normalise_ocr_text(completion.text)
            self._last_layout = parse_layout_blocks(completion.text)
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version=self._version,
                text=text,
                char_confidences=None,
                raw_score=completion.raw_score,
                time_ms=(time.perf_counter() - started) * 1000.0,
            )
        except ReaderError as exc:
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version=self._version,
                text="",
                raw_score=None,
                char_confidences=None,
                time_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — soft fail per Reader protocol
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version=self._version,
                text="",
                raw_score=None,
                char_confidences=None,
                time_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    def read_layout(
        self, image: Any, region: Region, hint: ReadHint
    ) -> list[LayoutBlock]:
        """Return layout blocks from the last ``read``, or run a fresh read."""
        if self._last_layout is not None:
            return list(self._last_layout)
        reading = self.read(image, region, hint)
        if reading.error:
            return []
        return list(self._last_layout or [])

    def _require_client(self) -> DocumentVlmClient:
        if self._client is not None:
            return self._client
        base = resolve_endpoint_url(endpoint_env=self.endpoint_env)
        if base is None:
            raise ReaderError(
                f"{self.endpoint_env} is unset; document VLM needs a local server"
            )
        self._client = LocalDocumentVlmClient(
            base,
            model=self._model_id,
            timeout_s=self._timeout_s,
            temperature=0.0,
        )
        return self._client

    def _crop_or_page(self, image: Any, region: Region) -> Image.Image:
        if _is_page_level(region):
            return _as_rgb(image)
        cropped = crop_image(image, region.bbox)
        return _as_rgb(cropped)


def _is_page_level(region: Region) -> bool:
    """Full-page regions keep the whole image (layout / page OCR)."""
    box = region.bbox
    return box.x0 <= 0.01 and box.y0 <= 0.01 and box.x1 >= 0.99 and box.y1 >= 0.99


def _as_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    convert = getattr(image, "convert", None)
    if callable(convert):
        return convert("RGB")  # type: ignore[no-any-return]
    raise ReaderError("document VLM readers need a PIL-like image")


def create_reader(
    *,
    device: str,
    spec: ReaderSpec,
    client: DocumentVlmClient | None = None,
) -> LocalServerDocumentVlmReader:
    """Factory used by ``torus.readers.load``."""
    return LocalServerDocumentVlmReader(device=device, spec=spec, client=client)
