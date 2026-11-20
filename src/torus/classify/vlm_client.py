"""Local VLM HTTP client (stdlib only). Never call non-local hosts."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from PIL import Image

from torus.config.schemas import ClassifyVlmConfig
from torus.core.errors import ClassifyError

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@runtime_checkable
class VlmClient(Protocol):
    """Minimal chat+image contract for the zero-shot classifier."""

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        image: Image.Image | None = None,
    ) -> str: ...


class LocalHttpVlmClient:
    """POST chat completions to a local Ollama-style ``/api/chat`` endpoint."""

    def __init__(
        self,
        base_url: str,
        config: ClassifyVlmConfig,
        *,
        model: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._config = config
        self._model = model or config.model
        _assert_local_url(self._base_url)

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        image: Image.Image | None = None,
    ) -> str:
        if not self._model:
            raise ClassifyError(
                "classify.vlm.model is unset; set it in config or local.yaml"
            )
        payload: dict[str, object] = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self._config.temperature},
        }
        if image is not None:
            # Ollama accepts base64 images on the last user message.
            b64 = _image_to_jpeg_b64(image)
            msgs: list[dict[str, object]] = [dict(m) for m in messages]
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    msg["images"] = [b64]
                    break
            payload["messages"] = msgs

        url = f"{self._base_url}{self._config.path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._config.timeout_s
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ClassifyError(f"local VLM request failed: {exc}") from exc

        return _parse_chat_content(raw)


def resolve_vlm_base_url(config: ClassifyVlmConfig) -> str | None:
    """Read the local endpoint from the configured environment variable."""
    value = os.environ.get(config.endpoint_env, "").strip()
    return value or None


def build_default_vlm_client(config: ClassifyVlmConfig) -> VlmClient | None:
    """Construct a client when endpoint + model are configured; else None."""
    base = resolve_vlm_base_url(config)
    if base is None or not config.model:
        return None
    return LocalHttpVlmClient(base, config, model=config.model)


def _assert_local_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        # Allow private network hostnames only when clearly loopback-like.
        # Internal LAN hosts are permitted by AGENTS.md; reject clearly public.
        if host.endswith(".local") or host.startswith("10."):
            return
        if host.startswith("192.168.") or host.startswith("172."):
            return
        raise ClassifyError(
            f"VLM endpoint host {host!r} is not local/internal; refused"
        )


def _image_to_jpeg_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_chat_content(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassifyError("VLM response was not JSON") from exc
    # Ollama /api/chat
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    # OpenAI-compatible
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
    # Plain "response" field (Ollama generate)
    if isinstance(data.get("response"), str):
        return str(data["response"])
    raise ClassifyError("VLM response missing message content")
