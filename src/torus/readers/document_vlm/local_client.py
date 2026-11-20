"""Local/internal HTTP client for document VLMs (stdlib only).

Mirrors the classify VLM client contract but lives under ``readers/`` so stages
do not import each other. Refuses non-local hosts. Never logs response text.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from PIL import Image

from torus.core.errors import ReaderError

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_ENDPOINT_ENV = "TORUS_LLM_URL"
_DEFAULT_OPENAI_PATH = "/v1/chat/completions"
_DEFAULT_OLLAMA_PATH = "/api/chat"
_PATH_ENV = "TORUS_DOC_VLM_PATH"


@dataclass(frozen=True)
class VlmCompletion:
    """Parsed model reply. ``raw_score`` is only set from real log-probs."""

    text: str
    raw_score: float | None = None


@runtime_checkable
class DocumentVlmClient(Protocol):
    """Minimal chat+image contract for document-VLM readers."""

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        image: Image.Image | None = None,
    ) -> VlmCompletion: ...


def resolve_endpoint_url(*, endpoint_env: str = _DEFAULT_ENDPOINT_ENV) -> str | None:
    """Return the configured local base URL, or ``None`` when unset."""
    value = os.environ.get(endpoint_env, "").strip()
    return value or None


def resolve_api_path() -> str:
    """Path under the base URL; OpenAI-compatible by default (vLLM)."""
    override = os.environ.get(_PATH_ENV, "").strip()
    if override:
        return override if override.startswith("/") else f"/{override}"
    return _DEFAULT_OPENAI_PATH


def assert_local_url(base_url: str) -> None:
    """Raise ``ReaderError`` when the host is not local or internal."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in _LOCAL_HOSTS:
        return
    if host.endswith(".local") or host.startswith("10."):
        return
    if host.startswith("192.168.") or host.startswith("172."):
        return
    raise ReaderError(f"document VLM endpoint host {host!r} is not local/internal")


class LocalDocumentVlmClient:
    """POST chat+image to a local Ollama or OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        path: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._path = path if path is not None else resolve_api_path()
        self._timeout_s = timeout_s
        self._temperature = temperature
        assert_local_url(self._base_url)

    @property
    def model(self) -> str:
        return self._model

    @property
    def path(self) -> str:
        return self._path

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        image: Image.Image | None = None,
    ) -> VlmCompletion:
        if not self._model:
            raise ReaderError("document VLM model id is unset")
        if self._is_ollama_path():
            payload = self._ollama_payload(messages, image=image)
        else:
            payload = self._openai_payload(messages, image=image)

        url = f"{self._base_url}{self._path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ReaderError(f"local document VLM request failed: {exc}") from exc

        return parse_completion(raw, openai_style=not self._is_ollama_path())

    def _is_ollama_path(self) -> bool:
        return self._path.rstrip("/").endswith("/api/chat") or self._path in {
            _DEFAULT_OLLAMA_PATH,
            "/api/generate",
        }

    def _ollama_payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        image: Image.Image | None,
    ) -> dict[str, object]:
        msgs: list[dict[str, object]] = [dict(m) for m in messages]
        if image is not None:
            b64 = image_to_jpeg_b64(image)
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    msg["images"] = [b64]
                    break
        return {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

    def _openai_payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        image: Image.Image | None,
    ) -> dict[str, object]:
        openai_msgs: list[dict[str, object]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if image is not None and role == "user" and msg is messages[-1]:
                b64 = image_to_jpeg_b64(image)
                openai_msgs.append(
                    {
                        "role": role,
                        "content": [
                            {"type": "text", "text": str(content)},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                )
            else:
                openai_msgs.append({"role": role, "content": str(content)})
        # Do not request logprobs: many vision servers reject the flag.
        # If the server includes them anyway, parse_completion will use them.
        return {
            "model": self._model,
            "messages": openai_msgs,
            "temperature": self._temperature,
            "stream": False,
        }


def image_to_jpeg_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_completion(raw: str, *, openai_style: bool = True) -> VlmCompletion:
    """Extract assistant text and optional log-prob score. No invented scores."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReaderError("document VLM response was not JSON") from exc

    text = _extract_text(data)
    if text is None:
        raise ReaderError("document VLM response missing message content")

    score: float | None = None
    if openai_style:
        score = _score_from_openai_logprobs(data)
    return VlmCompletion(text=text, raw_score=score)


def _extract_text(data: dict[str, Any]) -> str | None:
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])

    if isinstance(data.get("response"), str):
        return str(data["response"])
    return None


def _score_from_openai_logprobs(data: dict[str, Any]) -> float | None:
    """Mean token probability from chat ``logprobs`` when the server provides it."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    probs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        lp = item.get("logprob")
        if isinstance(lp, (int, float)):
            probs.append(math.exp(float(lp)))
    if not probs:
        return None
    mean = sum(probs) / len(probs)
    if mean != mean or not 0.0 <= mean <= 1.0:
        return None
    return mean
