"""Local LLM HTTP client for the field mapper (stdlib only, local hosts only)."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from torus.config.schemas import MappingConfig
from torus.core.errors import MappingError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@runtime_checkable
class JsonLlmClient(Protocol):
    """Chat completion constrained to a JSON schema; returns the raw JSON text."""

    @property
    def model_id(self) -> str: ...

    def complete_json(
        self,
        messages: Sequence[dict[str, str]],
        schema: Mapping[str, object],
    ) -> str: ...


def assert_local_url(base_url: str) -> None:
    """Refuse anything that is not loopback, a private IP or an internal name."""
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        raise MappingError("LLM endpoint has no host; refused")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost" or host.endswith((".local", ".internal", ".lan")):
            return
        raise MappingError("LLM endpoint host is not local/internal; refused") from None
    if not (ip.is_loopback or ip.is_private):
        raise MappingError("LLM endpoint host is not local/internal; refused")


class LocalHttpLlmClient:
    """POST to a local Ollama ``/api/chat`` or OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, config: MappingConfig) -> None:
        if not config.model:
            raise MappingError("mapping.model is unset; set it in config")
        assert_local_url(base_url)
        self._url = base_url.rstrip("/") + config.path
        self._config = config
        self._model = config.model

    @property
    def model_id(self) -> str:
        return self._model

    def complete_json(
        self,
        messages: Sequence[dict[str, str]],
        schema: Mapping[str, object],
    ) -> str:
        cfg = self._config
        payload: dict[str, object]
        if cfg.path.endswith("/api/chat"):
            payload = {
                "model": self._model,
                "messages": list(messages),
                "stream": False,
                "think": False,
                "format": dict(schema),
                "options": {"temperature": cfg.temperature, "seed": cfg.seed},
            }
        else:
            payload = {
                "model": self._model,
                "messages": list(messages),
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "field_map",
                        "schema": dict(schema),
                        "strict": True,
                    },
                },
            }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise MappingError(
                f"local LLM request failed: {type(exc).__name__}"
            ) from exc
        return clean_json_text(parse_chat_content(raw))


def build_default_llm_client(config: MappingConfig) -> JsonLlmClient | None:
    """Construct a client when the mapper is enabled and fully configured."""
    base = os.environ.get(config.endpoint_env, "").strip()
    if not config.enabled or not base or not config.model:
        return None
    return LocalHttpLlmClient(base, config)


def parse_chat_content(raw: str) -> str:
    """Extract assistant text from an Ollama or OpenAI-style response body."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MappingError("LLM response was not JSON") from exc
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return str(message["content"])
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
    raise MappingError("LLM response had no message content")


def clean_json_text(text: str) -> str:
    """Drop Qwen3 think blocks and markdown fences around the JSON body."""
    text = _THINK_BLOCK.sub("", text).strip()
    fenced = _FENCE.match(text)
    return fenced.group(1) if fenced else text
