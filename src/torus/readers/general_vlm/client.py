"""OpenAI-compatible chat client that asks for token log-probabilities.

Talks only to a local/internal server (``TORUS_LLM_URL``): the host guard is
shared with the document-VLM client. Never logs prompts or replies.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from torus.core.errors import ReaderError
from torus.readers.document_vlm.local_client import (
    assert_local_url,
    image_to_jpeg_b64,
)

DEFAULT_PATH = "/v1/chat/completions"


@dataclass(frozen=True)
class Completion:
    """Assistant text plus the sequence probability when the server gave log-probs."""

    text: str
    sequence_prob: float | None = None
    token_count: int = 0


class LogprobChatClient:
    """POST chat+image; request logprobs, retry once without if the server refuses."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        path: str = DEFAULT_PATH,
        timeout_s: float = 60.0,
        max_tokens: int = 128,
    ) -> None:
        assert_local_url(base_url)
        self._url = base_url.rstrip("/") + path
        self._model = model
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    def complete(
        self, messages: Sequence[dict[str, str]], *, image: Image.Image
    ) -> Completion:
        try:
            raw = self._post(self._payload(messages, image, logprobs=True))
        except urllib.error.HTTPError as exc:
            if exc.code not in (400, 422):
                raise ReaderError(f"local VLM request failed: HTTP {exc.code}") from exc
            # Server rejects the logprobs flag: read without a score (never invented).
            try:
                raw = self._post(self._payload(messages, image, logprobs=False))
            except urllib.error.URLError as exc2:
                raise ReaderError(f"local VLM request failed: {exc2}") from exc2
        except urllib.error.URLError as exc:
            raise ReaderError(f"local VLM request failed: {exc}") from exc
        return parse_completion(raw)

    def _post(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return str(response.read().decode("utf-8"))

    def _payload(
        self, messages: Sequence[dict[str, str]], image: Image.Image, *, logprobs: bool
    ) -> dict[str, Any]:
        body: list[dict[str, Any]] = [dict(m) for m in messages]
        last = body[-1]
        data_url = f"data:image/jpeg;base64,{image_to_jpeg_b64(image)}"
        last["content"] = [
            {"type": "text", "text": last["content"]},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": body,
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if logprobs:
            payload["logprobs"] = True
        return payload


def parse_completion(raw: str) -> Completion:
    """Extract text and ``exp(sum(token logprobs))`` from an OpenAI-style reply.

    The sequence probability is the model's own probability of the whole output
    (a raw score for later calibration). Missing, empty or malformed log-probs
    give ``None``.
    """
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ReaderError("local VLM response missing message content") from exc
    if not isinstance(text, str):
        raise ReaderError("local VLM response content is not text")

    logprobs = choice.get("logprobs")
    items = logprobs.get("content") if isinstance(logprobs, dict) else None
    values: list[float] = []
    if isinstance(items, list):
        for item in items:
            lp = item.get("logprob") if isinstance(item, dict) else None
            if isinstance(lp, (int, float)) and not isinstance(lp, bool):
                values.append(float(lp))
            else:
                return Completion(text=text)  # partial log-probs: no score
    if not values:
        return Completion(text=text)
    prob = math.exp(sum(values))
    if prob != prob or not 0.0 <= prob <= 1.0:
        return Completion(text=text, token_count=len(values))
    return Completion(text=text, sequence_prob=prob, token_count=len(values))
