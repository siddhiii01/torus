"""Unit tests for the Qwen3-VL field-crop reader (P5.4). No model is needed.

The HTTP client is exercised against a throwaway stdlib server on 127.0.0.1.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from PIL import Image

from torus.config.loader import default_config_dir, load_readers
from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.general_vlm.client import (
    Completion,
    LogprobChatClient,
    parse_completion,
)
from torus.readers.general_vlm.prompt import build_messages
from torus.readers.load import instantiate_reader


def _spec(model: str | None = None, module: str = "qwen3_vl_8b") -> ReaderSpec:
    return ReaderSpec(
        name="qwen3_vl_8b",
        family="general_vlm",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, handwriting=True),
        model=model,
        version="t1",
        module=f"torus.readers.general_vlm.{module}",
    )


def _region(x0: float = 0.0, x1: float = 1.0) -> Region:
    return Region(
        id="r1",
        page_index=2,
        bbox=BBox(x0=x0, y0=0.0, x1=x1, y1=1.0),
        kind=RegionKind.FIELD_VALUE,
    )


def _reply(text: str, logprobs: list[Any] | None) -> str:
    choice: dict[str, Any] = {"message": {"role": "assistant", "content": text}}
    if logprobs is not None:
        choice["logprobs"] = {
            "content": [{"token": "t", "logprob": v} for v in logprobs]
        }
    return json.dumps({"choices": [choice]})


# --- prompt -----------------------------------------------------------------


def test_prompt_carries_hint_constraints_only() -> None:
    hint = ReadHint(
        charset=Charset.DIGITS, script="latin", language="en", expected_length=10
    )
    user = build_messages(hint)[-1]["content"]
    for needle in ("digits 0-9 only", "Script: latin", "Language: en", "exactly 10"):
        assert needle in user


def test_prompt_open_hint_has_no_constraint_block() -> None:
    assert "constraints" not in build_messages(ReadHint())[-1]["content"]


def test_prompt_length_bounds_when_no_exact_length() -> None:
    user = build_messages(ReadHint(min_length=3, max_length=8))[-1]["content"]
    assert "Minimum length: 3" in user and "Maximum length: 8" in user


# --- log-prob parsing ---------------------------------------------------------


def test_sequence_prob_is_exp_of_summed_logprobs() -> None:
    result = parse_completion(_reply("482913", [-0.1, -0.2, -0.05]))
    assert result.text == "482913"
    assert result.sequence_prob == pytest.approx(math.exp(-0.35))
    assert result.token_count == 3


@pytest.mark.parametrize(
    "logprobs", [None, [], [-0.1, None], [-0.1, "x"], [True], [0.5]]
)
def test_no_score_without_usable_logprobs(logprobs: list[Any] | None) -> None:
    assert parse_completion(_reply("abc", logprobs)).sequence_prob is None


@pytest.mark.parametrize("raw", ["not json", "{}", '{"choices": []}'])
def test_malformed_reply_is_a_reader_error(raw: str) -> None:
    with pytest.raises(ReaderError):
        parse_completion(raw)


# --- reader with a fake client --------------------------------------------------


class _FakeClient:
    def __init__(self, completion: Completion | Exception) -> None:
        self._completion = completion
        self.calls: list[tuple[Any, Image.Image]] = []

    def complete(self, messages: Any, *, image: Image.Image) -> Completion:
        self.calls.append((messages, image))
        if isinstance(self._completion, Exception):
            raise self._completion
        return self._completion


def _reader(client: _FakeClient, model: str | None = None) -> Any:
    from torus.readers.general_vlm.qwen3_vl_8b import create_reader

    return create_reader(device="cpu", spec=_spec(model), client=client)


def test_reader_crops_region_and_returns_real_score() -> None:
    client = _FakeClient(Completion("  A12  ", sequence_prob=0.8, token_count=2))
    page = Image.new("RGB", (200, 100), "white")
    reading = _reader(client).read(page, _region(0.5, 1.0), ReadHint())
    assert reading.text == "A12"  # whitespace only; no content rewriting
    assert reading.raw_score == 0.8
    assert reading.char_confidences is None
    assert (reading.reader_name, reading.reader_version) == ("qwen3_vl_8b", "t1")
    assert reading.error is None and reading.region_id == "r1"
    assert client.calls[0][1].size == (100, 100)


def test_reader_without_server_score_leaves_score_none() -> None:
    reading = _reader(_FakeClient(Completion("x"))).read(
        Image.new("RGB", (10, 10)), _region(), ReadHint()
    )
    assert reading.text == "x" and reading.raw_score is None


def test_reader_failure_is_a_soft_error() -> None:
    client = _FakeClient(ReaderError("server down"))
    reading = _reader(client).read(Image.new("RGB", (10, 10)), _region(), ReadHint())
    assert reading.text == "" and reading.raw_score is None
    assert reading.error == "server down"


def test_reader_non_image_is_a_soft_error() -> None:
    reading = _reader(_FakeClient(Completion("x"))).read(
        object(), _region(), ReadHint()
    )
    assert reading.error is not None and reading.text == ""


def test_missing_server_url_is_a_soft_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORUS_LLM_URL", raising=False)
    reader = instantiate_reader(_spec(), device=DeviceKind.CPU)
    reading = reader.read(Image.new("RGB", (10, 10)), _region(), ReadHint())
    assert reading.error is not None and "TORUS_LLM_URL" in reading.error


def test_size_modules_bind_their_default_model_tag() -> None:
    for size in ("2b", "4b", "8b"):
        reader = instantiate_reader(
            _spec(module=f"qwen3_vl_{size}"), device=DeviceKind.CPU
        )
        assert reader.model_id == f"qwen3-vl:{size}"  # type: ignore[attr-defined]
    override = instantiate_reader(_spec("custom-tag"), device=DeviceKind.CPU)
    assert override.model_id == "custom-tag"  # type: ignore[attr-defined]


def test_non_local_server_is_refused() -> None:
    with pytest.raises(ReaderError, match="not local"):
        LogprobChatClient("http://example.com:8000", model="m")


# --- real HTTP client against a local stub server --------------------------------


class _Stub(BaseHTTPRequestHandler):
    reject_logprobs = False
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        if type(self).reject_logprobs and body.get("logprobs"):
            self.send_response(400)
            self.end_headers()
            return
        payload = _reply("77", [-0.1, -0.1] if body.get("logprobs") else None)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *args: Any) -> None:  # silence test output
        pass


@pytest.fixture
def stub_url() -> Iterator[str]:
    _Stub.seen = []
    _Stub.reject_logprobs = False
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_client_requests_logprobs_and_sends_image(stub_url: str) -> None:
    client = LogprobChatClient(stub_url, model="m")
    result = client.complete(build_messages(ReadHint()), image=Image.new("RGB", (8, 8)))
    sent = _Stub.seen[0]
    assert sent["logprobs"] is True and sent["temperature"] == 0.0
    parts = sent["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result.text == "77"
    assert result.sequence_prob == pytest.approx(math.exp(-0.2))


def test_client_retries_without_logprobs_when_rejected(stub_url: str) -> None:
    _Stub.reject_logprobs = True
    client = LogprobChatClient(stub_url, model="m")
    result = client.complete(build_messages(ReadHint()), image=Image.new("RGB", (8, 8)))
    assert [("logprobs" in b) for b in _Stub.seen] == [True, False]
    assert result.text == "77" and result.sequence_prob is None


# --- config ----------------------------------------------------------------------


def test_shipped_config_lists_qwen_sizes_disabled() -> None:
    cfg = load_readers(default_config_dir() / "readers.yaml")
    by_name = {r.name: r for r in cfg.readers}
    for size in ("2b", "4b", "8b"):
        spec = by_name[f"qwen3_vl_{size}"]
        assert spec.enabled is False
        assert spec.model is None
        assert spec.module == f"torus.readers.general_vlm.qwen3_vl_{size}"
        assert spec.capabilities.per_char_confidence is False
