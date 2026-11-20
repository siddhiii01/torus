"""Unit tests for document-VLM helpers and local-server reader (no real models)."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from PIL import Image

from torus.config.schemas import DeviceKind, ReaderCaps, ReadersConfig, ReaderSpec
from torus.core.models import BBox, ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.doctor.extras import FAMILY_EXTRA, ExtraStatus, family_extra_installed
from torus.readers.document_vlm.common import (
    build_ocr_messages,
    normalise_ocr_text,
    parse_layout_blocks,
)
from torus.readers.document_vlm.local_client import (
    LocalDocumentVlmClient,
    VlmCompletion,
    assert_local_url,
    parse_completion,
)
from torus.readers.document_vlm.local_server import LocalServerDocumentVlmReader
from torus.readers.document_vlm.olmocr import DEFAULT_MODEL as OLMOCR_MODEL
from torus.readers.document_vlm.olmocr import create_reader as create_olmocr
from torus.readers.load import instantiate_reader
from torus.readers.registry import ReaderRegistry


def _region(
    *,
    rid: str = "r1",
    bbox: BBox | None = None,
    kind: RegionKind = RegionKind.FIELD_VALUE,
) -> Region:
    return Region(
        id=rid,
        page_index=0,
        bbox=bbox or BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.3),
        kind=kind,
    )


def test_document_vlm_family_needs_no_extra() -> None:
    assert FAMILY_EXTRA.get("document_vlm") is None
    ok, reason = family_extra_installed(
        "document_vlm",
        {"vlm": ExtraStatus(name="vlm", installed=False, missing_modules=("torch",))},
    )
    assert ok is True
    assert reason is None


def test_hint_prompt_includes_charset_and_length() -> None:
    msgs = build_ocr_messages(
        ReadHint(charset=Charset.DIGITS, expected_length=6, script="latin")
    )
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "digits" in user.lower()
    assert "6" in user
    assert "latin" in user.lower()


def test_normalise_strips_fence_and_json_text() -> None:
    assert normalise_ocr_text("```text\nABC\n```") == "ABC"
    assert normalise_ocr_text('{"text": "PIN-12"}') == "PIN-12"


def test_parse_layout_blocks_normalised_boxes() -> None:
    raw = json.dumps(
        {
            "blocks": [
                {"text": "A", "bbox": [0.0, 0.0, 0.2, 0.1]},
                {"text": "B", "x0": 0.2, "y0": 0.0, "x1": 0.4, "y1": 0.1},
            ]
        }
    )
    blocks = parse_layout_blocks(raw)
    assert blocks is not None
    assert len(blocks) == 2
    assert blocks[0].text == "A"
    assert blocks[0].bbox is not None
    assert blocks[1].bbox is not None


def test_parse_layout_rejects_invented_out_of_range_boxes() -> None:
    raw = json.dumps({"blocks": [{"text": "X", "bbox": [0.0, 0.0, 2.0, 1.0]}]})
    blocks = parse_layout_blocks(raw)
    assert blocks is not None
    assert blocks[0].bbox is None


def test_assert_local_url_rejects_public_host() -> None:
    from torus.core.errors import ReaderError

    with pytest.raises(ReaderError, match="not local"):
        assert_local_url("https://example.com")


def test_parse_completion_logprobs_mean_probability() -> None:
    # Two tokens with logprob ln(0.5) → mean probability 0.5.
    lp = math.log(0.5)
    raw = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "OK"},
                    "logprobs": {
                        "content": [
                            {"token": "O", "logprob": lp},
                            {"token": "K", "logprob": lp},
                        ]
                    },
                }
            ]
        }
    )
    result = parse_completion(raw, openai_style=True)
    assert result.text == "OK"
    assert result.raw_score == pytest.approx(0.5)


def test_parse_completion_without_logprobs_score_is_none() -> None:
    raw = json.dumps({"choices": [{"message": {"content": "plain"}}]})
    result = parse_completion(raw, openai_style=True)
    assert result.text == "plain"
    assert result.raw_score is None


def test_reader_soft_fails_when_endpoint_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORUS_LLM_URL", raising=False)
    reader = LocalServerDocumentVlmReader(
        device="cpu",
        spec=ReaderSpec(
            name="olmocr",
            family="document_vlm",
            enabled=True,
            min_vram_gb=0,
            min_ram_gb=0,
            device=DeviceKind.CPU,
            capabilities=ReaderCaps(printed=True, layout=True),
            module="torus.readers.document_vlm.olmocr",
        ),
        model="olmOCR-2",
    )
    reading = reader.read(
        Image.new("RGB", (32, 16), (255, 255, 255)), _region(), ReadHint()
    )
    assert reading.error is not None
    assert "TORUS_LLM_URL" in reading.error
    assert reading.text == ""
    assert reading.raw_score is None


class _FakeVlmHandler(BaseHTTPRequestHandler):
    body: dict[str, Any] = {
        "choices": [{"message": {"content": "42"}}],
    }

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        payload = json.dumps(self.body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


@pytest.fixture
def local_vlm_server() -> Any:
    server = HTTPServer(("127.0.0.1", 0), _FakeVlmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _StubClient:
    def complete(self, messages: Any, *, image: Any = None) -> VlmCompletion:
        _ = messages, image
        return VlmCompletion(text="42", raw_score=None)


def test_local_server_reader_reads_via_stub_client() -> None:
    reader = LocalServerDocumentVlmReader(
        device="cpu",
        client=_StubClient(),
        model="fake-ocr",
        spec=ReaderSpec(
            name="glm_ocr",
            family="document_vlm",
            enabled=True,
            min_vram_gb=0,
            min_ram_gb=0,
            device=DeviceKind.CPU,
            capabilities=ReaderCaps(printed=True, layout=True),
            module="torus.readers.document_vlm.glm_ocr",
        ),
    )
    image = Image.new("RGB", (64, 32), (240, 240, 240))
    reading = reader.read(image, _region(), ReadHint(charset=Charset.DIGITS))
    assert reading.error is None
    assert reading.text == "42"
    assert reading.raw_score is None
    assert reading.reader_name == "glm_ocr"


def test_http_client_roundtrip(
    local_vlm_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORUS_LLM_URL", local_vlm_server)
    client = LocalDocumentVlmClient(
        local_vlm_server,
        model="fake-ocr",
        path="/v1/chat/completions",
        timeout_s=5.0,
    )
    result = client.complete(
        [{"role": "user", "content": "read"}],
        image=Image.new("RGB", (8, 8), (255, 255, 255)),
    )
    assert result.text == "42"
    assert result.raw_score is None


def test_olmocr_factory_default_model() -> None:
    spec = ReaderSpec(
        name="olmocr",
        family="document_vlm",
        enabled=False,
        min_vram_gb=0,
        min_ram_gb=0,
        device=DeviceKind.CPU,
        capabilities=ReaderCaps(printed=True, layout=True),
        module="torus.readers.document_vlm.olmocr",
    )
    reader = create_olmocr(device="cpu", spec=spec)
    assert reader.model_id == OLMOCR_MODEL


def test_instantiate_all_document_vlm_modules() -> None:
    modules = [
        "torus.readers.document_vlm.paddleocr_vl",
        "torus.readers.document_vlm.glm_ocr",
        "torus.readers.document_vlm.mineru",
        "torus.readers.document_vlm.dots_ocr",
        "torus.readers.document_vlm.olmocr",
        "torus.readers.document_vlm.deepseek_ocr",
        "torus.readers.document_vlm.chandra",
    ]
    for module in modules:
        leaf = module.rsplit(".", 1)[-1]
        spec = ReaderSpec(
            name=leaf,
            family="document_vlm",
            enabled=True,
            min_vram_gb=0,
            min_ram_gb=0,
            device=DeviceKind.CPU,
            capabilities=ReaderCaps(printed=True, layout=True),
            module=module,
        )
        reader = instantiate_reader(spec, device=DeviceKind.CPU)
        assert reader.name == leaf
        assert callable(reader.capabilities)
        assert callable(reader.read)


def test_repo_config_registers_document_vlm_disabled() -> None:
    from torus.config import load_readers
    from torus.config.loader import default_config_dir

    cfg = load_readers(default_config_dir() / "readers.yaml")
    names = [
        "paddleocr_vl",
        "glm_ocr",
        "mineru",
        "dots_ocr",
        "olmocr",
        "deepseek_ocr",
        "chandra",
    ]
    for name in names:
        spec = cfg.get(name)
        assert spec is not None
        assert spec.enabled is False
        assert spec.family == "document_vlm"
        assert spec.module is not None
        assert spec.module.startswith("torus.readers.document_vlm.")


def test_registry_skips_disabled_document_vlm() -> None:
    from torus.doctor.hardware import HardwareInfo

    cfg = ReadersConfig(
        readers=[
            ReaderSpec(
                name="olmocr",
                family="document_vlm",
                enabled=False,
                min_vram_gb=0,
                min_ram_gb=0,
                device=DeviceKind.CPU,
                capabilities=ReaderCaps(printed=True),
                module="torus.readers.document_vlm.olmocr",
            )
        ]
    )
    hw = HardwareInfo(
        os_name="Windows",
        os_release="10",
        arch="AMD64",
        python_version="3.12.0",
        cpu_name="Test CPU",
        cpu_count=4,
        ram_gb=32.0,
        gpus=(),
        cuda_available=False,
        notes=(),
    )
    reg = ReaderRegistry(
        cfg,
        hardware=hw,
        extras={"vlm": ExtraStatus("vlm", False, ("torch",))},
    )
    assert reg.skip_reasons().get("olmocr") == "disabled"
    assert reg.get("olmocr") is None
