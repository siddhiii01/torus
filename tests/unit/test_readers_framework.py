"""Unit tests for the reader framework (P5.1). No model stacks required."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from torus.config.schemas import (
    DeviceKind,
    ReaderCaps,
    ReadersConfig,
    ReaderSpec,
)
from torus.core.models import BBox, ReadHint, Reading, Region
from torus.core.types import RegionKind
from torus.doctor.eligibility import ReaderStatus
from torus.doctor.extras import ExtraStatus
from torus.doctor.hardware import GpuInfo, HardwareInfo
from torus.readers import (
    ReaderRegistry,
    ReaderRunner,
    ReadingCache,
    crop_hash,
    resolve_device,
    run_with_timeout,
)
from torus.readers.load import instantiate_reader
from torus.readers.stub.echo import EchoReader, create_reader


def _hw(
    *,
    ram_gb: float | None = 16.0,
    vram_gb: float = 0.0,
    cuda: bool = False,
) -> HardwareInfo:
    gpus: tuple[GpuInfo, ...] = ()
    if cuda and vram_gb > 0:
        gpus = (GpuInfo(index=0, name="Test GPU", vram_total_gb=vram_gb),)
    return HardwareInfo(
        os_name="Linux",
        os_release="6.0",
        arch="x86_64",
        python_version="3.12.0",
        cpu_name="Test CPU",
        cpu_count=4,
        ram_gb=ram_gb,
        gpus=gpus,
        cuda_available=cuda and bool(gpus),
        notes=(),
    )


def _extras(**installed: bool) -> dict[str, ExtraStatus]:
    base = {
        "paddle": ExtraStatus("paddle", False, ("paddle", "paddleocr")),
        "doctr": ExtraStatus("doctr", False, ("doctr",)),
        "vlm": ExtraStatus("vlm", False, ("torch", "transformers")),
        "train": ExtraStatus("train", False, ("accelerate",)),
        "review": ExtraStatus("review", False, ("streamlit",)),
    }
    out: dict[str, ExtraStatus] = {}
    for name, status in base.items():
        if installed.get(name, False):
            out[name] = ExtraStatus(name, True, ())
        else:
            out[name] = status
    return out


def _spec(
    *,
    name: str = "stub_echo",
    family: str = "stub",
    enabled: bool = True,
    device: DeviceKind = DeviceKind.CPU,
    module: str | None = "torus.readers.stub.echo",
    version: str | None = "0",
    min_vram_gb: float = 0.0,
    min_ram_gb: float = 0.0,
    model: str | None = None,
) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family=family,
        enabled=enabled,
        min_vram_gb=min_vram_gb,
        min_ram_gb=min_ram_gb,
        device=device,
        capabilities=ReaderCaps(printed=True, scripts=["latin"]),
        model=model,
        version=version,
        module=module,
    )


def _region(rid: str = "r1") -> Region:
    return Region(
        id=rid,
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.3),
        kind=RegionKind.FIELD_VALUE,
    )


def _rgb(color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (40, 20), color)


def _runner_with(
    reader: Any,
    *,
    name: str,
    timeout_s: float = 5.0,
    batch_size: int = 2,
    cache_max: int = 64,
    version: str = "1",
    cache: ReadingCache | None = None,
) -> tuple[ReaderRunner, Any]:
    cfg = ReadersConfig(
        timeout_s=timeout_s,
        batch_size=batch_size,
        cache_max_entries=cache_max,
        readers=[
            _spec(
                name=name,
                module="torus.readers.stub.echo",
                version=version,
            )
        ],
    )
    reg = ReaderRegistry(cfg, hardware=_hw(), extras=_extras())
    entry = reg.get_entry(name)
    assert entry is not None
    entry._instance = reader  # noqa: SLF001 — test injection
    entry.skip_reason = None
    entry.spec = entry.spec.model_copy(update={"version": version})
    runner = ReaderRunner(
        reg,
        timeout_s=timeout_s,
        batch_size=batch_size,
        cache=cache,
    )
    return runner, reader


# --- device -----------------------------------------------------------------


def test_resolve_device_cpu_and_auto_fallback() -> None:
    cpu_spec = _spec(device=DeviceKind.CPU)
    assert resolve_device(cpu_spec, _hw()) is DeviceKind.CPU

    auto = _spec(device=DeviceKind.AUTO, min_vram_gb=2.0)
    assert resolve_device(auto, _hw()) is DeviceKind.CPU
    assert resolve_device(auto, _hw(cuda=True, vram_gb=8.0)) is DeviceKind.CUDA
    assert resolve_device(auto, _hw(cuda=True, vram_gb=1.0)) is DeviceKind.CPU


def test_resolve_device_explicit_cuda() -> None:
    cuda_spec = _spec(device=DeviceKind.CUDA, min_vram_gb=4.0)
    assert resolve_device(cuda_spec, _hw(cuda=True, vram_gb=8.0)) is DeviceKind.CUDA
    assert resolve_device(cuda_spec, _hw()) is DeviceKind.CPU


# --- cache / crop hash ------------------------------------------------------


def test_crop_hash_differs_for_different_pixels() -> None:
    region = _region()
    h1 = crop_hash(_rgb((10, 20, 30)), region)
    h2 = crop_hash(_rgb((200, 100, 50)), region)
    assert h1 != h2
    assert len(h1) == 64


def test_reading_cache_lru_and_skips_errors() -> None:
    cache = ReadingCache(max_entries=2)
    ok = Reading(
        region_id="r1",
        reader_name="stub",
        reader_version="0",
        text="A",
    )
    err = Reading(
        region_id="r1",
        reader_name="stub",
        reader_version="0",
        error="fail",
    )
    cache.put("k1", ok)
    cache.put("k_err", err)
    assert len(cache) == 1
    assert cache.get("k_err") is None

    cache.put("k2", ok.model_copy(update={"text": "B"}))
    cache.put("k3", ok.model_copy(update={"text": "C"}))
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    assert cache.get("k3") is not None


def test_cache_disabled_when_max_zero() -> None:
    cache = ReadingCache(max_entries=0)
    reading = Reading(region_id="r1", reader_name="stub", reader_version="0", text="x")
    cache.put("k", reading)
    assert cache.get("k") is None


# --- timeout ----------------------------------------------------------------


def test_run_with_timeout_success_and_deadline() -> None:
    assert run_with_timeout(lambda: 7, timeout_s=1.0) == 7
    with pytest.raises(TimeoutError):
        run_with_timeout(lambda: time.sleep(0.5), timeout_s=0.05)


# --- stub + load ------------------------------------------------------------


def test_stub_echo_create_reader_no_invented_confidence() -> None:
    spec = _spec()
    reader = create_reader(device="cpu", spec=spec)
    assert isinstance(reader, EchoReader)
    reading = reader.read(_rgb(), _region(), ReadHint())
    assert reading.text == ""
    assert reading.raw_score is None
    assert reading.char_confidences is None
    assert reading.error is None


def test_instantiate_reader_from_module() -> None:
    reader = instantiate_reader(_spec(), device=DeviceKind.CPU)
    assert reader.name == "stub_echo"
    assert reader.capabilities().printed is True


# --- registry ---------------------------------------------------------------


def test_registry_skips_disabled_and_missing_module() -> None:
    cfg = ReadersConfig(
        readers=[
            _spec(name="off", enabled=False),
            _spec(
                name="missing_mod",
                enabled=True,
                module="torus.readers.does_not_exist.mod",
            ),
            _spec(name="ok", enabled=True),
        ]
    )
    reg = ReaderRegistry(cfg, hardware=_hw(), extras=_extras())
    assert reg.get("off") is None
    assert "disabled" in (reg.skip_reasons().get("off") or "")

    entry = reg.get_entry("missing_mod")
    assert entry is not None
    assert entry.available is True
    assert reg.get("missing_mod") is None
    assert entry.available is False

    assert "ok" in reg.available_names()
    assert reg.get("ok") is not None


def test_registry_skips_not_ready_weights() -> None:
    cfg = ReadersConfig(
        readers=[
            _spec(
                name="need_weights",
                enabled=True,
                model="nonexistent-weights-xyz",
            )
        ]
    )
    reg = ReaderRegistry(
        cfg,
        hardware=_hw(),
        extras=_extras(),
        model_root=Path("/tmp/torus-no-models"),
    )
    entry = reg.get_entry("need_weights")
    assert entry is not None
    assert entry.assessment.status is ReaderStatus.CAN_RUN_WEIGHTS_MISSING
    assert entry.available is False
    assert reg.get("need_weights") is None


def test_registry_lazy_load_only_on_get() -> None:
    """Disabled paddle entry must not be imported when building the registry."""
    # Other tests (P5.2) legitimately import the reader module; evict it so this
    # asserts the registry itself does not.
    for loaded in [m for m in sys.modules if m.startswith("torus.readers.paddle")]:
        del sys.modules[loaded]
    cfg = ReadersConfig(
        readers=[
            _spec(
                name="paddle_printed",
                family="paddle",
                enabled=False,
                module="torus.readers.paddle.printed",
            ),
            _spec(name="stub_echo", enabled=True),
        ]
    )
    reg = ReaderRegistry(cfg, hardware=_hw(), extras=_extras())
    assert "torus.readers.paddle" not in sys.modules
    assert reg.get("stub_echo") is not None
    assert "torus.readers.paddle" not in sys.modules


# --- runner -----------------------------------------------------------------


class _CountingReader:
    name = "counter"

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> ReaderCaps:
        return ReaderCaps(printed=True)

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        _ = image, hint
        self.calls += 1
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version="1",
            text=f"t-{region.id}",
            raw_score=None,
        )


class _SlowReader:
    name = "slow"

    def capabilities(self) -> ReaderCaps:
        return ReaderCaps(printed=True)

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        _ = image, hint
        time.sleep(0.4)
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version="1",
            text="late",
        )


class _BatchReader:
    name = "batchy"

    def __init__(self) -> None:
        self.batch_calls = 0
        self.read_calls = 0

    def capabilities(self) -> ReaderCaps:
        return ReaderCaps(printed=True)

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        _ = image, hint
        self.read_calls += 1
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version="1",
            text=region.id,
        )

    def read_batch(
        self, image: Any, regions: list[Region], hint: ReadHint
    ) -> list[Reading]:
        _ = image, hint
        self.batch_calls += 1
        return [
            Reading(
                region_id=r.id,
                reader_name=self.name,
                reader_version="1",
                text=r.id,
            )
            for r in regions
        ]


def test_runner_caches_by_crop_hash_and_version() -> None:
    runner, counter = _runner_with(_CountingReader(), name="counter", version="1")
    img = _rgb((1, 2, 3))
    region = _region("a")
    r1 = runner.read("counter", img, region)
    r2 = runner.read("counter", img, region)
    assert r1.text == "t-a"
    assert r2.text == "t-a"
    assert counter.calls == 1

    runner.read("counter", _rgb((9, 9, 9)), region)
    assert counter.calls == 2

    shared = ReadingCache(64)
    runner_a, c_a = _runner_with(
        _CountingReader(), name="counter", version="1", cache=shared
    )
    runner_b, c_b = _runner_with(
        _CountingReader(), name="counter", version="2", cache=shared
    )
    img2 = _rgb((5, 5, 5))
    runner_a.read("counter", img2, region)
    runner_b.read("counter", img2, region)
    assert c_a.calls == 1
    assert c_b.calls == 1


def test_runner_timeout_soft_fails() -> None:
    runner, _ = _runner_with(_SlowReader(), name="slow", timeout_s=0.05)
    reading = runner.read("slow", _rgb(), _region())
    assert reading.error is not None
    assert "exceeded" in reading.error
    assert reading.text == ""
    assert reading.raw_score is None


def test_runner_batch_chunks_and_preserves_order() -> None:
    runner, batchy = _runner_with(
        _BatchReader(), name="batchy", batch_size=2, timeout_s=5.0
    )
    regions = [_region(f"r{i}") for i in range(5)]
    readings = runner.read_many("batchy", _rgb(), regions)
    assert [r.text for r in readings] == [f"r{i}" for i in range(5)]
    assert batchy.batch_calls == 3
    assert batchy.read_calls == 0


def test_runner_unknown_reader_soft_fails() -> None:
    cfg = ReadersConfig(readers=[_spec(enabled=True)])
    reg = ReaderRegistry(cfg, hardware=_hw(), extras=_extras())
    runner = ReaderRunner(reg)
    reading = runner.read("nope", _rgb(), _region())
    assert reading.error is not None
    assert "unknown reader" in reading.error


def test_disabled_reader_does_not_crash_pipeline() -> None:
    cfg = ReadersConfig(readers=[_spec(enabled=False)])
    reg = ReaderRegistry(cfg, hardware=_hw(), extras=_extras())
    runner = ReaderRunner(reg)
    reading = runner.read("stub_echo", _rgb(), _region())
    assert reading.error == "disabled"
    assert reading.text == ""


def test_repo_readers_config_loads_framework_knobs() -> None:
    from torus.config import load_readers
    from torus.config.loader import default_config_dir

    cfg = load_readers(default_config_dir() / "readers.yaml")
    assert cfg.timeout_s == 30.0
    assert cfg.batch_size == 8
    assert cfg.cache_max_entries == 1024
    stub = cfg.get("stub_echo")
    assert stub is not None
    assert stub.module == "torus.readers.stub.echo"
