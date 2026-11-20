"""Unit tests for torus doctor (P0.6). No real GPUs or model stacks required."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from torus.cli import main
from torus.config.schemas import DeviceKind, ReaderCaps, ReaderSpec
from torus.doctor import (
    HardwareInfo,
    ReaderStatus,
    assess_reader,
    detect_extras,
    format_report,
    probe_hardware,
    run_doctor,
    weights_present,
)
from torus.doctor.extras import ExtraStatus, family_extra_installed, module_available
from torus.doctor.hardware import GpuInfo


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


def _reader(
    *,
    name: str = "r1",
    family: str = "stub",
    device: DeviceKind = DeviceKind.CPU,
    min_ram_gb: float = 0.0,
    min_vram_gb: float = 0.0,
    model: str | None = None,
    enabled: bool = False,
) -> ReaderSpec:
    return ReaderSpec(
        name=name,
        family=family,
        enabled=enabled,
        min_vram_gb=min_vram_gb,
        min_ram_gb=min_ram_gb,
        device=device,
        capabilities=ReaderCaps(printed=True),
        model=model,
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


def test_probe_hardware_returns_basic_fields() -> None:
    info = probe_hardware()
    assert info.os_name
    assert info.cpu_count >= 1
    assert info.python_version
    assert isinstance(info.gpus, tuple)


def test_detect_extras_keys_match_pyproject() -> None:
    extras = detect_extras()
    assert set(extras) >= {"paddle", "doctr", "vlm", "train", "review"}
    # Base install: pydantic is present; paddle markers usually are not.
    assert module_available("pydantic") is True


def test_weights_present_none_and_paths(tmp_path: Path) -> None:
    assert weights_present(None) is True
    assert weights_present("") is True
    assert weights_present("missing-model", root=tmp_path) is False
    (tmp_path / "mymodel").mkdir()
    assert weights_present("mymodel", root=tmp_path) is True
    abs_file = tmp_path / "abs.bin"
    abs_file.write_bytes(b"x")
    assert weights_present(str(abs_file), root=None) is True


def test_family_extra_stub_ok_paddle_requires() -> None:
    extras = _extras(paddle=False)
    assert family_extra_installed("stub", extras) == (True, None)
    ok, reason = family_extra_installed("paddle", extras)
    assert ok is False
    assert reason is not None
    assert "paddle" in reason


def test_assess_ready_stub() -> None:
    result = assess_reader(_reader(), _hw(), _extras())
    assert result.status is ReaderStatus.READY
    assert result.reasons == ()


def test_assess_not_supported_low_ram() -> None:
    result = assess_reader(
        _reader(min_ram_gb=64.0),
        _hw(ram_gb=8.0),
        _extras(),
    )
    assert result.status is ReaderStatus.NOT_SUPPORTED
    assert any("RAM" in r for r in result.reasons)


def test_assess_not_supported_cuda_without_gpu() -> None:
    result = assess_reader(
        _reader(device=DeviceKind.CUDA, min_vram_gb=4.0),
        _hw(cuda=False),
        _extras(),
    )
    assert result.status is ReaderStatus.NOT_SUPPORTED
    assert any("CUDA" in r for r in result.reasons)


def test_assess_cuda_ok_with_vram() -> None:
    result = assess_reader(
        _reader(device=DeviceKind.CUDA, min_vram_gb=4.0),
        _hw(cuda=True, vram_gb=8.0),
        _extras(),
    )
    assert result.status is ReaderStatus.READY


def test_assess_auto_falls_back_to_cpu_without_gpu() -> None:
    result = assess_reader(
        _reader(device=DeviceKind.AUTO, min_vram_gb=2.0, min_ram_gb=4.0),
        _hw(ram_gb=16.0, cuda=False),
        _extras(),
    )
    assert result.status is ReaderStatus.READY


def test_assess_weights_missing(tmp_path: Path) -> None:
    result = assess_reader(
        _reader(model="weights/v1"),
        _hw(),
        _extras(),
        model_root=tmp_path,
    )
    assert result.status is ReaderStatus.CAN_RUN_WEIGHTS_MISSING
    assert any("weights missing" in r for r in result.reasons)


def test_assess_paddle_extra_missing_beats_weights() -> None:
    result = assess_reader(
        _reader(family="paddle", model="w"),
        _hw(),
        _extras(paddle=False),
        model_root=Path("/nonexistent"),
    )
    assert result.status is ReaderStatus.NOT_SUPPORTED
    assert any("extra" in r for r in result.reasons)


def test_assess_paddle_ready_when_extra_and_no_model() -> None:
    result = assess_reader(
        _reader(family="paddle"),
        _hw(ram_gb=8.0),
        _extras(paddle=True),
    )
    assert result.status is ReaderStatus.READY


def _write_minimal_config(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "templates").mkdir()
    (root / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "dpi": 200,
                "confidence": {"confirmed_min": 0.9, "likely_min": 0.7},
                "voting": {
                    "critical_min_families": 2,
                    "noncritical_min_readers": 1,
                },
                "regression": {
                    "noise_margin": 0.01,
                    "silent_error_tolerance": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "readers.yaml").write_text(
        yaml.safe_dump(
            {
                "readers": [
                    {
                        "name": "stub_echo",
                        "family": "stub",
                        "enabled": False,
                        "min_vram_gb": 0,
                        "min_ram_gb": 0,
                        "device": "cpu",
                    },
                    {
                        "name": "paddle_printed",
                        "family": "paddle",
                        "enabled": False,
                        "min_vram_gb": 0,
                        "min_ram_gb": 4,
                        "device": "cpu",
                        "model": "paddle/ocr",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "page_types.yaml").write_text(
        yaml.safe_dump(
            {
                "page_types": [
                    {"id": "other", "description": "Catch-all."},
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "templates" / "demo_v1.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "demo_v1",
                "version": 1,
                "description": "demo",
                "fields": [
                    {
                        "key": "x",
                        "excel_header": "X",
                        "description": "x",
                        "type": "text",
                        "look_in": ["other"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_doctor_and_format(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    _write_minimal_config(cfg_dir)
    model_root = tmp_path / "models"
    model_root.mkdir()

    report = run_doctor(
        config_dir=cfg_dir,
        hardware=_hw(ram_gb=16.0),
        extras=_extras(paddle=True),
        model_root=model_root,
    )
    assert len(report.readers) == 2
    by_name = {r.name: r for r in report.readers}
    assert by_name["stub_echo"].status is ReaderStatus.READY
    assert by_name["paddle_printed"].status is ReaderStatus.CAN_RUN_WEIGHTS_MISSING

    text = format_report(report)
    assert "Torus doctor" in text
    assert "stub_echo" in text
    assert "ready" in text
    assert "can run (weights missing)" in text
    # Metadata only — no synthetic field values from fixtures beyond names.
    assert "excel_header" not in text


def test_cli_doctor_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_dir = tmp_path / "config"
    _write_minimal_config(cfg_dir)
    code = main(["doctor", "--config-dir", str(cfg_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Torus doctor" in captured.out
    assert "Readers" in captured.out


def test_cli_doctor_bad_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["doctor", "--config-dir", str(tmp_path / "missing")])
    captured = capsys.readouterr()
    assert code == 1
    assert "config error" in captured.err
