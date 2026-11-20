"""Assemble and format a doctor report (no document content)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from torus.config import TorusConfig, load_config
from torus.doctor.eligibility import ReaderAssessment, assess_readers
from torus.doctor.extras import ExtraStatus, detect_extras
from torus.doctor.hardware import HardwareInfo, probe_hardware
from torus.doctor.weights import model_dir


@dataclass(frozen=True)
class DoctorReport:
    """Full machine + reader eligibility snapshot."""

    hardware: HardwareInfo
    extras: dict[str, ExtraStatus]
    model_root: Path | None
    data_dir_set: bool
    config_hash: str
    readers: tuple[ReaderAssessment, ...]


def run_doctor(
    *,
    config: TorusConfig | None = None,
    config_dir: Path | str | None = None,
    hardware: HardwareInfo | None = None,
    extras: dict[str, ExtraStatus] | None = None,
    model_root: Path | None = None,
) -> DoctorReport:
    """Probe the machine and assess every reader in config.

    Injection points exist so unit tests never need real GPUs or extras.
    """
    cfg = config if config is not None else load_config(config_dir)
    hw = hardware if hardware is not None else probe_hardware()
    ex = extras if extras is not None else detect_extras()
    root = model_root if model_root is not None else model_dir()

    assessments = assess_readers(
        list(cfg.readers.readers),
        hw,
        ex,
        model_root=root,
    )
    return DoctorReport(
        hardware=hw,
        extras=ex,
        model_root=root,
        data_dir_set=bool(os.environ.get("TORUS_DATA_DIR")),
        config_hash=cfg.config_hash,
        readers=tuple(assessments),
    )


def format_report(report: DoctorReport) -> str:
    """Human-readable multi-line report for the CLI."""
    hw = report.hardware
    lines: list[str] = [
        "Torus doctor",
        "============",
        "",
        "Hardware",
        "--------",
        f"  OS:       {hw.os_name} {hw.os_release} ({hw.arch})",
        f"  Python:   {hw.python_version}",
        f"  CPU:      {hw.cpu_name} ({hw.cpu_count} logical)",
        f"  RAM:      {_fmt_gb(hw.ram_gb)}",
    ]
    if hw.gpus:
        lines.append(f"  CUDA:     yes ({len(hw.gpus)} GPU(s))")
        for gpu in hw.gpus:
            lines.append(
                f"    [{gpu.index}] {gpu.name} — {gpu.vram_total_gb:.1f} GiB VRAM"
            )
    else:
        lines.append("  CUDA:     no")
    for note in hw.notes:
        lines.append(f"  note:     {note}")

    lines.extend(
        [
            "",
            "Environment",
            "-----------",
            f"  TORUS_MODEL_DIR: {_fmt_path(report.model_root)}",
            f"  TORUS_DATA_DIR:  {'set' if report.data_dir_set else 'unset'}",
            f"  config hash:     {report.config_hash[:12]}…",
            "",
            "Extras",
            "------",
        ]
    )
    for name in sorted(report.extras):
        status = report.extras[name]
        if status.installed:
            lines.append(f"  {name}: installed")
        else:
            missing = ", ".join(status.missing_modules)
            lines.append(f"  {name}: missing ({missing})")

    lines.extend(["", "Readers", "-------"])
    if not report.readers:
        lines.append("  (none configured)")
    for assessment in report.readers:
        enabled = "enabled" if assessment.enabled else "disabled"
        lines.append(
            f"  {assessment.name} [{assessment.family}] — "
            f"{assessment.status.value} ({enabled})"
        )
        for reason in assessment.reasons:
            lines.append(f"    · {reason}")

    lines.append("")
    return "\n".join(lines)


def _fmt_gb(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f} GiB"


def _fmt_path(path: Path | None) -> str:
    if path is None:
        return "unset"
    return str(path)
