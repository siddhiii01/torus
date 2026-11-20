"""OS / CPU / RAM / GPU probes. Stdlib only; works on Windows and Linux."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    """One visible GPU (NVIDIA via nvidia-smi when available)."""

    index: int
    name: str
    vram_total_gb: float


@dataclass(frozen=True)
class HardwareInfo:
    """Local machine facts used for reader eligibility."""

    os_name: str
    os_release: str
    arch: str
    python_version: str
    cpu_name: str
    cpu_count: int
    ram_gb: float | None
    gpus: tuple[GpuInfo, ...] = field(default_factory=tuple)
    cuda_available: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_vram_gb(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.vram_total_gb for g in self.gpus)


def probe_hardware() -> HardwareInfo:
    """Detect OS, CPU, RAM and GPU without importing optional stacks."""
    notes: list[str] = []
    ram_gb = _probe_ram_gb()
    if ram_gb is None:
        notes.append("RAM could not be determined on this platform.")

    gpus, gpu_notes = _probe_gpus()
    notes.extend(gpu_notes)

    return HardwareInfo(
        os_name=platform.system() or "unknown",
        os_release=platform.release() or "unknown",
        arch=platform.machine() or "unknown",
        python_version=platform.python_version(),
        cpu_name=_probe_cpu_name(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=ram_gb,
        gpus=tuple(gpus),
        cuda_available=bool(gpus),
        notes=tuple(notes),
    )


def _probe_cpu_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip() or platform.processor()
    if system == "Windows":
        name = _windows_cpu_name()
        if name:
            return name
    proc = platform.processor()
    return proc.strip() if proc else "unknown"


def _windows_cpu_name() -> str | None:
    # Prefer CIM via PowerShell; avoid wmic (deprecated on newer Windows).
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps:
        try:
            completed = subprocess.run(
                [
                    ps,
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode == 0:
            lines = completed.stdout.strip().splitlines()
            if lines:
                return lines[0].strip() or None
    return None


def _probe_ram_gb() -> float | None:
    system = platform.system()
    if system == "Linux":
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    kib = int(parts[1])
                    return round(kib / (1024 * 1024), 2)
        return None
    if system == "Windows":
        return _windows_ram_gb()
    if system == "Darwin":
        return _darwin_ram_gb()
    return None


def _windows_ram_gb() -> float | None:
    import sys

    if sys.platform != "win32":
        return None

    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    # windll exists only on Windows; guarded by sys.platform above.
    ok = bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)))
    if not ok:
        return None
    total_bytes = int(stat.ullTotalPhys)
    return round(total_bytes / (1024**3), 2)


def _darwin_ram_gb() -> float | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    raw = completed.stdout.strip()
    if not raw.isdigit():
        return None
    return round(int(raw) / (1024**3), 2)


def _probe_gpus() -> tuple[list[GpuInfo], list[str]]:
    notes: list[str] = []
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        notes.append("nvidia-smi not found; treating as no NVIDIA GPU.")
        return [], notes
    try:
        completed = subprocess.run(
            [
                nvsmi,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"nvidia-smi failed: {exc}")
        return [], notes
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        notes.append(f"nvidia-smi exited {completed.returncode}: {err or 'no output'}")
        return [], notes

    gpus: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            # memory.total is MiB when nounits is set
            vram_mib = float(re.sub(r"[^\d.]", "", parts[2]) or "0")
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=parts[1],
                vram_total_gb=round(vram_mib / 1024.0, 2),
            )
        )
    if not gpus:
        notes.append("nvidia-smi returned no GPU rows.")
    return gpus, notes
