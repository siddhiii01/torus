"""Resolve CPU/CUDA for a reader from its spec and doctor hardware probe."""

from __future__ import annotations

from torus.config.schemas import DeviceKind, ReaderSpec
from torus.doctor.hardware import HardwareInfo


def resolve_device(spec: ReaderSpec, hardware: HardwareInfo) -> DeviceKind:
    """Pick the concrete device a reader instance should use.

    Rules match ``torus doctor`` / SETUP: ``cpu`` stays on CPU; ``cuda`` requires
    a GPU that meets ``min_vram_gb``; ``auto`` prefers CUDA when VRAM is enough
    and otherwise falls back to CPU (never crashes).
    """
    if spec.device is DeviceKind.CPU:
        return DeviceKind.CPU

    if spec.device is DeviceKind.CUDA:
        if not hardware.cuda_available or hardware.max_vram_gb <= 0:
            return DeviceKind.CPU
        if spec.min_vram_gb > 0 and hardware.max_vram_gb < spec.min_vram_gb:
            return DeviceKind.CPU
        return DeviceKind.CUDA

    # AUTO
    if not hardware.cuda_available or hardware.max_vram_gb <= 0:
        return DeviceKind.CPU
    if spec.min_vram_gb > 0 and hardware.max_vram_gb < spec.min_vram_gb:
        return DeviceKind.CPU
    return DeviceKind.CUDA
