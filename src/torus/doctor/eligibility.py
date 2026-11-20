"""Map hardware + extras + weights onto per-reader status labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from torus.config.schemas import DeviceKind, ReaderSpec
from torus.doctor.extras import ExtraStatus, family_extra_installed
from torus.doctor.hardware import HardwareInfo
from torus.doctor.weights import weights_present


class ReaderStatus(StrEnum):
    """Labels printed by ``torus doctor`` (see docs/SETUP.md §4)."""

    READY = "ready"
    CAN_RUN_WEIGHTS_MISSING = "can run (weights missing)"
    NOT_SUPPORTED = "not supported here"


@dataclass(frozen=True)
class ReaderAssessment:
    """Eligibility result for one reader from ``config/readers.yaml``."""

    name: str
    family: str
    enabled: bool
    status: ReaderStatus
    reasons: tuple[str, ...]
    device: DeviceKind
    min_ram_gb: float
    min_vram_gb: float


def hardware_supports(
    reader: ReaderSpec,
    hardware: HardwareInfo,
) -> tuple[bool, list[str]]:
    """Check RAM / VRAM / device preference against probed hardware."""
    reasons: list[str] = []

    if hardware.ram_gb is not None and hardware.ram_gb < reader.min_ram_gb:
        reasons.append(
            f"RAM {hardware.ram_gb:.1f} GiB < required {reader.min_ram_gb:.1f} GiB"
        )
    elif hardware.ram_gb is None and reader.min_ram_gb > 0:
        reasons.append(
            f"RAM unknown; cannot verify required {reader.min_ram_gb:.1f} GiB"
        )

    device = reader.device
    if device is DeviceKind.CPU:
        # CPU readers ignore VRAM; min_vram_gb in config is informational only.
        return (not reasons), reasons

    if device is DeviceKind.CUDA:
        if not hardware.cuda_available or hardware.max_vram_gb <= 0:
            reasons.append("CUDA GPU required but none detected")
        elif hardware.max_vram_gb < reader.min_vram_gb:
            reasons.append(
                f"VRAM {hardware.max_vram_gb:.1f} GiB < required "
                f"{reader.min_vram_gb:.1f} GiB"
            )
        return (not reasons), reasons

    # AUTO: GPU preferred when available; CPU fallback is allowed (SETUP §3).
    if (
        reader.min_vram_gb > 0
        and hardware.cuda_available
        and hardware.max_vram_gb < reader.min_vram_gb
    ):
        # Undersized GPU is fine: reader can still use CPU under auto.
        pass
    return (not reasons), reasons


def assess_reader(
    reader: ReaderSpec,
    hardware: HardwareInfo,
    extras: dict[str, ExtraStatus],
    *,
    model_root: Path | None = None,
) -> ReaderAssessment:
    """Decide ready / weights missing / not supported for one reader."""
    reasons: list[str] = []

    hw_ok, hw_reasons = hardware_supports(reader, hardware)
    reasons.extend(hw_reasons)

    extra_ok, extra_reason = family_extra_installed(reader.family, extras)
    if not extra_ok and extra_reason:
        reasons.append(extra_reason)

    if not hw_ok or not extra_ok:
        status = ReaderStatus.NOT_SUPPORTED
        out_reasons = tuple(reasons)
    elif not weights_present(reader.model, root=model_root):
        status = ReaderStatus.CAN_RUN_WEIGHTS_MISSING
        if model_root is None and reader.model and not Path(reader.model).is_absolute():
            out_reasons = (
                "weights missing (TORUS_MODEL_DIR unset and model path not absolute)",
            )
        else:
            out_reasons = (f"weights missing for model {reader.model!r}",)
    else:
        status = ReaderStatus.READY
        out_reasons = ()

    return ReaderAssessment(
        name=reader.name,
        family=reader.family,
        enabled=reader.enabled,
        status=status,
        reasons=out_reasons,
        device=reader.device,
        min_ram_gb=reader.min_ram_gb,
        min_vram_gb=reader.min_vram_gb,
    )


def assess_readers(
    readers: list[ReaderSpec],
    hardware: HardwareInfo,
    extras: dict[str, ExtraStatus],
    *,
    model_root: Path | None = None,
) -> list[ReaderAssessment]:
    """Assess every reader in config order."""
    return [assess_reader(r, hardware, extras, model_root=model_root) for r in readers]
