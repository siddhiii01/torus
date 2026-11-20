"""Reader registry: eligibility from doctor, lazy load, skip-not-crash."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torus.config.schemas import DeviceKind, ReadersConfig, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.protocols import Reader
from torus.doctor.eligibility import (
    ReaderAssessment,
    ReaderStatus,
    assess_reader,
)
from torus.doctor.extras import ExtraStatus, detect_extras
from torus.doctor.hardware import HardwareInfo, probe_hardware
from torus.doctor.weights import model_dir
from torus.logging import get_logger
from torus.readers.device import resolve_device
from torus.readers.load import instantiate_reader

log = get_logger(__name__)


@dataclass
class RegisteredReader:
    """One config entry with eligibility and a lazily created instance."""

    spec: ReaderSpec
    assessment: ReaderAssessment
    device: DeviceKind
    skip_reason: str | None = None
    _instance: Reader | None = field(default=None, repr=False, compare=False)
    _load_error: str | None = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def available(self) -> bool:
        return self.skip_reason is None

    def get(self) -> Reader | None:
        """Return the reader instance, loading on first use. Soft-skip on error."""
        if self.skip_reason is not None:
            return None
        if self._instance is not None:
            return self._instance
        if self._load_error is not None:
            return None
        try:
            self._instance = instantiate_reader(self.spec, device=self.device)
        except ReaderError as exc:
            self._load_error = str(exc)
            self.skip_reason = self._load_error
            log.info(
                "reader_load_skipped",
                reader_name=self.spec.name,
                error_type=type(exc).__name__,
                status_code="skipped",
            )
            return None
        return self._instance


class ReaderRegistry:
    """Config-driven registry. Missing / disabled / unsupported → skip."""

    def __init__(
        self,
        config: ReadersConfig,
        *,
        hardware: HardwareInfo | None = None,
        extras: dict[str, ExtraStatus] | None = None,
        model_root: Path | None = None,
        require_enabled: bool = True,
        require_ready: bool = True,
    ) -> None:
        self._config = config
        self._hardware = hardware if hardware is not None else probe_hardware()
        self._extras = extras if extras is not None else detect_extras()
        root = model_root if model_root is not None else model_dir()
        self._by_name: dict[str, RegisteredReader] = {}

        for spec in config.readers:
            assessment = assess_reader(
                spec,
                self._hardware,
                self._extras,
                model_root=root,
            )
            skip = _skip_reason(
                spec,
                assessment,
                require_enabled=require_enabled,
                require_ready=require_ready,
            )
            device = resolve_device(spec, self._hardware)
            entry = RegisteredReader(
                spec=spec,
                assessment=assessment,
                device=device,
                skip_reason=skip,
            )
            self._by_name[spec.name] = entry

    @property
    def config(self) -> ReadersConfig:
        return self._config

    @property
    def hardware(self) -> HardwareInfo:
        return self._hardware

    def names(self) -> list[str]:
        return list(self._by_name)

    def available_names(self) -> list[str]:
        return [n for n, e in self._by_name.items() if e.available]

    def get_entry(self, name: str) -> RegisteredReader | None:
        return self._by_name.get(name)

    def get(self, name: str) -> Reader | None:
        entry = self._by_name.get(name)
        if entry is None:
            return None
        return entry.get()

    def skip_reasons(self) -> dict[str, str]:
        return {
            name: entry.skip_reason
            for name, entry in self._by_name.items()
            if entry.skip_reason is not None
        }


def _skip_reason(
    spec: ReaderSpec,
    assessment: ReaderAssessment,
    *,
    require_enabled: bool,
    require_ready: bool,
) -> str | None:
    if require_enabled and not spec.enabled:
        return "disabled"
    if not spec.module:
        return "no module path"
    if require_ready and assessment.status is not ReaderStatus.READY:
        if assessment.reasons:
            return "; ".join(assessment.reasons)
        return assessment.status.value
    return None
