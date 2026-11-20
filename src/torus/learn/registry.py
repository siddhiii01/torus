"""Model registry (P10.3): versioned fine-tuned weights under TORUS_MODEL_DIR.

Layout::

    <root>/<reader>/<version>/entry.json   immutable metadata
    <root>/<reader>/<version>/files/...    the weights (copied in, hashed)
    <root>/<reader>/current.json           pointer written only by promotion

Entries are never edited after registration. Weights never go in git; metadata
holds hashes and metrics, no document content.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.config.schemas import ReadersConfig
from torus.learn.dataset import LearnError

REGISTRY_SUBDIR = "registry"
CURRENT = "current"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    reader: str
    version: str
    base_model: str = Field(min_length=1)
    recipe: str = Field(min_length=1)
    data_hash: str = Field(min_length=1)
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    files: dict[str, str] = Field(description="relative path -> sha256")


class CurrentPointer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    previous: str | None = None
    promoted_at: datetime


def model_root() -> Path:
    raw = os.environ.get("TORUS_MODEL_DIR", "").strip()
    if not raw or not Path(raw).is_dir():
        raise LearnError("TORUS_MODEL_DIR must be set to an existing directory")
    return Path(raw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str, what: str) -> str:
    if not _SAFE.match(name):
        raise LearnError(f"{what} is not a safe name")
    return name


class ModelRegistry:
    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._root = root
        self._clock = clock

    @classmethod
    def from_env(cls) -> ModelRegistry:
        return cls(model_root() / REGISTRY_SUBDIR)

    def _reader_dir(self, reader: str) -> Path:
        return self._root / _safe(reader, "reader name")

    def _version_dir(self, reader: str, version: str) -> Path:
        return self._reader_dir(reader) / _safe(version, "version")

    def versions(self, reader: str) -> list[str]:
        base = self._reader_dir(reader)
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if (p / "entry.json").is_file()]
        return sorted(
            found, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v)
        )

    def register(
        self,
        reader: str,
        weights: Path,
        *,
        base_model: str,
        recipe: str,
        data_hash: str,
        metrics: Mapping[str, float] | None = None,
    ) -> RegistryEntry:
        """Copy ``weights`` (file or folder) in as the next integer version."""
        if not weights.exists():
            raise LearnError("weights path does not exist")
        existing = [int(v) for v in self.versions(reader) if v.isdigit()]
        version = str(max(existing, default=0) + 1)
        vdir = self._version_dir(reader, version)
        files_dir = vdir / "files"
        files_dir.mkdir(parents=True)
        if weights.is_dir():
            shutil.copytree(weights, files_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(weights, files_dir / weights.name)
        hashes = {
            p.relative_to(files_dir).as_posix(): _sha256(p)
            for p in sorted(files_dir.rglob("*"))
            if p.is_file()
        }
        if not hashes:
            shutil.rmtree(vdir)
            raise LearnError("weights contain no files")
        entry = RegistryEntry(
            reader=reader,
            version=version,
            base_model=base_model,
            recipe=recipe,
            data_hash=data_hash,
            created_at=self._clock(),
            metrics=dict(metrics or {}),
            files=hashes,
        )
        (vdir / "entry.json").write_text(entry.model_dump_json(indent=2), "utf-8")
        return entry

    def get(self, reader: str, version: str) -> RegistryEntry:
        path = self._version_dir(reader, version) / "entry.json"
        if not path.is_file():
            raise LearnError("registry version not found")
        try:
            return RegistryEntry.model_validate_json(path.read_text("utf-8"))
        except (ValidationError, ValueError) as exc:
            raise LearnError("invalid registry entry") from exc

    def verify(self, reader: str, version: str) -> bool:
        """True when every stored weight file still matches its recorded hash."""
        entry = self.get(reader, version)
        files_dir = self._version_dir(reader, version) / "files"
        for rel, digest in entry.files.items():
            path = files_dir / rel
            if not path.is_file() or _sha256(path) != digest:
                return False
        return True

    def weights_path(self, reader: str, version: str) -> Path:
        """Single weights file when there is exactly one, else the folder."""
        entry = self.get(reader, version)
        files_dir = self._version_dir(reader, version) / "files"
        if len(entry.files) == 1:
            return files_dir / next(iter(entry.files))
        return files_dir

    def current(self, reader: str) -> CurrentPointer | None:
        path = self._reader_dir(reader) / "current.json"
        if not path.is_file():
            return None
        try:
            return CurrentPointer.model_validate_json(path.read_text("utf-8"))
        except (ValidationError, ValueError) as exc:
            raise LearnError("invalid current pointer") from exc

    def set_current(self, reader: str, version: str) -> CurrentPointer:
        """Point ``current`` at ``version``. Only promotion should call this."""
        self.get(reader, version)
        if not self.verify(reader, version):
            raise LearnError("weights do not match their recorded hash")
        old = self.current(reader)
        pointer = CurrentPointer(
            version=version,
            previous=old.version if old else None,
            promoted_at=self._clock(),
        )
        (self._reader_dir(reader) / "current.json").write_text(
            pointer.model_dump_json(indent=2), "utf-8"
        )
        return pointer

    def restore_current(self, reader: str, previous: CurrentPointer | None) -> None:
        """Undo a failed promotion (put the old pointer back, or remove it)."""
        path = self._reader_dir(reader) / "current.json"
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(previous.model_dump_json(indent=2), "utf-8")

    def resolve(self, reader: str, version: str) -> str:
        """Turn ``current`` into a concrete version id."""
        if version != CURRENT:
            self.get(reader, version)
            return version
        pointer = self.current(reader)
        if pointer is None:
            raise LearnError("no promoted version for this reader")
        return pointer.version


def apply_registry(config: ReadersConfig, registry: ModelRegistry) -> ReadersConfig:
    """Return a copy whose ``registry_version`` readers use registry weights.

    ``model`` becomes the weights path and ``version`` the registry version, so
    a calibrator fitted for the old (name, version) set reads as stale. A
    disabled reader that cannot resolve is left untouched; an enabled one fails
    loudly rather than silently falling back to the base weights.
    """
    readers = []
    for spec in config.readers:
        if spec.registry_version is None:
            readers.append(spec)
            continue
        try:
            version = registry.resolve(spec.name, spec.registry_version)
            if not registry.verify(spec.name, version):
                raise LearnError("registry weights failed hash check")
            path = registry.weights_path(spec.name, version)
        except LearnError:
            if spec.enabled:
                raise
            readers.append(spec)
            continue
        readers.append(spec.model_copy(update={"model": str(path), "version": version}))
    return config.model_copy(update={"readers": readers})
