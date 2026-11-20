"""Config bundles: one versioned, verifiable unit of deployment config (P11.6).

A bundle is a ``.tar.gz`` of a config directory plus a ``manifest.json`` with a
SHA-256 per file. Its id is the hash of those hashes, so the same content always
gives the same id. Installing verifies every file and validates the config
before it can be activated. Activation and rollback only move a pointer file, so
both are atomic and instant; old bundles stay on disk until removed.

Layout under the bundle root (for example ``/var/lib/torus/bundles``)::

    bundles/<bundle_id>/...config files...
    state.json      {"current": "<id>", "history": ["<older id>", ...]}
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Final

from torus.config import load_config
from torus.core.errors import ConfigError
from torus.service.errors import ServiceError
from torus.service.settings import load_service_config

MANIFEST: Final[str] = "manifest.json"
_ID = re.compile(r"^[0-9a-f]{16,64}$")
_MAX_FILE_BYTES: Final[int] = 64 * 1024 * 1024
_SKIP_DIRS: Final[frozenset[str]] = frozenset({"__pycache__"})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect(config_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(config_dir.rglob("*")):
        rel = path.relative_to(config_dir)
        if not path.is_file() or _SKIP_DIRS & set(rel.parts):
            continue
        files[rel.as_posix()] = path.read_bytes()
    return files


def bundle_id_for(hashes: dict[str, str]) -> str:
    payload = json.dumps(sorted(hashes.items())).encode()
    return _sha256(payload)[:32]


def validate_config_dir(config_dir: Path) -> None:
    """Fail loudly if the directory is not a complete, valid configuration."""
    load_config(config_dir)
    load_service_config(config_dir / "service.yaml")


def create_bundle(config_dir: Path, out: Path) -> str:
    """Validate ``config_dir`` and write the bundle archive; return its id."""
    validate_config_dir(config_dir)
    files = _collect(config_dir)
    hashes = {name: _sha256(data) for name, data in files.items()}
    bundle_id = bundle_id_for(hashes)
    manifest = {
        "format": 1,
        "bundle_id": bundle_id,
        "created_at": time.time(),
        "files": hashes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        _add(tar, MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode())
        for name, data in files.items():
            _add(tar, f"config/{name}", data)
    return bundle_id


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # reproducible archives
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _read_archive(archive: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read a bundle in memory, refusing anything but plain files."""
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > _MAX_FILE_BYTES:
                raise ServiceError("bundle contains a non-regular or oversized entry")
            parts = Path(member.name).parts
            if member.name.startswith(("/", "\\")) or ".." in parts:
                raise ServiceError("bundle contains an unsafe path")
            handle = tar.extractfile(member)
            if handle is None:
                raise ServiceError("unreadable bundle entry")
            members[member.name] = handle.read()
    if MANIFEST not in members:
        raise ServiceError("bundle has no manifest")
    manifest = json.loads(members.pop(MANIFEST))
    return manifest, members


def install_bundle(archive: Path, root: Path) -> str:
    """Verify, validate and unpack a bundle. Does not activate it."""
    try:
        manifest, members = _read_archive(archive)
    except (tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise ServiceError("bundle is not readable") from exc
    expected: dict[str, str] = manifest.get("files", {})
    actual = {
        name.removeprefix("config/"): _sha256(data)
        for name, data in members.items()
        if name.startswith("config/")
    }
    if actual != expected:
        raise ServiceError("bundle contents do not match its manifest")
    bundle_id = bundle_id_for(expected)
    if manifest.get("bundle_id") != bundle_id or not _ID.match(bundle_id):
        raise ServiceError("bundle id does not match its contents")
    target = root / "bundles" / bundle_id
    if target.exists():
        return bundle_id
    root.joinpath("bundles").mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root / "bundles", prefix=".stage-"))
    try:
        for name, data in members.items():
            dest = staging / name.removeprefix("config/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        try:
            validate_config_dir(staging)
        except ConfigError as exc:
            raise ServiceError("bundle config failed validation") from exc
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return bundle_id


# -- activation pointer -------------------------------------------------------


def _state_path(root: Path) -> Path:
    return root / "state.json"


def read_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.is_file():
        return {"current": None, "history": []}
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return state


def _write_state(root: Path, state: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(root).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, _state_path(root))  # atomic on POSIX and Windows


def activate(root: Path, bundle_id: str) -> None:
    if not _ID.match(bundle_id) or not (root / "bundles" / bundle_id).is_dir():
        raise ServiceError("bundle is not installed")
    state = read_state(root)
    if state["current"] == bundle_id:
        return
    history = [h for h in state["history"] if h != bundle_id]
    if state["current"]:
        history.append(state["current"])
    _write_state(root, {"current": bundle_id, "history": history})


def rollback(root: Path) -> str:
    """Re-activate the previously active bundle; returns its id."""
    state = read_state(root)
    history: list[str] = list(state["history"])
    while history:
        previous = history.pop()
        if (root / "bundles" / previous).is_dir():
            _write_state(root, {"current": previous, "history": history})
            return previous
    raise ServiceError("no earlier bundle to roll back to")


def current_path(root: Path) -> Path:
    current = read_state(root)["current"]
    if not current:
        raise ServiceError("no bundle is active")
    return root / "bundles" / str(current)
