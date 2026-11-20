from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import pytest

from torus.core.errors import ConfigError
from torus.service import bundle
from torus.service.errors import ServiceError

REPO_CONFIG = Path("config")


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    dest = tmp_path / "cfg"
    shutil.copytree(REPO_CONFIG, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _make(cfg: Path, tmp_path: Path, name: str) -> tuple[Path, str]:
    out = tmp_path / name
    return out, bundle.create_bundle(cfg, out)


def test_create_is_reproducible_and_content_addressed(
    cfg: Path, tmp_path: Path
) -> None:
    _, first = _make(cfg, tmp_path, "a.tgz")
    _, second = _make(cfg, tmp_path, "b.tgz")
    assert first == second
    (cfg / "service.yaml").write_text(
        (cfg / "service.yaml").read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    _, third = _make(cfg, tmp_path, "c.tgz")
    assert third != first


def test_create_refuses_invalid_config(cfg: Path, tmp_path: Path) -> None:
    (cfg / "service.yaml").write_text("default_template_id: x\nbogus: 1\n")
    with pytest.raises(ConfigError):
        bundle.create_bundle(cfg, tmp_path / "bad.tgz")


def test_install_activate_rollback(cfg: Path, tmp_path: Path) -> None:
    root = tmp_path / "root"
    a_arc, a = _make(cfg, tmp_path, "a.tgz")
    (cfg / "service.yaml").write_text(
        (cfg / "service.yaml").read_text(encoding="utf-8") + "\n# v2\n",
        encoding="utf-8",
    )
    b_arc, b = _make(cfg, tmp_path, "b.tgz")

    assert bundle.install_bundle(a_arc, root) == a
    assert bundle.install_bundle(a_arc, root) == a  # idempotent
    bundle.activate(root, a)
    assert bundle.install_bundle(b_arc, root) == b
    assert bundle.current_path(root).name == a  # installing does not activate
    bundle.activate(root, b)
    assert bundle.current_path(root).name == b
    assert bundle.rollback(root) == a
    assert bundle.current_path(root).name == a
    with pytest.raises(ServiceError, match="no earlier"):
        bundle.rollback(root)
    # An installed bundle is a complete, loadable config.
    bundle.validate_config_dir(bundle.current_path(root))


def test_activate_requires_installed(tmp_path: Path) -> None:
    with pytest.raises(ServiceError):
        bundle.activate(tmp_path, "a" * 32)
    with pytest.raises(ServiceError):
        bundle.activate(tmp_path, "../etc")


def _rewrite(src: Path, dest: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    with tarfile.open(src) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}  # type: ignore[union-attr]
    members = mutate(members)
    with tarfile.open(dest, "w:gz") as out:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))


def test_install_detects_tampering(cfg: Path, tmp_path: Path) -> None:
    arc, _ = _make(cfg, tmp_path, "a.tgz")
    bad = tmp_path / "bad.tgz"
    _rewrite(arc, bad, lambda m: {**m, "config/service.yaml": b"x: 1\n"})
    with pytest.raises(ServiceError, match="manifest"):
        bundle.install_bundle(bad, tmp_path / "root")
    assert not (tmp_path / "root" / "bundles").exists() or not any(
        (tmp_path / "root" / "bundles").iterdir()
    )


def test_install_rejects_path_traversal(cfg: Path, tmp_path: Path) -> None:
    arc, _ = _make(cfg, tmp_path, "a.tgz")
    bad = tmp_path / "bad.tgz"
    _rewrite(arc, bad, lambda m: {**m, "config/../../evil": b"x"})
    with pytest.raises(ServiceError, match="unsafe"):
        bundle.install_bundle(bad, tmp_path / "root")


def test_install_rejects_garbage(tmp_path: Path) -> None:
    junk = tmp_path / "junk.tgz"
    junk.write_bytes(b"not a tarball")
    with pytest.raises(ServiceError):
        bundle.install_bundle(junk, tmp_path / "root")
