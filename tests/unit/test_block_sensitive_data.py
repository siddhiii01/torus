"""Unit tests for tools/block_sensitive_data.py (P0.8 data safety)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "tools" / "block_sensitive_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("block_sensitive_data", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["block_sensitive_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guard():
    return _load_module()


@pytest.mark.parametrize(
    "path",
    [
        "data/deb/cases/case_a/source.pdf",
        "data/deb/cases/case_a/gold.json",
        "data/deb/cases/case_a/meta.json",
        "data/deb/raw/sheet.xlsx",
        "samples/scan.png",
        "tests/fixtures/real/pack.jpg",
        r"data\deb\cases\case_a\source.pdf",
    ],
)
def test_blocks_sensitive_under_data_paths(guard, path: str) -> None:
    assert guard.is_blocked_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "data/README.md",
        "docs/SETUP.md",
        "docs/plan.pdf",
        "config/pipeline.yaml",
        "tests/fixtures/synth/SOURCES.md",
        "src/torus/core/models.py",
        "tools/block_sensitive_data.py",
    ],
)
def test_allows_non_sensitive_paths(guard, path: str) -> None:
    assert guard.is_blocked_path(path) is False


def test_check_paths_returns_only_blocked(guard) -> None:
    paths = [
        "data/README.md",
        "data/deb/cases/x/source.pdf",
        "docs/SETUP.md",
        "samples/a.csv",
    ]
    assert guard.check_paths(paths) == [
        "data/deb/cases/x/source.pdf",
        "samples/a.csv",
    ]


def test_main_exits_nonzero_on_blocked(
    guard, capsys: pytest.CaptureFixture[str]
) -> None:
    code = guard.main(["data/deb/cases/x/gold.json"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Data-safety check failed" in err
    assert "gold.json" in err
    # Must not dump document content (file was never opened).
    assert "TORUS_DATA_DIR" in err


def test_main_exits_zero_when_clean(guard) -> None:
    assert guard.main(["data/README.md", "docs/SETUP.md"]) == 0
