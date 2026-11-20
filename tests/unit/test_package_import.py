"""Smoke tests for the installable package skeleton (P0.1)."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_torus_version() -> None:
    import torus

    assert torus.__version__ == "0.1.0"


def test_stage_packages_importable() -> None:
    packages = [
        "torus.core",
        "torus.config",
        "torus.ingest",
        "torus.preprocess",
        "torus.classify",
        "torus.layout",
        "torus.readers",
        "torus.consensus",
        "torus.validate",
        "torus.mapping",
        "torus.output",
        "torus.review",
        "torus.learn",
        "torus.eval",
        "torus.service",
        "torus.logging",
        "torus.doctor",
        "torus.cli",
    ]
    for name in packages:
        importlib.import_module(name)


def test_layout_dirs_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = [
        root / "config" / "templates",
        root / "config" / "families",
        root / "resources" / "lookups",
        root / "tests" / "component",
        root / "tests" / "integration",
        root / "tests" / "readers",
        root / "tests" / "fixtures" / "synth",
        root / "legacy",
        root / "src" / "torus",
    ]
    missing = [str(path) for path in expected if not path.is_dir()]
    assert missing == []
