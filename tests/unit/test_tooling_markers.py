"""Smoke tests that pytest markers from P0.2 are registered."""

from __future__ import annotations

import pytest


def test_model_and_slow_markers_registered(pytestconfig: pytest.Config) -> None:
    raw = pytestconfig.getini("markers")
    names = {entry.split(":", 1)[0].strip() for entry in raw}
    assert "model" in names
    assert "slow" in names


@pytest.mark.model
def test_model_marker_skippable() -> None:
    """Exists so CI can prove `-m 'not model'` excludes model tests."""
    assert True


@pytest.mark.slow
def test_slow_marker_runs_in_default_suite() -> None:
    """`slow` is registered; default CI still runs it (only `model` is excluded)."""
    assert True
