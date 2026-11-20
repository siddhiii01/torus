"""Resolve ``TORUS_DATA_DIR`` and test-set directory paths (P1.1)."""

from __future__ import annotations

import os
from pathlib import Path

from torus.core.errors import EvalError

_ENV_DATA_DIR = "TORUS_DATA_DIR"
_TESTSETS_DIRNAME = "testsets"

SOURCE_FILENAME = "source.pdf"
EXPECTED_FILENAME = "expected.json"
META_FILENAME = "meta.json"
PAGES_FILENAME = "pages.json"
LOCATIONS_FILENAME = "locations.json"

CASE_FILENAMES: frozenset[str] = frozenset(
    {SOURCE_FILENAME, EXPECTED_FILENAME, META_FILENAME}
)
# Optional gold labels (P3.5 page types, P4.8 field locations).
OPTIONAL_CASE_FILENAMES: frozenset[str] = frozenset(
    {PAGES_FILENAME, LOCATIONS_FILENAME}
)


def resolve_data_dir(*, env: dict[str, str] | None = None) -> Path:
    """Return ``TORUS_DATA_DIR`` as an absolute path.

    Raises ``EvalError`` when the variable is unset or empty.
    Does not require the directory to exist (callers may create it).
    """
    mapping = os.environ if env is None else env
    raw = mapping.get(_ENV_DATA_DIR, "").strip()
    if not raw:
        raise EvalError(
            f"{_ENV_DATA_DIR} is not set; real test sets live outside the repo "
            f"(see docs/SETUP.md and docs/TESTSET_FORMAT.md)"
        )
    return Path(raw).expanduser().resolve()


def get_testsets_root(data_dir: Path | None = None) -> Path:
    """Return ``<data_dir>/testsets``.

    Named without a ``test_`` / ``testsets_`` prefix so pytest does not collect
    it when imported into test modules.
    """
    root = data_dir if data_dir is not None else resolve_data_dir()
    return root / _TESTSETS_DIRNAME


def set_dir(set_id: str, *, data_dir: Path | None = None) -> Path:
    """Return ``<data_dir>/testsets/<set_id>``."""
    _require_path_segment(set_id, what="set id")
    return get_testsets_root(data_dir) / set_id


def case_dir(set_id: str, case_id: str, *, data_dir: Path | None = None) -> Path:
    """Return ``<data_dir>/testsets/<set_id>/<case_id>``."""
    _require_path_segment(case_id, what="case id")
    return set_dir(set_id, data_dir=data_dir) / case_id


def _require_path_segment(value: str, *, what: str) -> None:
    stripped = value.strip()
    if not stripped or stripped != value or "/" in value or "\\" in value:
        raise EvalError(f"invalid {what}: {value!r}")
    if value in {".", ".."}:
        raise EvalError(f"invalid {what}: {value!r}")
