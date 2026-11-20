"""Model-weight presence checks under ``TORUS_MODEL_DIR``."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_MODEL_DIR = "TORUS_MODEL_DIR"


def model_dir() -> Path | None:
    """Return ``TORUS_MODEL_DIR`` if set, else None."""
    raw = os.environ.get(_ENV_MODEL_DIR)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def weights_present(model: str | None, root: Path | None = None) -> bool:
    """True when the reader needs no weights, or a matching path exists.

    ``model`` may be a relative path under the model dir, an absolute path, or
    a simple name (matched as a directory or file under the model dir).
    """
    if model is None or model.strip() == "":
        return True
    model = model.strip()
    candidate = Path(model)
    if candidate.is_absolute():
        return candidate.exists()

    base = root if root is not None else model_dir()
    if base is None:
        return False
    return (base / model).exists()
