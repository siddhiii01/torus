"""Load gold and predicted field locations for layout eval (P4.8)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from torus.core.errors import EvalError
from torus.eval.paths import LOCATIONS_FILENAME
from torus.eval.schema import LocationsDocument


def load_locations_document(
    path: Path,
    *,
    case_id: str | None = None,
) -> LocationsDocument:
    """Parse a ``locations.json``-shaped file into :class:`LocationsDocument`."""
    if not path.is_file():
        raise EvalError(f"missing locations file: {path.name}")
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid JSON in {path.name}: {exc.msg}") from exc
    except OSError as exc:
        raise EvalError(f"cannot read {path.name}: {exc.strerror}") from exc
    if not isinstance(data, dict):
        raise EvalError(f"{path.name} must be a JSON object")
    try:
        doc = LocationsDocument.model_validate(data)
    except ValidationError as exc:
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            parts.append(f"{loc}: {msg}" if loc else str(msg))
        summary = "; ".join(parts)
        raise EvalError(f"invalid locations in {path.name}: {summary}") from exc
    if case_id is not None and doc.case_id is not None and doc.case_id != case_id:
        raise EvalError(f"locations case_id {doc.case_id!r} does not match {case_id!r}")
    return doc


def load_gold_locations(case_path: Path, *, case_id: str) -> LocationsDocument | None:
    """Load optional gold ``locations.json`` from a case folder; ``None`` if absent."""
    path = case_path / LOCATIONS_FILENAME
    if not path.is_file():
        return None
    return load_locations_document(path, case_id=case_id)


def resolve_location_prediction_path(
    pred_dir: Path,
    case_id: str,
    *,
    variant: str | None = None,
) -> Path | None:
    """Return the field-location prediction file for *case_id*, or ``None``.

    Search order:

    1. ``{pred_dir}/{case_id}_{variant}.locations.json`` when *variant* is set
    2. ``{pred_dir}/{case_id}.locations.json``
    3. ``{pred_dir}/{case_id}/locations.json``
    4. ``{pred_dir}/{case_id}_locations.json``
    """
    if variant:
        candidate = pred_dir / f"{case_id}_{variant}.locations.json"
        if candidate.is_file():
            return candidate
    direct = pred_dir / f"{case_id}.locations.json"
    if direct.is_file():
        return direct
    nested = pred_dir / case_id / LOCATIONS_FILENAME
    if nested.is_file():
        return nested
    alt = pred_dir / f"{case_id}_locations.json"
    if alt.is_file():
        return alt
    return None


def load_pred_locations(
    pred_dir: Path,
    case_id: str,
    *,
    variant: str | None = None,
) -> LocationsDocument | None:
    """Load predicted field locations for *case_id* from *pred_dir*, or ``None``."""
    path = resolve_location_prediction_path(pred_dir, case_id, variant=variant)
    if path is None:
        return None
    return load_locations_document(path, case_id=case_id)
