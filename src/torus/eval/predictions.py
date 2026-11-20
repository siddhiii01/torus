"""Load Excel-shaped prediction JSON for evaluation (legacy-compatible).

Prediction files are produced by the legacy extract scripts or by a future
pipeline writer. Layout matches ``scripts/deb_extract.py`` output:

* ``{"rows": [ {column: value, ...}, ... ], ...}`` or
* a bare JSON list of row objects.

No field values are logged; errors refer to paths and case ids only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torus.core.errors import EvalError
from torus.eval.align import RowDict
from torus.eval.schema import ExpectedCell


def resolve_prediction_path(
    pred_dir: Path,
    case_id: str,
    *,
    variant: str | None = None,
) -> Path | None:
    """Return the prediction file for *case_id*, or ``None`` if missing.

    Search order:

    1. ``{pred_dir}/{case_id}_{variant}.json`` when *variant* is set
    2. ``{pred_dir}/{case_id}.json``
    3. ``{pred_dir}/{case_id}/pred.json``
    """
    if variant:
        candidate = pred_dir / f"{case_id}_{variant}.json"
        if candidate.is_file():
            return candidate
    direct = pred_dir / f"{case_id}.json"
    if direct.is_file():
        return direct
    nested = pred_dir / case_id / "pred.json"
    if nested.is_file():
        return nested
    return None


def _cell_as_expected(value: Any) -> ExpectedCell:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Legacy extractors occasionally emit nested structures; refuse rather than
    # silently stringify (that would invent a match rule).
    raise EvalError(
        f"prediction cell must be a JSON scalar, got {type(value).__name__}"
    )


def parse_prediction_payload(data: object, *, case_id: str) -> list[RowDict]:
    """Normalise a loaded JSON payload into a list of row dicts."""
    if isinstance(data, list):
        raw_rows = data
    elif isinstance(data, dict):
        if "rows" not in data:
            raise EvalError(
                f"prediction for case {case_id} must contain a 'rows' list "
                f"or be a JSON array"
            )
        raw_rows = data["rows"]
    else:
        raise EvalError(f"prediction for case {case_id} must be a JSON object or array")

    if not isinstance(raw_rows, list):
        raise EvalError(f"prediction rows for case {case_id} must be a list")

    rows: list[RowDict] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise EvalError(
                f"prediction rows[{index}] for case {case_id} must be an object"
            )
        cleaned: dict[str, ExpectedCell] = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key.strip():
                raise EvalError(
                    f"prediction rows[{index}] for case {case_id} has an "
                    f"invalid column key"
                )
            if key.startswith("_"):
                continue
            try:
                cleaned[key.strip()] = _cell_as_expected(value)
            except EvalError as exc:
                raise EvalError(
                    f"prediction rows[{index}][{key!r}] for case {case_id}: {exc}"
                ) from exc
        rows.append(cleaned)
    return rows


def load_prediction_rows(path: Path, *, case_id: str) -> list[RowDict]:
    """Read prediction rows from *path* (metadata-only errors)."""
    if not path.is_file():
        raise EvalError(f"prediction file missing for case {case_id}: {path.name}")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalError(
            f"invalid JSON in prediction for case {case_id} ({path.name}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise EvalError(
            f"cannot read prediction for case {case_id} ({path.name}): {exc.strerror}"
        ) from exc
    return parse_prediction_payload(data, case_id=case_id)
