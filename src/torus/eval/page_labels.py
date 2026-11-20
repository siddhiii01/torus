"""Load gold and predicted page-type labels for classification eval (P3.5)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from torus.core.errors import EvalError
from torus.eval.paths import PAGES_FILENAME
from torus.eval.schema import PageLabelsDocument


def load_page_labels_document(
    path: Path,
    *,
    case_id: str | None = None,
) -> PageLabelsDocument:
    """Parse a ``pages.json``-shaped file into :class:`PageLabelsDocument`."""
    if not path.is_file():
        raise EvalError(f"missing page labels file: {path.name}")
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
        doc = PageLabelsDocument.model_validate(data)
    except ValidationError as exc:
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            parts.append(f"{loc}: {msg}" if loc else str(msg))
        summary = "; ".join(parts)
        raise EvalError(f"invalid page labels in {path.name}: {summary}") from exc
    if case_id is not None and doc.case_id is not None and doc.case_id != case_id:
        raise EvalError(
            f"page labels case_id {doc.case_id!r} does not match {case_id!r}"
        )
    return doc


def load_gold_page_labels(
    case_path: Path, *, case_id: str
) -> PageLabelsDocument | None:
    """Load optional gold ``pages.json`` from a case folder; ``None`` if absent."""
    path = case_path / PAGES_FILENAME
    if not path.is_file():
        return None
    return load_page_labels_document(path, case_id=case_id)


def resolve_page_prediction_path(
    pred_dir: Path,
    case_id: str,
    *,
    variant: str | None = None,
) -> Path | None:
    """Return the page-type prediction file for *case_id*, or ``None``.

    Search order:

    1. ``{pred_dir}/{case_id}_{variant}.pages.json`` when *variant* is set
    2. ``{pred_dir}/{case_id}.pages.json``
    3. ``{pred_dir}/{case_id}/pages.json``
    4. ``{pred_dir}/{case_id}_pages.json``
    """
    if variant:
        candidate = pred_dir / f"{case_id}_{variant}.pages.json"
        if candidate.is_file():
            return candidate
    direct = pred_dir / f"{case_id}.pages.json"
    if direct.is_file():
        return direct
    nested = pred_dir / case_id / PAGES_FILENAME
    if nested.is_file():
        return nested
    alt = pred_dir / f"{case_id}_pages.json"
    if alt.is_file():
        return alt
    return None


def load_pred_page_labels(
    pred_dir: Path,
    case_id: str,
    *,
    variant: str | None = None,
) -> PageLabelsDocument | None:
    """Load predicted page types for *case_id* from *pred_dir*, or ``None``."""
    path = resolve_page_prediction_path(pred_dir, case_id, variant=variant)
    if path is None:
        return None
    return load_page_labels_document(path, case_id=case_id)
