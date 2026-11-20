"""Load and validate evaluation case folders (P1.1)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from torus.core.errors import EvalError
from torus.eval.locations import load_gold_locations
from torus.eval.page_labels import load_gold_page_labels
from torus.eval.paths import (
    EXPECTED_FILENAME,
    META_FILENAME,
    SOURCE_FILENAME,
    get_testsets_root,
    set_dir,
)
from torus.eval.schema import (
    CaseMeta,
    ExpectedDocument,
    LocationsDocument,
    PageLabelsDocument,
    Split,
)


@dataclass(frozen=True, slots=True)
class ValidatedCase:
    """A case directory that passed layout and schema checks."""

    path: Path
    set_id: str
    case_id: str
    meta: CaseMeta
    expected: ExpectedDocument
    source_path: Path
    pages: PageLabelsDocument | None = None
    locations: LocationsDocument | None = None


def load_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object from *path*; never logs file contents on failure."""
    if not path.is_file():
        raise EvalError(f"missing file: {path.name} (under case {path.parent.name})")
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(
            f"invalid JSON in {path.name} (case {path.parent.name}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise EvalError(
            f"cannot read {path.name} (case {path.parent.name}): {exc.strerror}"
        ) from exc
    if not isinstance(data, dict):
        raise EvalError(f"{path.name} (case {path.parent.name}) must be a JSON object")
    return data


def _parse_meta(data: dict[str, object], *, case_folder: str) -> CaseMeta:
    try:
        return CaseMeta.model_validate(data)
    except ValidationError as exc:
        summary = _summarize_pydantic(exc)
        raise EvalError(
            f"invalid {META_FILENAME} for case {case_folder}: {summary}"
        ) from exc


def _parse_expected(data: dict[str, object], *, case_folder: str) -> ExpectedDocument:
    try:
        return ExpectedDocument.model_validate(data)
    except ValidationError as exc:
        raise EvalError(
            f"invalid {EXPECTED_FILENAME} for case {case_folder}: "
            f"{_summarize_pydantic(exc)}"
        ) from exc


def _summarize_pydantic(exc: ValidationError) -> str:
    """Compact error summary without dumping document field values."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) if parts else "validation failed"


def validate_case_dir(
    path: Path,
    *,
    set_id: str | None = None,
    template_keys: frozenset[str] | Sequence[str] | None = None,
    require_source: bool = True,
) -> ValidatedCase:
    """Validate one case folder: required files, schemas, and id consistency.

    Parameters
    ----------
    path:
        Directory containing ``source.pdf``, ``expected.json``, ``meta.json``.
    set_id:
        Optional set name (parent folder). When omitted, uses ``path.parent.name``.
    template_keys:
        When provided, every column key in ``expected.json`` rows must be in this
        set (keys from the template config). Unknown keys fail loudly.
    require_source:
        When True (default), ``source.pdf`` must exist and be non-empty.
    """
    case_path = path.resolve()
    if not case_path.is_dir():
        raise EvalError(f"case path is not a directory: {case_path.name}")

    case_id_from_folder = case_path.name
    resolved_set = set_id if set_id is not None else case_path.parent.name

    meta_path = case_path / META_FILENAME
    expected_path = case_path / EXPECTED_FILENAME
    source_path = case_path / SOURCE_FILENAME

    meta = _parse_meta(load_json_object(meta_path), case_folder=case_id_from_folder)
    expected = _parse_expected(
        load_json_object(expected_path), case_folder=case_id_from_folder
    )

    if meta.case_id != case_id_from_folder:
        raise EvalError(
            f"meta.case_id {meta.case_id!r} does not match folder name "
            f"{case_id_from_folder!r}"
        )
    if expected.case_id is not None and expected.case_id != case_id_from_folder:
        raise EvalError(
            f"expected.case_id {expected.case_id!r} does not match folder name "
            f"{case_id_from_folder!r}"
        )
    if expected.template_id != meta.template_id:
        raise EvalError(
            f"template_id mismatch for case {case_id_from_folder}: "
            f"meta={meta.template_id!r} expected={expected.template_id!r}"
        )

    if require_source:
        if not source_path.is_file():
            raise EvalError(f"missing {SOURCE_FILENAME} for case {case_id_from_folder}")
        if source_path.stat().st_size == 0:
            raise EvalError(f"empty {SOURCE_FILENAME} for case {case_id_from_folder}")

    if template_keys is not None:
        allowed = frozenset(template_keys)
        unknown = sorted(expected.column_keys() - allowed)
        if unknown:
            # Report keys only — not cell values.
            raise EvalError(
                f"case {case_id_from_folder}: unknown column keys for template "
                f"{meta.template_id!r}: {', '.join(unknown)}"
            )

    pages = load_gold_page_labels(case_path, case_id=case_id_from_folder)
    locations = load_gold_locations(case_path, case_id=case_id_from_folder)
    if locations is not None and template_keys is not None:
        allowed = frozenset(template_keys)
        unknown = sorted(locations.column_keys() - allowed)
        if unknown:
            raise EvalError(
                f"case {case_id_from_folder}: unknown location column keys for "
                f"template {meta.template_id!r}: {', '.join(unknown)}"
            )

    return ValidatedCase(
        path=case_path,
        set_id=resolved_set,
        case_id=case_id_from_folder,
        meta=meta,
        expected=expected,
        source_path=source_path,
        pages=pages,
        locations=locations,
    )


def iter_case_dirs(set_path: Path) -> list[Path]:
    """Return sorted child directories that look like cases (contain ``meta.json``)."""
    if not set_path.is_dir():
        raise EvalError(f"test set path is not a directory: {set_path.name}")
    cases = [
        child
        for child in sorted(set_path.iterdir())
        if child.is_dir() and (child / META_FILENAME).is_file()
    ]
    return cases


def validate_testset(
    set_id: str,
    *,
    data_dir: Path | None = None,
    set_path: Path | None = None,
    split: Split | None = None,
    template_keys: frozenset[str] | Sequence[str] | None = None,
    require_source: bool = True,
) -> list[ValidatedCase]:
    """Validate every case under a named set (or an explicit *set_path*)."""
    root = set_path if set_path is not None else set_dir(set_id, data_dir=data_dir)
    validated: list[ValidatedCase] = []
    for case_path in iter_case_dirs(root):
        case = validate_case_dir(
            case_path,
            set_id=set_id,
            template_keys=template_keys,
            require_source=require_source,
        )
        if split is not None and case.meta.split != split:
            continue
        validated.append(case)
    return validated


def list_set_ids(*, data_dir: Path | None = None) -> list[str]:
    """Return sorted set directory names under ``testsets/``."""
    root = get_testsets_root(data_dir)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def assert_no_duplicate_case_ids(cases: Iterable[ValidatedCase]) -> None:
    """Fail if the same ``case_id`` appears more than once in *cases*."""
    seen: dict[str, str] = {}
    for case in cases:
        prior = seen.get(case.case_id)
        if prior is not None:
            raise EvalError(
                f"duplicate case_id {case.case_id!r} in sets {prior!r} and "
                f"{case.set_id!r}"
            )
        seen[case.case_id] = case.set_id
