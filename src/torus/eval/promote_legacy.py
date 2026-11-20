"""Promote legacy ``gold.json`` cases into the P1.1 test-set layout.

Writes only under ``TORUS_DATA_DIR`` (never into the git tree). Copies
``source.pdf`` when present; builds ``expected.json`` + ``meta.json`` from
legacy gold without carrying customer display names into meta notes.

Legacy layout::

    $TORUS_DATA_DIR/legacy/deb/cases/<case_id>/
      source.pdf
      gold.json
      meta.json

Target layout::

    $TORUS_DATA_DIR/testsets/<set_id>/<case_id>/
      source.pdf
      expected.json
      meta.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from torus.core.errors import EvalError
from torus.eval.paths import (
    EXPECTED_FILENAME,
    META_FILENAME,
    SOURCE_FILENAME,
    case_dir,
    resolve_data_dir,
)
from torus.eval.schema import (
    CaseMeta,
    CaseTags,
    ExpectedDocument,
    HandwritingLevel,
    ScanType,
    Split,
)
from torus.eval.validate import load_json_object, validate_case_dir

_DEFAULT_LEGACY_REL = Path("legacy") / "deb" / "cases"
_DEFAULT_TEMPLATE_ID = "deb_v1"
_DEFAULT_SET_ID = "deb_dev"


@dataclass(frozen=True, slots=True)
class PromoteResult:
    """Outcome of promoting one legacy case."""

    case_id: str
    dest: Path
    created: bool


def default_legacy_cases_dir(data_dir: Path | None = None) -> Path:
    """Return ``$TORUS_DATA_DIR/legacy/deb/cases``."""
    root = data_dir if data_dir is not None else resolve_data_dir()
    return root / _DEFAULT_LEGACY_REL


def _load_gold_rows(gold_path: Path, *, case_id: str) -> list[dict[str, object]]:
    data = load_json_object(gold_path)
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise EvalError(f"legacy gold for case {case_id} must contain a rows list")
    out: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvalError(
                f"legacy gold rows[{index}] for case {case_id} must be an object"
            )
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise EvalError(
                    f"legacy gold rows[{index}] for case {case_id} has a non-string key"
                )
            if key.startswith("_"):
                continue
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise EvalError(
                    f"legacy gold rows[{index}][{key!r}] for case {case_id} "
                    f"must be a JSON scalar"
                )
            cleaned[key] = value
        out.append(cleaned)
    return out


def _page_count_from_legacy_meta(meta_path: Path) -> int | None:
    if not meta_path.is_file():
        return None
    data = load_json_object(meta_path)
    raw = data.get("page_count_expected")
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def promote_legacy_case(
    legacy_case_path: Path,
    *,
    set_id: str = _DEFAULT_SET_ID,
    template_id: str = _DEFAULT_TEMPLATE_ID,
    split: Split = Split.DEV,
    data_dir: Path | None = None,
    force: bool = False,
    scan_type: ScanType | None = ScanType.SCAN,
    handwriting_level: HandwritingLevel | None = HandwritingLevel.MEDIUM,
    year: int | None = None,
) -> PromoteResult:
    """Copy one legacy case into ``testsets/<set_id>/<case_id>/``.

    Does not embed customer display names from legacy meta into the new
    ``meta.json`` (only structural tags and page counts).
    """
    case_id = legacy_case_path.name
    if not legacy_case_path.is_dir():
        raise EvalError(f"legacy case directory not found: {case_id}")

    gold_path = legacy_case_path / "gold.json"
    source_path = legacy_case_path / SOURCE_FILENAME
    if not gold_path.is_file():
        raise EvalError(f"legacy gold.json missing for case {case_id}")
    if not source_path.is_file():
        raise EvalError(f"legacy {SOURCE_FILENAME} missing for case {case_id}")

    dest = case_dir(set_id, case_id, data_dir=data_dir)
    already = (
        (dest / EXPECTED_FILENAME).is_file()
        and (dest / META_FILENAME).is_file()
        and (dest / SOURCE_FILENAME).is_file()
    )
    if already and not force:
        return PromoteResult(case_id=case_id, dest=dest, created=False)

    rows = _load_gold_rows(gold_path, case_id=case_id)
    page_count = _page_count_from_legacy_meta(legacy_case_path / "meta.json")

    expected = ExpectedDocument(
        template_id=template_id,
        case_id=case_id,
        version=1,
        rows=rows,  # type: ignore[arg-type]
    )
    tags = CaseTags(
        year=year,
        scan_type=scan_type,
        handwriting_level=handwriting_level,
    )
    meta = CaseMeta(
        case_id=case_id,
        split=split,
        template_id=template_id,
        tags=tags,
        page_count_expected=page_count,
        notes="promoted from legacy gold (P1.6 baseline)",
    )

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest / SOURCE_FILENAME)
    (dest / EXPECTED_FILENAME).write_text(
        json.dumps(expected.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    (dest / META_FILENAME).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    validate_case_dir(dest, set_id=set_id, require_source=True)
    return PromoteResult(case_id=case_id, dest=dest, created=True)


def promote_legacy_set(
    *,
    legacy_cases_dir: Path | None = None,
    set_id: str = _DEFAULT_SET_ID,
    template_id: str = _DEFAULT_TEMPLATE_ID,
    split: Split = Split.DEV,
    data_dir: Path | None = None,
    force: bool = False,
) -> list[PromoteResult]:
    """Promote every subdirectory under the legacy cases folder."""
    root = (
        legacy_cases_dir
        if legacy_cases_dir is not None
        else default_legacy_cases_dir(data_dir)
    )
    if not root.is_dir():
        raise EvalError(f"legacy cases directory not found: {root}")

    results: list[PromoteResult] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "gold.json").is_file():
            continue
        results.append(
            promote_legacy_case(
                child,
                set_id=set_id,
                template_id=template_id,
                split=split,
                data_dir=data_dir,
                force=force,
            )
        )
    if not results:
        raise EvalError(f"no legacy gold cases found under {root}")
    return results
