"""Local labelling tool: hand-fill expected.json under TORUS_DATA_DIR (P1.5).

No network calls. The source PDF is copied into the case folder only; nothing is
uploaded. Operators fill template columns by hand (CLI or optional Streamlit UI).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torus.config import TemplateConfig, default_config_dir, load_template
from torus.config.schemas import FieldSpec
from torus.core.errors import ConfigError, EvalError
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
    ExpectedCell,
    ExpectedDocument,
    HandwritingLevel,
    ScanType,
    Split,
)
from torus.eval.validate import validate_case_dir


# Sentinel: empty interactive input means "no gold for this column" (omit key).
class _OmitCell:
    __slots__ = ()


OMIT_CELL = _OmitCell()
ParsedCell = ExpectedCell | _OmitCell


@dataclass(frozen=True, slots=True)
class LabelCaseRequest:
    """Inputs for writing one labelled case folder."""

    pdf_path: Path
    set_id: str
    case_id: str
    template: TemplateConfig
    split: Split
    rows: Sequence[Mapping[str, ExpectedCell]]
    year: int | None = None
    scan_type: ScanType | None = None
    handwriting_level: HandwritingLevel | None = None
    page_count_expected: int | None = None
    notes: str = ""
    force: bool = False


@dataclass(frozen=True, slots=True)
class LabelledCaseResult:
    """Paths written for a labelled case (metadata only — no field values)."""

    path: Path
    set_id: str
    case_id: str
    template_id: str
    source_path: Path
    meta_path: Path
    expected_path: Path


def label_fields(template: TemplateConfig) -> list[FieldSpec]:
    """Columns operators should fill (excludes passthrough ops columns)."""
    return template.scored_fields()


def resolve_template(
    template_id: str, *, config_dir: Path | None = None
) -> TemplateConfig:
    """Load ``config/templates/<template_id>.yaml``."""
    root = config_dir if config_dir is not None else default_config_dir()
    path = root / "templates" / f"{template_id}.yaml"
    if not path.is_file():
        alt = root / "templates" / f"{template_id}.yml"
        if alt.is_file():
            path = alt
        else:
            raise ConfigError(f"template not found: {template_id!r} under {root}")
    return load_template(path)


def parse_cell_input(raw: str) -> ParsedCell:
    """Map a prompt string to a cell value.

    - blank → ``OMIT_CELL`` (key omitted from the row)
    - ``-`` → ``None`` (explicit empty gold)
    - otherwise → stripped string
    """
    text = raw.strip()
    if not text:
        return OMIT_CELL
    if text == "-":
        return None
    return text


def build_meta(
    *,
    case_id: str,
    split: Split,
    template_id: str,
    year: int | None = None,
    scan_type: ScanType | None = None,
    handwriting_level: HandwritingLevel | None = None,
    page_count_expected: int | None = None,
    notes: str = "",
) -> CaseMeta:
    """Build ``meta.json`` contents (metadata only)."""
    return CaseMeta(
        case_id=case_id,
        split=split,
        template_id=template_id,
        tags=CaseTags(
            year=year,
            scan_type=scan_type,
            handwriting_level=handwriting_level,
        ),
        page_count_expected=page_count_expected,
        notes=notes,
    )


def build_expected(
    *,
    template: TemplateConfig,
    case_id: str,
    rows: Sequence[Mapping[str, ExpectedCell]],
    version: int = 1,
) -> ExpectedDocument:
    """Build ``expected.json`` and reject unknown column keys."""
    allowed = frozenset(f.key for f in template.fields)
    cleaned: list[dict[str, ExpectedCell]] = []
    for index, row in enumerate(rows):
        out: dict[str, ExpectedCell] = {}
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if key not in allowed:
                raise EvalError(
                    f"unknown column key {key!r} for template {template.id!r} "
                    f"(row {index})"
                )
            out[key] = value
        cleaned.append(out)
    return ExpectedDocument(
        template_id=template.id,
        case_id=case_id,
        version=version,
        rows=cleaned,
    )


def open_pdf_locally(path: Path) -> None:
    """Open *path* in the OS default viewer. Local only — no upload."""
    pdf = path.expanduser().resolve()
    if not pdf.is_file():
        raise EvalError(f"PDF not found: {pdf.name}")
    if pdf.stat().st_size == 0:
        raise EvalError(f"empty PDF: {pdf.name}")
    try:
        if sys.platform == "win32":
            os.startfile(pdf)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(pdf)], check=False)
        else:
            subprocess.run(["xdg-open", str(pdf)], check=False)
    except OSError as exc:
        raise EvalError(f"could not open PDF {pdf.name}: {exc}") from exc


def _assert_under_data_dir(path: Path, data_dir: Path) -> None:
    try:
        path.resolve().relative_to(data_dir.resolve())
    except ValueError as exc:
        raise EvalError(
            "labelled cases must be written under TORUS_DATA_DIR "
            f"(refusing path outside {data_dir.name})"
        ) from exc


def write_labelled_case(
    request: LabelCaseRequest,
    *,
    data_dir: Path | None = None,
) -> LabelledCaseResult:
    """Copy the PDF and write ``meta.json`` + ``expected.json`` under the data dir."""
    root = data_dir if data_dir is not None else resolve_data_dir()
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pdf = request.pdf_path.expanduser().resolve()
    if not pdf.is_file():
        raise EvalError(f"PDF not found: {pdf.name}")
    if pdf.stat().st_size == 0:
        raise EvalError(f"empty PDF: {pdf.name}")

    dest = case_dir(request.set_id, request.case_id, data_dir=root)
    _assert_under_data_dir(dest, root)

    if dest.exists() and any(dest.iterdir()) and not request.force:
        raise EvalError(
            f"case folder already exists: {request.set_id}/{request.case_id} "
            "(pass force=True / --force to overwrite)"
        )

    meta = build_meta(
        case_id=request.case_id,
        split=request.split,
        template_id=request.template.id,
        year=request.year,
        scan_type=request.scan_type,
        handwriting_level=request.handwriting_level,
        page_count_expected=request.page_count_expected,
        notes=request.notes,
    )
    expected = build_expected(
        template=request.template,
        case_id=request.case_id,
        rows=request.rows,
    )

    dest.mkdir(parents=True, exist_ok=True)
    source_path = dest / SOURCE_FILENAME
    meta_path = dest / META_FILENAME
    expected_path = dest / EXPECTED_FILENAME

    shutil.copy2(pdf, source_path)
    _write_json(meta_path, meta.model_dump(mode="json"))
    _write_json(expected_path, expected.model_dump(mode="json"))

    validate_case_dir(
        dest,
        set_id=request.set_id,
        template_keys=frozenset(f.key for f in request.template.fields),
        require_source=True,
    )

    return LabelledCaseResult(
        path=dest.resolve(),
        set_id=request.set_id,
        case_id=request.case_id,
        template_id=request.template.id,
        source_path=source_path.resolve(),
        meta_path=meta_path.resolve(),
        expected_path=expected_path.resolve(),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prompt_rows_interactive(
    template: TemplateConfig,
    *,
    row_count: int,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> list[dict[str, ExpectedCell]]:
    """Prompt for *row_count* rows of template columns (CLI).

    Per cell: value | Enter = omit | ``-`` = explicit null.
    """
    if row_count < 1:
        raise EvalError("row_count must be >= 1")
    fields = label_fields(template)
    rows: list[dict[str, ExpectedCell]] = []
    for row_index in range(row_count):
        print_fn(f"Row {row_index + 1}/{row_count}")
        row: dict[str, ExpectedCell] = {}
        for spec in fields:
            hint = f"{spec.key} ({spec.excel_header})"
            if spec.critical:
                hint += " [critical]"
            raw = input_fn(f"  {hint}: ")
            parsed = parse_cell_input(str(raw))
            if isinstance(parsed, _OmitCell):
                continue
            row[spec.key] = parsed
        rows.append(row)
    return rows


def load_rows_json(path: Path) -> list[dict[str, ExpectedCell]]:
    """Load a list of row objects from a local JSON file (no network)."""
    if not path.is_file():
        raise EvalError(f"rows JSON not found: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid rows JSON ({path.name}): {exc.msg}") from exc
    if not isinstance(data, list):
        raise EvalError(f"rows JSON ({path.name}) must be a list of objects")
    rows: list[dict[str, ExpectedCell]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise EvalError(f"rows JSON [{index}] must be an object")
        out: dict[str, ExpectedCell] = {}
        for key, value in item.items():
            if not isinstance(key, str):
                raise EvalError(f"rows JSON [{index}] has a non-string key")
            if key.startswith("_"):
                continue
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise EvalError(
                    f"rows JSON [{index}][{key!r}] must be a JSON scalar or null"
                )
            out[key] = value
        rows.append(out)
    return rows
