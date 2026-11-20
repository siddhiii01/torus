"""Template import (P8.1): client Excel headers → ``config/templates/<id>.yaml``.

Only the header row of the workbook is retained; cell data is never read into
the result, stored, or logged. Existing field specs (descriptions, types,
validators, ...) are preserved: a sync only adds new columns, refreshes
``excel_col`` positions, drops columns the client removed, and optionally marks
columns ``passthrough`` (ops/system columns the engine must not fill).

Usage::

    python -m torus.output.template sync <client.xlsx> <template_id> \
        [--passthrough "Header"]... [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from torus.config.loader import default_config_dir
from torus.config.schemas import TemplateConfig
from torus.core.errors import ConfigError

_SCAN_ROWS = 10
_MAX_KEY_LEN = 48
_TEMPLATES_DIR = "templates"


@dataclass(frozen=True)
class HeaderCell:
    """One header cell: 1-based column and the header text exactly as in the file.

    Inner whitespace is kept so ``excel_header`` stays the exact client text;
    matching against existing specs uses a normalised key.
    """

    col: int
    text: str


@dataclass(frozen=True)
class SheetHeaders:
    sheet: str
    header_row: int
    cells: list[HeaderCell]


@dataclass
class SyncResult:
    """Outcome of a merge; ``data`` is the validated template mapping to write."""

    data: dict[str, Any]
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    passthrough_set: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.moved or self.passthrough_set)


def _norm_header(text: object) -> str:
    return " ".join(str(text).split())


def _header_match_key(text: str) -> str:
    return _norm_header(text).casefold()


def read_headers(
    xlsx: Path,
    *,
    sheet: str | None = None,
    header_row: int | None = None,
    scan_rows: int = _SCAN_ROWS,
) -> SheetHeaders:
    """Read the header row of one sheet (default: active sheet).

    With ``header_row`` unset, the row with the most non-blank text cells among
    the first ``scan_rows`` rows is taken (earliest wins ties). Rows beyond the
    header are never touched.
    """
    from openpyxl import load_workbook  # type: ignore[import-untyped]

    if not xlsx.is_file():
        raise ConfigError(f"template workbook not found: {xlsx}")
    try:
        wb = load_workbook(xlsx, read_only=True, data_only=True)
    except Exception as exc:  # openpyxl raises many types on a bad file
        raise ConfigError(
            f"cannot open workbook {xlsx.name}: {type(exc).__name__}"
        ) from exc
    try:
        if sheet is None:
            ws = wb.active
            if ws is None:
                raise ConfigError(f"workbook {xlsx.name} has no sheets")
        elif sheet in wb.sheetnames:
            ws = wb[sheet]
        else:
            raise ConfigError(
                f"sheet {sheet!r} not in workbook; sheets: {', '.join(wb.sheetnames)}"
            )
        last = header_row if header_row is not None else scan_rows
        rows: dict[int, list[HeaderCell]] = {}
        for r_idx, row in enumerate(
            ws.iter_rows(min_row=1, max_row=last, values_only=True), start=1
        ):
            cells = [
                HeaderCell(col=c_idx, text=v)
                for c_idx, v in enumerate(row, start=1)
                if isinstance(v, str) and v.strip()
            ]
            rows[r_idx] = cells
        sheet_name = ws.title
    finally:
        wb.close()

    if header_row is not None:
        chosen = header_row
    else:
        # Earliest row that is nearly as wide as the widest: skips title banners
        # without preferring a (possibly wider) data row below the header.
        widest = max((len(c) for c in rows.values()), default=0)
        floor = max(1, widest * 0.6)
        chosen = next((r for r, c in rows.items() if len(c) >= floor), 0)
    cells = rows.get(chosen, [])
    if not cells:
        raise ConfigError(
            f"no header cells found in {xlsx.name} sheet {sheet_name!r} "
            f"(row {chosen if chosen else 'n/a'})"
        )
    return SheetHeaders(sheet=sheet_name, header_row=chosen, cells=cells)


def _slug(header: str, col: int) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", header.casefold()).strip("_")[:_MAX_KEY_LEN]
    key = key.strip("_")
    if not key:
        return f"col_{col}"
    if key[0].isdigit():
        key = f"c_{key}"
    return key


def _unique(key: str, taken: set[str]) -> str:
    if key not in taken:
        return key
    n = 2
    while f"{key}_{n}" in taken:
        n += 1
    return f"{key}_{n}"


def _new_field(
    *, key: str, header: str, col: int, passthrough: bool, duplicate_of: str | None
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "key": key,
        "excel_header": header,
        "excel_col": col,
        # Placeholder: a developer enriches description/type/validators (S1).
        "description": header,
        "type": "text",
        "charset": "any",
        "validators": [],
        "look_in": [],
        "cardinality": "per_pack" if passthrough else "per_person",
        "critical": False,
        "match": "exact",
    }
    if passthrough:
        spec["passthrough"] = True
    if duplicate_of is not None:
        spec["duplicate_of"] = duplicate_of
    return spec


def merge_headers(
    existing: dict[str, Any] | None,
    headers: SheetHeaders,
    template_id: str,
    *,
    passthrough_headers: Sequence[str] = (),
) -> SyncResult:
    """Merge client headers into an existing template mapping (or start one)."""
    base: dict[str, Any] = dict(existing) if existing else {}
    old_fields: list[dict[str, Any]] = [dict(f) for f in base.get("fields", [])]
    # Existing specs claimed per header in file order, so repeated headers pair up
    # first-with-first (primary before its duplicate).
    by_header: dict[str, list[dict[str, Any]]] = {}
    for old in old_fields:
        by_header.setdefault(_header_match_key(old["excel_header"]), []).append(old)

    wanted_pt = {_header_match_key(h) for h in passthrough_headers}
    known_headers = {_header_match_key(c.text) for c in headers.cells}
    unknown_pt = sorted(wanted_pt - known_headers)
    if unknown_pt:
        raise ConfigError(
            "--passthrough header(s) not in the workbook: " + ", ".join(unknown_pt)
        )

    result = SyncResult(data=base)
    taken: set[str] = {s["key"] for s in old_fields}
    first_key_for_header: dict[str, str] = {}
    new_fields: list[dict[str, Any]] = []
    claimed: set[int] = set()

    for cell in headers.cells:
        hk = _header_match_key(cell.text)
        queue = by_header.get(hk, [])
        matched = queue.pop(0) if queue else None
        spec: dict[str, Any]
        if matched is not None:
            spec = matched
            claimed.add(id(spec))
            if spec.get("excel_col") != cell.col:
                result.moved.append(spec["key"])
            spec["excel_col"] = cell.col
            spec["excel_header"] = cell.text
        else:
            primary = first_key_for_header.get(hk)
            if primary is not None:
                key = _unique(f"{primary}_dup", taken)
                inherited_pt = bool(
                    next(
                        (f for f in new_fields + old_fields if f["key"] == primary), {}
                    ).get("passthrough", False)
                )
                spec = _new_field(
                    key=key,
                    header=cell.text,
                    col=cell.col,
                    passthrough=inherited_pt or hk in wanted_pt,
                    duplicate_of=primary,
                )
            else:
                key = _unique(_slug(cell.text, cell.col), taken)
                spec = _new_field(
                    key=key,
                    header=cell.text,
                    col=cell.col,
                    passthrough=hk in wanted_pt,
                    duplicate_of=None,
                )
            taken.add(key)
            result.added.append(key)
        if hk in wanted_pt and not spec.get("passthrough", False):
            spec["passthrough"] = True
            if spec["key"] not in result.added:
                result.passthrough_set.append(spec["key"])
        first_key_for_header.setdefault(hk, spec["key"])
        new_fields.append(spec)

    result.removed = [s["key"] for s in old_fields if id(s) not in claimed]
    if result.added or result.removed:
        base["version"] = int(base.get("version", 0)) + 1 if existing else 1

    base["id"] = template_id
    base.setdefault("version", 1)
    base.setdefault("description", "")
    base["fields"] = new_fields
    # Keep id/version/description first for readable diffs.
    ordered = {k: base[k] for k in ("id", "version", "description") if k in base}
    ordered.update({k: v for k, v in base.items() if k not in ordered})
    result.data = ordered

    try:
        TemplateConfig.model_validate(result.data)
    except ValidationError as exc:
        raise ConfigError(
            f"synced template {template_id!r} is invalid (a removed column may be "
            f"referenced by row_key or address_split):\n{exc}"
        ) from exc
    return result


def _split_leading_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines).rstrip("\n")


def render_yaml(data: dict[str, Any], *, header_comment: str = "") -> str:
    body = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, width=88, default_flow_style=False
    )
    return f"{header_comment}\n{body}" if header_comment else body


def sync_template(
    xlsx: Path,
    template_id: str,
    *,
    config_dir: Path | None = None,
    sheet: str | None = None,
    header_row: int | None = None,
    passthrough_headers: Sequence[str] = (),
    dry_run: bool = False,
) -> tuple[SyncResult, Path, SheetHeaders]:
    """Read ``xlsx`` headers and create/update ``config/templates/<id>.yaml``."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", template_id):
        raise ConfigError(f"invalid template id: {template_id!r}")
    root = config_dir if config_dir is not None else default_config_dir()
    path = root / _TEMPLATES_DIR / f"{template_id}.yaml"
    existing: dict[str, Any] | None = None
    comment = ""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ConfigError(f"template root must be a mapping: {path}")
        if loaded.get("id") not in (None, template_id):
            raise ConfigError(
                f"{path.name} has id {loaded.get('id')!r}, expected {template_id!r}"
            )
        existing = loaded
        comment = _split_leading_comments(text)
    headers = read_headers(xlsx, sheet=sheet, header_row=header_row)
    result = merge_headers(
        existing, headers, template_id, passthrough_headers=passthrough_headers
    )
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_yaml(result.data, header_comment=comment),
            encoding="utf-8",
            newline="\n",
        )
    return result, path, headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torus.output.template",
        description="Template import: client Excel headers → config/templates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Create or update a template from headers.")
    sync.add_argument(
        "xlsx", type=Path, help="Client workbook (headers only are read)."
    )
    sync.add_argument("template_id", help="Template id (config/templates/<id>.yaml).")
    sync.add_argument("--sheet", default=None, help="Sheet name (default: active).")
    sync.add_argument(
        "--header-row",
        type=int,
        default=None,
        help="1-based header row (default: auto-detect in the first rows).",
    )
    sync.add_argument(
        "--passthrough",
        action="append",
        default=[],
        metavar="HEADER",
        help="Exact header of a column the engine must not fill (repeatable).",
    )
    sync.add_argument("--config-dir", type=Path, default=None)
    sync.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, path, headers = sync_template(
            args.xlsx,
            args.template_id,
            config_dir=args.config_dir,
            sheet=args.sheet,
            header_row=args.header_row,
            passthrough_headers=args.passthrough,
            dry_run=args.dry_run,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"{verb} {path} (sheet {headers.sheet!r}, header row {headers.header_row}, "
        f"{len(headers.cells)} columns)"
    )
    print(
        f"added={len(result.added)} removed={len(result.removed)} "
        f"moved={len(result.moved)} passthrough_set={len(result.passthrough_set)}"
    )
    if result.added:
        print("needs enrichment (S1): " + ", ".join(result.added))
    if result.removed:
        print("dropped (not in workbook): " + ", ".join(result.removed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
