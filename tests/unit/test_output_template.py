"""P8.1: template import from client Excel headers (synthetic workbooks only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook

from torus.config.loader import load_config
from torus.config.schemas import TemplateConfig
from torus.core.errors import ConfigError
from torus.output.template import main, read_headers, sync_template

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _workbook(
    path: Path,
    headers: list[str | None],
    *,
    title_rows: int = 0,
    data_rows: int = 2,
) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet A"
    for i in range(title_rows):
        ws.cell(row=i + 1, column=1, value=f"banner {i}")
    header_row = title_rows + 1
    for c, h in enumerate(headers, start=1):
        if h is not None:
            ws.cell(row=header_row, column=c, value=h)
    for r in range(data_rows):
        for c in range(1, len(headers) + 1):
            ws.cell(row=header_row + 1 + r, column=c, value="SECRET-DATA")
    wb.save(path)
    return path


def _config_dir(tmp_path: Path) -> Path:
    root = tmp_path / "cfg"
    (root / "templates").mkdir(parents=True)
    return root


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_read_headers_auto_detects_row_and_skips_blank_columns(tmp_path: Path) -> None:
    xlsx = _workbook(
        tmp_path / "t.xlsx", ["Alpha", None, "Beta  Gamma ", "Delta"], title_rows=2
    )
    h = read_headers(xlsx)
    assert h.header_row == 3
    assert [(c.col, c.text) for c in h.cells] == [
        (1, "Alpha"),
        (3, "Beta  Gamma "),
        (4, "Delta"),
    ]


def test_read_headers_never_returns_data_cells(tmp_path: Path) -> None:
    xlsx = _workbook(tmp_path / "t.xlsx", ["A", "B"], data_rows=5)
    h = read_headers(xlsx)
    assert all("SECRET" not in c.text for c in h.cells)


def test_read_headers_unknown_sheet_and_missing_file(tmp_path: Path) -> None:
    xlsx = _workbook(tmp_path / "t.xlsx", ["A"])
    with pytest.raises(ConfigError, match="not in workbook"):
        read_headers(xlsx, sheet="nope")
    with pytest.raises(ConfigError, match="not found"):
        read_headers(tmp_path / "missing.xlsx")


def test_sync_creates_valid_template_with_dup_and_passthrough(tmp_path: Path) -> None:
    xlsx = _workbook(
        tmp_path / "t.xlsx", ["Full Name", "Ref No", "Full Name", "1st Phone"]
    )
    cfg = _config_dir(tmp_path)
    result, path, _ = sync_template(
        xlsx, "demo_v1", config_dir=cfg, passthrough_headers=["ref no"]
    )
    data = _load(path)
    tpl = TemplateConfig.model_validate(data)
    assert [f.key for f in tpl.fields] == [
        "full_name",
        "ref_no",
        "full_name_dup",
        "c_1st_phone",
    ]
    assert [f.excel_col for f in tpl.fields] == [1, 2, 3, 4]
    assert tpl.field("ref_no") is not None and tpl.field("ref_no").passthrough  # type: ignore[union-attr]
    dup = tpl.field("full_name_dup")
    assert dup is not None and dup.duplicate_of == "full_name"
    assert [f.key for f in tpl.scored_fields()] == [
        "full_name",
        "full_name_dup",
        "c_1st_phone",
    ]
    assert result.added == ["full_name", "ref_no", "full_name_dup", "c_1st_phone"]
    assert tpl.version == 1


def test_sync_preserves_enrichment_and_reports_changes(tmp_path: Path) -> None:
    cfg = _config_dir(tmp_path)
    first = _workbook(tmp_path / "a.xlsx", ["Name", "Mobile", "Old Col"])
    sync_template(first, "demo_v1", config_dir=cfg)
    path = cfg / "templates" / "demo_v1.yaml"
    data = _load(path)
    for f in data["fields"]:
        if f["key"] == "mobile":
            f.update(
                description="Primary mobile",
                type="phone_in",
                charset="digits",
                length=10,
                validators=["indian_mobile"],
                critical=True,
            )
    path.write_text("# keep me\n" + yaml.safe_dump(data, sort_keys=False))

    # Client reorders, drops "Old Col", adds "New Col".
    second = _workbook(tmp_path / "b.xlsx", ["Mobile", "Name", "New Col"])
    result, _, _ = sync_template(second, "demo_v1", config_dir=cfg)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# keep me")
    tpl = TemplateConfig.model_validate(yaml.safe_load(text))
    assert [f.key for f in tpl.fields] == ["mobile", "name", "new_col"]
    mobile = tpl.field("mobile")
    assert mobile is not None
    assert mobile.excel_col == 1
    assert mobile.critical and mobile.validators == ["indian_mobile"]
    assert result.added == ["new_col"]
    assert result.removed == ["old_col"]
    assert set(result.moved) == {"mobile", "name"}
    assert tpl.version == 2


def test_sync_is_idempotent(tmp_path: Path) -> None:
    cfg = _config_dir(tmp_path)
    xlsx = _workbook(tmp_path / "t.xlsx", ["A", "B", "A"])
    sync_template(xlsx, "demo_v1", config_dir=cfg)
    path = cfg / "templates" / "demo_v1.yaml"
    before = path.read_text(encoding="utf-8")
    result, _, _ = sync_template(xlsx, "demo_v1", config_dir=cfg)
    assert not result.changed
    assert path.read_text(encoding="utf-8") == before


def test_sync_can_mark_existing_column_passthrough(tmp_path: Path) -> None:
    cfg = _config_dir(tmp_path)
    xlsx = _workbook(tmp_path / "t.xlsx", ["A", "Checker"])
    sync_template(xlsx, "demo_v1", config_dir=cfg)
    result, path, _ = sync_template(
        xlsx, "demo_v1", config_dir=cfg, passthrough_headers=["Checker"]
    )
    assert result.passthrough_set == ["checker"]
    tpl = TemplateConfig.model_validate(_load(path))
    assert [f.key for f in tpl.scored_fields()] == ["a"]


def test_sync_rejects_unknown_passthrough_header(tmp_path: Path) -> None:
    xlsx = _workbook(tmp_path / "t.xlsx", ["A"])
    with pytest.raises(ConfigError, match="not in the workbook"):
        sync_template(
            xlsx,
            "demo_v1",
            config_dir=_config_dir(tmp_path),
            passthrough_headers=["Zed"],
        )


def test_sync_fails_loudly_when_row_key_column_removed(tmp_path: Path) -> None:
    cfg = _config_dir(tmp_path)
    (cfg / "templates" / "demo_v1.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "demo_v1",
                "version": 1,
                "row_key": {"primary": [{"sources": ["name"]}]},
                "fields": [
                    {
                        "key": "name",
                        "excel_header": "Name",
                        "description": "n",
                        "type": "name",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    xlsx = _workbook(tmp_path / "t.xlsx", ["Other"])
    path = cfg / "templates" / "demo_v1.yaml"
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        sync_template(xlsx, "demo_v1", config_dir=cfg)
    assert path.read_text(encoding="utf-8") == before


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = _config_dir(tmp_path)
    xlsx = _workbook(tmp_path / "t.xlsx", ["A"])
    result, path, _ = sync_template(xlsx, "demo_v1", config_dir=cfg, dry_run=True)
    assert result.added == ["a"]
    assert not path.exists()


def test_invalid_template_id_rejected(tmp_path: Path) -> None:
    xlsx = _workbook(tmp_path / "t.xlsx", ["A"])
    with pytest.raises(ConfigError, match="invalid template id"):
        sync_template(xlsx, "../evil", config_dir=_config_dir(tmp_path))


def test_cli_sync_and_error_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config_dir(tmp_path)
    xlsx = _workbook(tmp_path / "t.xlsx", ["A", "B"])
    rc = main(["sync", str(xlsx), "demo_v1", "--config-dir", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0 and "added=2" in out
    rc = main(
        ["sync", str(tmp_path / "nope.xlsx"), "demo_v1", "--config-dir", str(cfg)]
    )
    assert rc == 2


def test_deb_v1_round_trips_through_sync_without_change(tmp_path: Path) -> None:
    """A workbook built from deb_v1's own headers leaves the template untouched."""
    cfg = load_config(_REPO_ROOT / "config")
    tpl = cfg.template("deb_v1")
    width = max(f.excel_col or 0 for f in tpl.fields)
    headers: list[str | None] = [None] * width
    for f in tpl.fields:
        assert f.excel_col is not None
        headers[f.excel_col - 1] = f.excel_header
    xlsx = _workbook(tmp_path / "deb.xlsx", headers)
    out = _config_dir(tmp_path)
    (out / "templates" / "deb_v1.yaml").write_text(
        (_REPO_ROOT / "config" / "templates" / "deb_v1.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    result, path, _ = sync_template(xlsx, "deb_v1", config_dir=out, header_row=1)
    assert not result.changed
    assert TemplateConfig.model_validate(_load(path)) == tpl
