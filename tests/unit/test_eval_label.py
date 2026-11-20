"""Unit tests for the local labelling tool (P1.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torus.config import load_template
from torus.core import EvalError
from torus.eval import (
    OMIT_CELL,
    HandwritingLevel,
    ScanType,
    Split,
    validate_case_dir,
)
from torus.eval.label import (
    LabelCaseRequest,
    build_expected,
    build_meta,
    label_fields,
    open_pdf_locally,
    parse_cell_input,
    prompt_rows_interactive,
    resolve_template,
    write_labelled_case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STARTER = _REPO_ROOT / "config" / "templates" / "starter_v1.yaml"
_SYNTH_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


@pytest.fixture
def starter_template():
    return load_template(_STARTER)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TORUS_DATA_DIR", str(tmp_path))
    return tmp_path


def test_parse_cell_input() -> None:
    assert parse_cell_input("") is OMIT_CELL
    assert parse_cell_input("   ") is OMIT_CELL
    assert parse_cell_input("-") is None
    assert parse_cell_input("  hello ") == "hello"


def test_label_fields_excludes_passthrough(starter_template) -> None:
    keys = [f.key for f in label_fields(starter_template)]
    assert keys == ["person_name", "mobile_no", "address_pincode", "notes"]


def test_build_meta_and_expected(starter_template) -> None:
    meta = build_meta(
        case_id="pack_001",
        split=Split.DEV,
        template_id=starter_template.id,
        year=2024,
        scan_type=ScanType.SCAN,
        handwriting_level=HandwritingLevel.LIGHT,
        page_count_expected=3,
        notes="synth quality check",
    )
    assert meta.case_id == "pack_001"
    assert meta.tags.year == 2024

    expected = build_expected(
        template=starter_template,
        case_id="pack_001",
        rows=[{"person_name": "Synthetic A", "mobile_no": "9876543210"}],
    )
    assert expected.template_id == "starter_v1"
    assert expected.rows[0]["person_name"] == "Synthetic A"


def test_build_expected_rejects_unknown_keys(starter_template) -> None:
    with pytest.raises(EvalError, match="unknown column"):
        build_expected(
            template=starter_template,
            case_id="pack_001",
            rows=[{"not_a_field": "x"}],
        )


def test_prompt_rows_interactive(starter_template) -> None:
    answers = iter(
        [
            "Synthetic Person",
            "9876543210",
            "",
            "-",
        ]
    )
    rows = prompt_rows_interactive(
        starter_template,
        row_count=1,
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda *_a, **_k: None,
    )
    assert rows == [
        {
            "person_name": "Synthetic Person",
            "mobile_no": "9876543210",
            "notes": None,
        }
    ]


def test_write_labelled_case_round_trip(
    data_dir: Path, starter_template, tmp_path: Path
) -> None:
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(_SYNTH_PDF)

    result = write_labelled_case(
        LabelCaseRequest(
            pdf_path=pdf,
            set_id="deb_dev",
            case_id="pack_001",
            template=starter_template,
            split=Split.DEV,
            rows=[
                {
                    "person_name": "Synthetic Person",
                    "mobile_no": "9876543210",
                    "address_pincode": "560001",
                }
            ],
            year=2024,
            scan_type=ScanType.DIGITAL,
            handwriting_level=HandwritingLevel.NONE,
        ),
        data_dir=data_dir,
    )

    case_path = data_dir / "testsets" / "deb_dev" / "pack_001"
    assert result.path == case_path.resolve()
    assert (case_path / "source.pdf").is_file()
    assert (case_path / "source.pdf").read_bytes() == _SYNTH_PDF

    meta = json.loads((case_path / "meta.json").read_text(encoding="utf-8"))
    expected = json.loads((case_path / "expected.json").read_text(encoding="utf-8"))
    assert meta["case_id"] == "pack_001"
    assert meta["split"] == "dev"
    assert expected["template_id"] == "starter_v1"
    assert expected["rows"][0]["person_name"] == "Synthetic Person"

    validated = validate_case_dir(
        case_path,
        set_id="deb_dev",
        template_keys=frozenset(f.key for f in starter_template.fields),
    )
    assert validated.case_id == "pack_001"
    assert validated.expected.rows[0]["mobile_no"] == "9876543210"


def test_write_requires_data_dir(
    starter_template, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TORUS_DATA_DIR", raising=False)
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(_SYNTH_PDF)
    with pytest.raises(EvalError, match="TORUS_DATA_DIR"):
        write_labelled_case(
            LabelCaseRequest(
                pdf_path=pdf,
                set_id="s",
                case_id="c1",
                template=starter_template,
                split=Split.DEV,
                rows=[{}],
            )
        )


def test_write_refuses_overwrite_without_force(
    data_dir: Path, starter_template, tmp_path: Path
) -> None:
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(_SYNTH_PDF)
    req = LabelCaseRequest(
        pdf_path=pdf,
        set_id="s",
        case_id="c1",
        template=starter_template,
        split=Split.HIDDEN,
        rows=[{"person_name": "A"}],
    )
    write_labelled_case(req, data_dir=data_dir)
    with pytest.raises(EvalError, match="already exists"):
        write_labelled_case(req, data_dir=data_dir)

    write_labelled_case(
        LabelCaseRequest(
            pdf_path=pdf,
            set_id="s",
            case_id="c1",
            template=starter_template,
            split=Split.HIDDEN,
            rows=[{"person_name": "B"}],
            force=True,
        ),
        data_dir=data_dir,
    )
    expected = json.loads(
        (data_dir / "testsets" / "s" / "c1" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["rows"][0]["person_name"] == "B"


def test_write_rejects_missing_pdf(
    data_dir: Path, starter_template, tmp_path: Path
) -> None:
    with pytest.raises(EvalError, match="PDF"):
        write_labelled_case(
            LabelCaseRequest(
                pdf_path=tmp_path / "missing.pdf",
                set_id="s",
                case_id="c1",
                template=starter_template,
                split=Split.DEV,
                rows=[{}],
            ),
            data_dir=data_dir,
        )


def test_resolve_template_by_id() -> None:
    template = resolve_template("starter_v1", config_dir=_REPO_ROOT / "config")
    assert template.id == "starter_v1"


def test_open_pdf_locally_missing(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="PDF"):
        open_pdf_locally(tmp_path / "nope.pdf")


def test_cli_label_from_rows_json(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torus.cli import main

    pdf = tmp_path / "pack.pdf"
    pdf.write_bytes(_SYNTH_PDF)
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            [
                {
                    "person_name": "Synthetic CLI",
                    "mobile_no": "9123456780",
                    "address_pincode": "110001",
                    "notes": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TORUS_DATA_DIR", str(data_dir))
    code = main(
        [
            "label",
            "--pdf",
            str(pdf),
            "--set",
            "cli_set",
            "--case",
            "cli_case",
            "--template",
            "starter_v1",
            "--config-dir",
            str(_REPO_ROOT / "config"),
            "--split",
            "dev",
            "--rows-json",
            str(rows_path),
            "--no-open",
        ]
    )
    assert code == 0
    case = data_dir / "testsets" / "cli_set" / "cli_case"
    assert (case / "expected.json").is_file()
    assert (case / "meta.json").is_file()
    assert (case / "source.pdf").is_file()
