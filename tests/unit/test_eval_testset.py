"""Unit tests for evaluation test-set format (P1.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.fixtures.synth.testset import (
    MINI_SET_ID,
    MINI_TEMPLATE_ID,
    build_mini_testset,
)
from torus.config import load_template
from torus.core import EvalError
from torus.eval import (
    CaseMeta,
    ExpectedDocument,
    HandwritingLevel,
    ScanType,
    Split,
    assert_no_duplicate_case_ids,
    case_dir,
    get_testsets_root,
    list_set_ids,
    resolve_data_dir,
    set_dir,
    validate_case_dir,
    validate_testset,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STARTER_TEMPLATE = _REPO_ROOT / "config" / "templates" / "starter_v1.yaml"


def _write_case(
    root: Path,
    *,
    set_id: str,
    case_id: str,
    meta: dict[str, object],
    expected: dict[str, object],
    with_pdf: bool = True,
) -> Path:
    case = root / "testsets" / set_id / case_id
    case.mkdir(parents=True, exist_ok=True)
    (case / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (case / "expected.json").write_text(json.dumps(expected) + "\n", encoding="utf-8")
    if with_pdf:
        (case / "source.pdf").write_bytes(b"%PDF-1.4 synth")
    return case


def test_resolve_data_dir_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORUS_DATA_DIR", raising=False)
    with pytest.raises(EvalError, match="TORUS_DATA_DIR"):
        resolve_data_dir()


def test_path_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORUS_DATA_DIR", str(tmp_path))
    assert resolve_data_dir() == tmp_path.resolve()
    assert get_testsets_root() == tmp_path.resolve() / "testsets"
    assert set_dir("mini") == tmp_path.resolve() / "testsets" / "mini"
    assert case_dir("mini", "case_a") == (
        tmp_path.resolve() / "testsets" / "mini" / "case_a"
    )


def test_path_helpers_reject_traversal() -> None:
    with pytest.raises(EvalError):
        set_dir("../outside")
    with pytest.raises(EvalError):
        case_dir("mini", "a/b")


def test_case_meta_and_expected_schemas() -> None:
    meta = CaseMeta.model_validate(
        {
            "case_id": "c1",
            "split": "dev",
            "template_id": "starter_v1",
            "tags": {
                "year": 2024,
                "scan_type": "digital",
                "handwriting_level": "none",
            },
        }
    )
    assert meta.split is Split.DEV
    assert meta.tags.scan_type is ScanType.DIGITAL
    assert meta.tags.handwriting_level is HandwritingLevel.NONE

    expected = ExpectedDocument.model_validate(
        {
            "template_id": "starter_v1",
            "rows": [{"person_name": "Synthetic", "_excel_row": 2}],
        }
    )
    assert expected.column_keys() == frozenset({"person_name"})


def test_expected_rejects_nested_values() -> None:
    with pytest.raises(ValidationError):
        ExpectedDocument.model_validate(
            {
                "template_id": "starter_v1",
                "rows": [{"person_name": {"nested": True}}],
            }
        )


def test_validate_case_dir_ok(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        set_id="s",
        case_id="c1",
        meta={
            "case_id": "c1",
            "split": "hidden",
            "template_id": "starter_v1",
            "tags": {"year": 2022, "scan_type": "fax", "handwriting_level": "heavy"},
        },
        expected={
            "template_id": "starter_v1",
            "case_id": "c1",
            "rows": [{"mobile_no": "9000000000"}],
        },
    )
    validated = validate_case_dir(case, set_id="s")
    assert validated.case_id == "c1"
    assert validated.meta.split is Split.HIDDEN
    assert validated.expected.rows[0]["mobile_no"] == "9000000000"


def test_validate_case_id_mismatch(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        set_id="s",
        case_id="folder_id",
        meta={
            "case_id": "other_id",
            "split": "dev",
            "template_id": "starter_v1",
        },
        expected={"template_id": "starter_v1", "rows": []},
    )
    with pytest.raises(EvalError, match="does not match folder"):
        validate_case_dir(case)


def test_validate_template_id_mismatch(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        set_id="s",
        case_id="c1",
        meta={"case_id": "c1", "split": "dev", "template_id": "starter_v1"},
        expected={"template_id": "deb_v1", "rows": []},
    )
    with pytest.raises(EvalError, match="template_id mismatch"):
        validate_case_dir(case)


def test_validate_unknown_column_keys(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        set_id="s",
        case_id="c1",
        meta={"case_id": "c1", "split": "dev", "template_id": "starter_v1"},
        expected={
            "template_id": "starter_v1",
            "rows": [{"not_a_column": "x", "person_name": "Y"}],
        },
    )
    with pytest.raises(EvalError, match="unknown column keys"):
        validate_case_dir(case, template_keys=frozenset({"person_name", "mobile_no"}))


def test_validate_missing_source(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path,
        set_id="s",
        case_id="c1",
        meta={"case_id": "c1", "split": "dev", "template_id": "starter_v1"},
        expected={"template_id": "starter_v1", "rows": []},
        with_pdf=False,
    )
    with pytest.raises(EvalError, match="source.pdf"):
        validate_case_dir(case)
    ok = validate_case_dir(case, require_source=False)
    assert ok.case_id == "c1"


def test_mini_testset_validates_against_starter(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    template = load_template(_STARTER_TEMPLATE)
    keys = frozenset(f.key for f in template.fields)
    assert template.id == MINI_TEMPLATE_ID

    cases = validate_testset(
        MINI_SET_ID,
        data_dir=data_dir,
        template_keys=keys,
    )
    assert len(cases) == 2
    splits = {c.meta.split for c in cases}
    assert splits == {Split.DEV, Split.HIDDEN}
    assert list_set_ids(data_dir=data_dir) == [MINI_SET_ID]
    # Optional gold side-cars from the mini set builder.
    assert all(c.pages is not None for c in cases)
    assert all(c.locations is not None for c in cases)
    assert all(len(c.locations.fields) >= 1 for c in cases)  # type: ignore[union-attr]

    dev_only = validate_testset(
        MINI_SET_ID, data_dir=data_dir, split=Split.DEV, template_keys=keys
    )
    assert len(dev_only) == 1
    assert dev_only[0].case_id == "case_dev_001"

    assert_no_duplicate_case_ids(cases)


def test_assert_duplicate_case_ids(tmp_path: Path) -> None:
    data_dir = build_mini_testset(tmp_path / "data")
    cases = validate_testset(MINI_SET_ID, data_dir=data_dir)
    with pytest.raises(EvalError, match="duplicate case_id"):
        assert_no_duplicate_case_ids([*cases, cases[0]])
