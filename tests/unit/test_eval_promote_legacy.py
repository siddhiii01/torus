"""Unit tests for legacy gold → P1.1 promote (P1.6)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.synth.testset.build_mini_set import write_synth_pdf
from torus.eval.promote_legacy import promote_legacy_case, promote_legacy_set
from torus.eval.schema import Split
from torus.eval.validate import validate_case_dir, validate_testset


def _write_legacy_case(root: Path, case_id: str, *, rows: list[dict]) -> Path:
    case = root / case_id
    case.mkdir(parents=True)
    write_synth_pdf(case / "source.pdf", label=f"LEGACY {case_id}")
    (case / "gold.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "gold_version": "v0",
                "rows": rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (case / "meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "page_count_expected": 1,
                "gold_status": "frozen_v0",
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return case


def test_promote_legacy_case(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy" / "deb" / "cases"
    data_dir = tmp_path / "data"
    _write_legacy_case(
        legacy_root,
        "synth_pack_001",
        rows=[
            {
                "_excel_row": 1,
                "person_name": "Synthetic Person",
                "mobile_no": "9876543210",
                "address_type": "H1",
                "address_source": "application form",
            }
        ],
    )

    result = promote_legacy_case(
        legacy_root / "synth_pack_001",
        set_id="deb_dev",
        template_id="starter_v1",
        data_dir=data_dir,
        force=True,
    )
    assert result.created is True
    validated = validate_case_dir(
        result.dest,
        set_id="deb_dev",
        require_source=True,
    )
    assert validated.meta.split == Split.DEV
    assert validated.meta.template_id == "starter_v1"
    assert validated.expected.rows[0]["person_name"] == "Synthetic Person"
    assert "_excel_row" not in validated.expected.rows[0]
    # Customer display names from legacy meta must not land in new meta.
    assert "customer_name" not in validated.meta.model_dump()


def test_promote_legacy_set_idempotent(tmp_path: Path) -> None:
    legacy_root = tmp_path / "cases"
    data_dir = tmp_path / "data"
    _write_legacy_case(
        legacy_root,
        "a",
        rows=[{"person_name": "A", "mobile_no": "1"}],
    )
    _write_legacy_case(
        legacy_root,
        "b",
        rows=[{"person_name": "B", "mobile_no": "2"}],
    )

    first = promote_legacy_set(
        legacy_cases_dir=legacy_root,
        set_id="deb_dev",
        template_id="starter_v1",
        data_dir=data_dir,
    )
    assert len(first) == 2
    assert all(r.created for r in first)

    second = promote_legacy_set(
        legacy_cases_dir=legacy_root,
        set_id="deb_dev",
        template_id="starter_v1",
        data_dir=data_dir,
        force=False,
    )
    assert len(second) == 2
    assert all(not r.created for r in second)

    cases = validate_testset("deb_dev", data_dir=data_dir)
    assert {c.case_id for c in cases} == {"a", "b"}
