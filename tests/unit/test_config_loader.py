"""Unit tests for torus.config load / validate / hash (P0.4, P0.10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from torus.config import (
    TemplateConfig,
    TorusConfig,
    compute_config_hash,
    default_config_dir,
    load_config,
    load_template,
)
from torus.core import ConfigError, ExtractionResult, FieldStatus, FieldValue
from torus.core.protocols import TemplateSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEB_COLUMNS = _REPO_ROOT / "schemas" / "deb" / "excel_columns.json"
_DEB_GROUPS = _REPO_ROOT / "schemas" / "deb" / "field_groups.json"


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _minimal_tree(root: Path) -> None:
    """Write a tiny valid config tree under ``root``."""
    _write_yaml(
        root / "pipeline.yaml",
        {
            "dpi": 200,
            "confidence": {"confirmed_min": 0.9, "likely_min": 0.7},
            "voting": {
                "critical_min_families": 2,
                "noncritical_min_readers": 1,
            },
            "regression": {"noise_margin": 0.01, "silent_error_tolerance": 0},
        },
    )
    _write_yaml(
        root / "readers.yaml",
        {
            "readers": [
                {
                    "name": "stub_a",
                    "family": "stub",
                    "enabled": False,
                    "min_vram_gb": 0,
                    "min_ram_gb": 0,
                    "device": "cpu",
                    "capabilities": {"printed": True},
                }
            ]
        },
    )
    _write_yaml(
        root / "page_types.yaml",
        {
            "page_types": [
                {"id": "application_form", "description": "Form page."},
                {"id": "other", "description": "Catch-all."},
            ]
        },
    )
    _write_yaml(
        root / "templates" / "demo_v1.yaml",
        {
            "id": "demo_v1",
            "version": 1,
            "description": "Test template",
            "fields": [
                {
                    "key": "mobile_no",
                    "excel_header": "Mobile No",
                    "description": "Ten-digit mobile.",
                    "type": "phone_in",
                    "charset": "digits",
                    "length": 10,
                    "validators": ["indian_mobile"],
                    "look_in": ["application_form"],
                    "cardinality": "per_person",
                    "critical": True,
                }
            ],
        },
    )


def test_default_config_dir_points_at_repo_config() -> None:
    cfg_dir = default_config_dir()
    assert cfg_dir.name == "config"
    assert (cfg_dir / "pipeline.yaml").is_file()
    assert (cfg_dir / "readers.yaml").is_file()
    assert (cfg_dir / "page_types.yaml").is_file()
    assert (cfg_dir / "templates" / "starter_v1.yaml").is_file()


def test_repo_starters_load_and_hash() -> None:
    cfg = load_config()
    assert isinstance(cfg, TorusConfig)
    assert len(cfg.config_hash) == 64
    assert int(cfg.config_hash, 16) >= 0
    assert "starter_v1" in cfg.templates
    assert "deb_v1" in cfg.templates
    starter = cfg.template("starter_v1")
    mobile = starter.field("mobile_no")
    assert mobile is not None
    assert mobile.critical is True
    assert mobile.length == 10
    assert cfg.pipeline.dpi == 300
    assert cfg.readers.get("stub_echo") is not None
    assert "application_form" in cfg.page_types.ids()


# Pack types required by P3.1. Specific ID issuers live under id_card cues so
# existing template look_in: [id_card] keeps working until look_in is widened.
_REQUIRED_PAGE_TYPE_IDS = frozenset(
    {
        "application_form",
        "form_continuation",
        "verification_report",
        "id_card",
        "bank_statement",
        "approval_memo",
        "checklist",
        "brochure_terms",
        "photo_signature",
        "blank",
        "other",
    }
)


def test_page_types_catalogue_is_populated() -> None:
    """P3.1: every pack type has description, examples and cues."""
    cfg = load_config()
    ids = cfg.page_types.ids()
    assert ids >= _REQUIRED_PAGE_TYPE_IDS
    for page_type_id in _REQUIRED_PAGE_TYPE_IDS:
        spec = cfg.page_types.get(page_type_id)
        assert spec is not None
        assert spec.description.strip()
        assert spec.examples, f"{page_type_id}: examples required"
        assert spec.visual_cues, f"{page_type_id}: visual_cues required"
        assert spec.text_cues, f"{page_type_id}: text_cues required"
    # id_card cues must cover the common identity-document families in packs
    id_card = cfg.page_types.get("id_card")
    assert id_card is not None
    cue_blob = " ".join(
        [
            id_card.description,
            *id_card.examples,
            *id_card.visual_cues,
            *id_card.text_cues,
        ]
    ).lower()
    for fragment in (
        "voter",
        "passport",
        "driving",
        "ration",
        "identity",
        "tax",
    ):
        assert fragment in cue_blob, f"id_card cues missing {fragment!r}"


def test_deb_v1_loads_and_matches_schemas() -> None:
    """P0.10: deb_v1 covers excel_columns + field_groups and validates."""
    columns = json.loads(_DEB_COLUMNS.read_text(encoding="utf-8"))
    groups = json.loads(_DEB_GROUPS.read_text(encoding="utf-8"))
    # field_groups.json only distinguishes exact|fuzzy; P1.2 maps legacy
    # "exact" → template match=normalised (type-aware), fuzzy stays fuzzy.
    scored_match = {
        entry["key"]: ("normalised" if entry["match"] == "exact" else entry["match"])
        for entry in groups["scored"]
    }
    passthrough_keys = set(groups["passthrough"])
    # Duplicate header column is not listed under scored or passthrough.
    expected_passthrough = passthrough_keys | {"bank_lan_dup"}

    cfg = load_config()
    template = cfg.template("deb_v1")
    assert template.id == "deb_v1"
    assert isinstance(template, TemplateSpec)
    assert template.row_key is not None
    assert len(template.row_key.primary) == 3

    schema_keys = [col["key"] for col in columns["columns"]]
    template_keys = [field.key for field in template.fields]
    assert template_keys == schema_keys
    assert len(template.fields) == 73

    for col in columns["columns"]:
        field = template.field(col["key"])
        assert field is not None
        assert field.excel_header == col["excel_header"]
        assert field.excel_col == col["excel_col"]
        assert field.duplicate_of == col["duplicate_of"]
        assert field.description
        if col["key"] in scored_match:
            assert field.passthrough is False
            assert field.match == scored_match[col["key"]]
            # page_count is pack metadata from ingest, not located on a page type.
            if col["key"] != "page_count":
                assert field.look_in, f"{col['key']}: scored fields need look_in"
        else:
            assert col["key"] in expected_passthrough
            assert field.passthrough is True
            assert field.critical is False
            assert field.look_in == []
            assert field.validators == []

    mobile = template.field("mobile_no")
    assert mobile is not None
    assert mobile.type.value == "phone_in"
    assert mobile.length == 10
    assert mobile.validators == ["indian_mobile"]
    assert mobile.critical is True
    assert mobile.cardinality.value == "per_person"

    pin = template.field("address_pincode")
    assert pin is not None
    assert pin.type.value == "pin"
    assert pin.length == 6
    assert pin.validators == ["indian_pin"]
    assert pin.cardinality.value == "per_address"

    fuzzy = {key for key, match in scored_match.items() if match == "fuzzy"}
    assert fuzzy == {
        "address_line1",
        "address_line2",
        "address_line3",
        "business_type",
    }
    for key in fuzzy:
        field = template.field(key)
        assert field is not None
        assert field.match == "fuzzy"

    dup = template.field("bank_lan_dup")
    assert dup is not None
    assert dup.duplicate_of == "bank_lan"
    assert dup.passthrough is True

    # Row-key fields used for alignment (schemas/deb/row_key.md) are present.
    for key in ("person_name", "customer_name", "address_type", "address_source"):
        field = template.field(key)
        assert field is not None
        assert field.passthrough is False
        assert field.critical is True


def test_duplicate_of_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_dup_v1.yaml"
    _write_yaml(
        path,
        {
            "id": "bad_dup_v1",
            "fields": [
                {
                    "key": "a",
                    "excel_header": "A",
                    "description": "a",
                    "type": "text",
                    "duplicate_of": "missing_key",
                }
            ],
        },
    )
    with pytest.raises(ConfigError, match="duplicate_of"):
        load_template(path)


def test_template_satisfies_protocol() -> None:
    cfg = load_config()
    template = cfg.template("starter_v1")
    assert isinstance(template, TemplateSpec)
    assert template.id == "starter_v1"


def test_config_hash_stable_and_sensitive(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    first = load_config(tmp_path)
    second = load_config(tmp_path)
    assert first.config_hash == second.config_hash
    assert first.config_hash == compute_config_hash(
        first.pipeline, first.readers, first.page_types, first.templates
    )

    pipeline_path = tmp_path / "pipeline.yaml"
    data = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    data["dpi"] = 300
    _write_yaml(pipeline_path, data)
    third = load_config(tmp_path)
    assert third.config_hash != first.config_hash


def test_local_overrides_flip_enabled_and_change_hash(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    before = load_config(tmp_path)
    stub_before = before.readers.get("stub_a")
    assert stub_before is not None
    assert stub_before.enabled is False

    _write_yaml(tmp_path / "local.yaml", {"readers": {"stub_a": True}})
    after = load_config(tmp_path)
    stub_after = after.readers.get("stub_a")
    assert stub_after is not None
    assert stub_after.enabled is True
    assert after.config_hash != before.config_hash


def test_unknown_local_reader_raises(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    _write_yaml(tmp_path / "local.yaml", {"readers": {"no_such_reader": True}})
    with pytest.raises(ConfigError, match="unknown readers"):
        load_config(tmp_path)


def test_invalid_pipeline_raises(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    _write_yaml(
        tmp_path / "pipeline.yaml",
        {
            "dpi": 300,
            "confidence": {"confirmed_min": 0.5, "likely_min": 0.9},
            "voting": {
                "critical_min_families": 2,
                "noncritical_min_readers": 1,
            },
            "regression": {"noise_margin": 0.01, "silent_error_tolerance": 0},
        },
    )
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(tmp_path)


def test_duplicate_reader_names_raise(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    _write_yaml(
        tmp_path / "readers.yaml",
        {
            "readers": [
                {
                    "name": "dup",
                    "family": "a",
                    "min_vram_gb": 0,
                    "min_ram_gb": 0,
                },
                {
                    "name": "dup",
                    "family": "b",
                    "min_vram_gb": 0,
                    "min_ram_gb": 0,
                },
            ]
        },
    )
    with pytest.raises(ConfigError, match="unique"):
        load_config(tmp_path)


def test_look_in_unknown_page_type_raises(tmp_path: Path) -> None:
    _minimal_tree(tmp_path)
    _write_yaml(
        tmp_path / "templates" / "demo_v1.yaml",
        {
            "id": "demo_v1",
            "fields": [
                {
                    "key": "x",
                    "excel_header": "X",
                    "description": "x",
                    "type": "text",
                    "look_in": ["not_a_real_page_type"],
                }
            ],
        },
    )
    with pytest.raises(ConfigError, match="unknown page types"):
        load_config(tmp_path)


def test_template_id_must_match_filename(tmp_path: Path) -> None:
    path = tmp_path / "wrong_name.yaml"
    _write_yaml(
        path,
        {
            "id": "right_id",
            "fields": [],
        },
    )
    with pytest.raises(ConfigError, match="must match file stem"):
        load_template(path)


def test_missing_config_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config directory not found"):
        load_config(tmp_path / "does_not_exist")


def test_extraction_result_records_config_hash() -> None:
    cfg = load_config()
    result = ExtractionResult(
        document_id="doc1",
        template_id="starter_v1",
        rows=[
            {
                "notes": FieldValue(
                    column_key="notes",
                    value=None,
                    status=FieldStatus.NOT_FOUND,
                    confidence=0.0,
                )
            }
        ],
        config_hash=cfg.config_hash,
    )
    assert result.config_hash == cfg.config_hash


def test_torus_config_unknown_template_raises() -> None:
    cfg = load_config()
    with pytest.raises(KeyError, match="unknown template"):
        cfg.template("nope")


def test_load_template_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "solo_v1.yaml"
    _write_yaml(
        path,
        {
            "id": "solo_v1",
            "fields": [
                {
                    "key": "a",
                    "excel_header": "A",
                    "description": "a",
                    "type": "text",
                }
            ],
        },
    )
    template = load_template(path)
    assert isinstance(template, TemplateConfig)
    assert template.id == "solo_v1"
    assert template.field("a") is not None
