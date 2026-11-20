"""Unit tests for form-family fingerprint, save/load, and hints (P4.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from torus.config import (
    FormFamilySpec,
    FormFamilyStatus,
    LayoutFamiliesConfig,
    load_config,
    load_families,
    load_family,
)
from torus.config.schemas import FieldSpec
from torus.core import BBox, Region, RegionKind
from torus.core.types import FieldType
from torus.layout import (
    AnchoredField,
    RelativePlacement,
    anchor_fields,
    family_id_for_fingerprint,
    fingerprint_layout,
    match_family,
    observe_and_maybe_save,
)
from torus.layout.match import LabelFieldMatch, LabelMatchKind


def _bbox(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _region(rid: str, bbox: BBox, *, kind: RegionKind = RegionKind.TEXT_LINE) -> Region:
    return Region(id=rid, page_index=0, bbox=bbox, kind=kind)


def _anchored(
    field_key: str,
    label: BBox,
    value: BBox,
    *,
    placement: RelativePlacement = RelativePlacement.RIGHT,
) -> AnchoredField:
    return AnchoredField(
        field_key=field_key,
        match=LabelFieldMatch(
            field_key=field_key,
            label_text=field_key,
            kind=LabelMatchKind.EXACT,
            score=1.0,
            matched_phrase=field_key,
        ),
        label_region=_region(f"lab-{field_key}", label),
        value_region=_region(f"val-{field_key}", value, kind=RegionKind.FIELD_VALUE),
        placement=placement,
    )


def test_fingerprint_translation_invariant() -> None:
    a = fingerprint_layout(
        [("mobile_no", 0.10, 0.20), ("person_name", 0.10, 0.40)],
        position_bin=0.05,
    )
    b = fingerprint_layout(
        [("mobile_no", 0.15, 0.25), ("person_name", 0.15, 0.45)],
        position_bin=0.05,
    )
    assert a == b


def test_fingerprint_differs_for_different_relative_layout() -> None:
    a = fingerprint_layout(
        [("mobile_no", 0.10, 0.20), ("person_name", 0.10, 0.40)],
        position_bin=0.05,
    )
    b = fingerprint_layout(
        [("mobile_no", 0.10, 0.20), ("person_name", 0.50, 0.20)],
        position_bin=0.05,
    )
    assert a != b


def test_family_id_stable() -> None:
    fp = fingerprint_layout([("a", 0.1, 0.1), ("b", 0.2, 0.3)])
    assert family_id_for_fingerprint(fp) == f"fam_{fp}"


def test_observe_promotes_to_generated(tmp_path: Path) -> None:
    families = tmp_path / "families"
    cfg = LayoutFamiliesConfig(min_observations=2, min_labels=2)
    anchored = [
        _anchored("mobile_no", _bbox(0.1, 0.2, 0.2, 0.24), _bbox(0.25, 0.2, 0.4, 0.24)),
        _anchored(
            "person_name",
            _bbox(0.1, 0.4, 0.2, 0.44),
            _bbox(0.25, 0.4, 0.5, 0.44),
        ),
    ]
    first = observe_and_maybe_save(anchored, families, config=cfg)
    assert first is not None
    assert first.status is FormFamilyStatus.OBSERVING
    assert first.observation_count == 1
    assert (families / f"{first.id}.yaml").is_file()

    second = observe_and_maybe_save(anchored, families, config=cfg)
    assert second is not None
    assert second.status is FormFamilyStatus.GENERATED
    assert second.observation_count == 2

    loaded = load_family(families / f"{second.id}.yaml")
    assert loaded.status is FormFamilyStatus.GENERATED
    assert {lab.field_key for lab in loaded.labels} == {"mobile_no", "person_name"}


def test_observe_disabled_returns_none(tmp_path: Path) -> None:
    anchored = [
        _anchored("mobile_no", _bbox(0.1, 0.2, 0.2, 0.24), _bbox(0.25, 0.2, 0.4, 0.24)),
        _anchored(
            "person_name",
            _bbox(0.1, 0.4, 0.2, 0.44),
            _bbox(0.25, 0.4, 0.5, 0.44),
        ),
    ]
    assert (
        observe_and_maybe_save(
            anchored,
            tmp_path / "families",
            config=LayoutFamiliesConfig(enabled=False),
        )
        is None
    )


def test_load_families_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_families(tmp_path / "nope") == {}


def test_match_family_by_fingerprint(tmp_path: Path) -> None:
    families_dir = tmp_path / "families"
    cfg = LayoutFamiliesConfig(
        min_observations=1, min_labels=2, use_as_hint=["generated"]
    )
    anchored = [
        _anchored("mobile_no", _bbox(0.1, 0.2, 0.2, 0.24), _bbox(0.25, 0.2, 0.4, 0.24)),
        _anchored(
            "person_name",
            _bbox(0.1, 0.4, 0.2, 0.44),
            _bbox(0.25, 0.4, 0.5, 0.44),
        ),
    ]
    saved = observe_and_maybe_save(anchored, families_dir, config=cfg)
    assert saved is not None
    catalogue = load_families(families_dir)
    hit = match_family(anchored, catalogue, config=cfg)
    assert hit is not None
    assert hit.id == saved.id


def test_anchor_works_without_family_hint() -> None:
    """Hints are optional: text match + relative search alone is enough."""
    fields = [
        FieldSpec(
            key="mobile_no",
            excel_header="Mobile No",
            description="Mobile",
            type=FieldType.PHONE_IN,
            synonyms=["Mobile"],
        ),
        FieldSpec(
            key="person_name",
            excel_header="Name",
            description="Name",
            type=FieldType.NAME,
            synonyms=["Full Name"],
        ),
    ]
    regions = [
        _region("l1", _bbox(0.1, 0.2, 0.22, 0.24)),
        _region("v1", _bbox(0.28, 0.2, 0.45, 0.24), kind=RegionKind.FIELD_VALUE),
        _region("l2", _bbox(0.1, 0.4, 0.2, 0.44)),
        _region("v2", _bbox(0.28, 0.4, 0.5, 0.44), kind=RegionKind.FIELD_VALUE),
    ]
    texts = {"l1": "Mobile No", "l2": "Name", "v1": "x", "v2": "y"}
    got = anchor_fields(regions, fields, texts, family_hint=None)
    assert {a.field_key for a in got} == {"mobile_no", "person_name"}


def test_pipeline_loads_families_config() -> None:
    cfg = load_config().pipeline.layout.families
    assert cfg.enabled is True
    assert cfg.min_observations >= 1
    assert "generated" in cfg.use_as_hint


def test_family_spec_rejects_duplicate_keys() -> None:
    from pydantic import ValidationError

    from torus.config.schemas import FamilyLabelHint, FamilyPoint, FamilySize

    with pytest.raises(ValidationError):
        FormFamilySpec(
            id="fam_x",
            fingerprint="abcd",
            labels=[
                FamilyLabelHint(
                    field_key="a",
                    label_center=FamilyPoint(x=0.1, y=0.1),
                    label_size=FamilySize(w=0.1, h=0.02),
                ),
                FamilyLabelHint(
                    field_key="a",
                    label_center=FamilyPoint(x=0.2, y=0.2),
                    label_size=FamilySize(w=0.1, h=0.02),
                ),
            ],
        )
