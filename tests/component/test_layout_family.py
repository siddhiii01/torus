"""Component tests: form-family auto-save on random-layout synth pages (P4.7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.synth import generate_label_anchor_page

from torus.config import FieldSpec, FormFamilyStatus, LayoutFamiliesConfig, load_config
from torus.core.types import FieldType
from torus.layout import (
    anchor_fields,
    fingerprint_anchored,
    match_family,
    observe_and_maybe_save,
)


def _fields_for_synth() -> list[FieldSpec]:
    return [
        FieldSpec(
            key="mobile_no",
            excel_header="Mobile No",
            description="Primary mobile number",
            type=FieldType.PHONE_IN,
            synonyms=["Mobile", "Mob. No.", "Cell"],
        ),
        FieldSpec(
            key="address_pincode",
            excel_header="PIN",
            description="Postal PIN",
            type=FieldType.PIN,
            synonyms=["PIN Code", "Pincode", "Zip"],
        ),
        FieldSpec(
            key="person_name",
            excel_header="Name",
            description="Person name",
            type=FieldType.NAME,
            synonyms=["Full Name", "Applicant Name"],
        ),
        FieldSpec(
            key="notes",
            excel_header="Notes",
            description="Free notes",
            type=FieldType.TEXT,
            synonyms=["Remark", "Comments"],
        ),
    ]


def _label_texts_from_synth(synth: object) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in synth.regions:  # type: ignore[attr-defined]
        if str(item.region.id).startswith("label-") and item.text:
            out[item.region.id] = item.text
    return out


def _anchor_synth(seed: int, *, field_count: int = 3):
    synth = generate_label_anchor_page(seed=seed, field_count=field_count)
    cfg = load_config().pipeline.layout.anchor
    regions = [item.region for item in synth.regions]
    anchored = anchor_fields(
        regions,
        _fields_for_synth(),
        _label_texts_from_synth(synth),
        config=cfg,
    )
    return synth, anchored


@pytest.mark.parametrize("seed", [1, 7, 19, 42, 99])
def test_anchor_without_family_on_random_seeds(seed: int) -> None:
    """No family file needed: random layouts still locate fields (no fixed coords)."""
    _synth, anchored = _anchor_synth(seed, field_count=3)
    assert len(anchored) >= 2


def test_repeated_same_seed_auto_saves_generated(tmp_path: Path) -> None:
    families_dir = tmp_path / "families"
    fam_cfg = LayoutFamiliesConfig(min_observations=2, min_labels=2)
    _synth, anchored = _anchor_synth(seed=11, field_count=3)
    assert len(anchored) >= 2

    first = observe_and_maybe_save(anchored, families_dir, config=fam_cfg)
    assert first is not None
    assert first.status is FormFamilyStatus.OBSERVING

    # Same layout again (identical seed → identical relative geometry).
    _synth2, anchored2 = _anchor_synth(seed=11, field_count=3)
    assert fingerprint_anchored(anchored) == fingerprint_anchored(anchored2)

    second = observe_and_maybe_save(anchored2, families_dir, config=fam_cfg)
    assert second is not None
    assert second.status is FormFamilyStatus.GENERATED
    assert second.observation_count == 2
    assert (families_dir / f"{second.id}.yaml").is_file()


def test_different_seeds_different_fingerprints() -> None:
    _a, anchored_a = _anchor_synth(seed=3, field_count=3)
    _b, anchored_b = _anchor_synth(seed=4, field_count=3)
    if len(anchored_a) < 2 or len(anchored_b) < 2:
        pytest.skip("synth did not anchor enough fields")
    assert fingerprint_anchored(anchored_a) != fingerprint_anchored(anchored_b)


def test_hint_does_not_change_required_correctness(tmp_path: Path) -> None:
    """With and without a matched family, the same fields are found."""
    fam_cfg = LayoutFamiliesConfig(
        min_observations=1,
        min_labels=2,
        use_as_hint=["generated"],
    )
    families_dir = tmp_path / "families"
    _synth, anchored = _anchor_synth(seed=21, field_count=3)
    assert len(anchored) >= 2
    saved = observe_and_maybe_save(anchored, families_dir, config=fam_cfg)
    assert saved is not None

    catalogue = {saved.id: saved}
    hint = match_family(anchored, catalogue, config=fam_cfg)
    assert hint is not None

    synth = generate_label_anchor_page(seed=21, field_count=3)
    regions = [item.region for item in synth.regions]
    texts = _label_texts_from_synth(synth)
    anchor_cfg = load_config().pipeline.layout.anchor

    without = anchor_fields(
        regions, _fields_for_synth(), texts, config=anchor_cfg, family_hint=None
    )
    with_hint = anchor_fields(
        regions,
        _fields_for_synth(),
        texts,
        config=anchor_cfg,
        family_hint=hint,
        family_match_tolerance=fam_cfg.match_tolerance,
    )
    assert {a.field_key for a in without} == {a.field_key for a in with_hint}
