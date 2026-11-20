"""Component tests: label anchoring on random-layout synth pages (P4.2)."""

from __future__ import annotations

import pytest
from tests.fixtures.synth import generate_label_anchor_page

from torus.config import FieldSpec, load_config
from torus.core.types import FieldType
from torus.layout import anchor_fields, bbox_iou


def _fields_for_synth() -> list[FieldSpec]:
    """Field specs with synonyms covering the synth label variants."""
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
    """Mock printed-text reader: ground-truth label strings only."""
    out: dict[str, str] = {}
    for item in synth.regions:  # type: ignore[attr-defined]
        if str(item.region.id).startswith("label-") and item.text:
            out[item.region.id] = item.text
    return out


@pytest.mark.parametrize("seed", [1, 7, 19, 42, 99])
def test_anchor_locates_fields_on_random_seed(seed: int) -> None:
    synth = generate_label_anchor_page(seed=seed)
    cfg = load_config().pipeline.layout.anchor
    regions = [item.region for item in synth.regions]
    anchored = anchor_fields(
        regions,
        _fields_for_synth(),
        _label_texts_from_synth(synth),
        config=cfg,
    )

    expected = {row["field_key"] for row in synth.meta["fields"]}
    got = {a.field_key for a in anchored}
    assert expected <= got

    by_key = {a.field_key: a for a in anchored}
    for row in synth.meta["fields"]:
        hit = by_key[str(row["field_key"])]
        gt_value = next(
            item.region
            for item in synth.regions
            if item.region.id == row["value_region_id"]
        )
        assert bbox_iou(hit.value_region.bbox, gt_value.bbox) >= 0.5
        assert hit.value_region.label_anchor is not None
        assert hit.value_region.label_anchor.text is not None


def test_anchor_recall_across_seeds_meets_phase_bar() -> None:
    """≥97% of expected fields located across random layouts (phase acceptance)."""
    cfg = load_config().pipeline.layout.anchor
    fields = _fields_for_synth()
    matched = 0
    total = 0
    for seed in range(0, 40):
        synth = generate_label_anchor_page(seed=seed)
        regions = [item.region for item in synth.regions]
        anchored = anchor_fields(
            regions,
            fields,
            _label_texts_from_synth(synth),
            config=cfg,
        )
        by_key = {a.field_key: a for a in anchored}
        for row in synth.meta["fields"]:
            total += 1
            hit = by_key.get(str(row["field_key"]))
            if hit is None:
                continue
            gt_value = next(
                item.region
                for item in synth.regions
                if item.region.id == row["value_region_id"]
            )
            if bbox_iou(hit.value_region.bbox, gt_value.bbox) >= 0.5:
                matched += 1

    assert total > 0
    recall = matched / total
    assert recall >= 0.97, f"anchor recall {recall:.3f} < 0.97 ({matched}/{total})"


def test_starter_template_synonyms_load() -> None:
    cfg = load_config()
    mobile = cfg.template("starter_v1").field("mobile_no")
    assert mobile is not None
    assert "Mob. No." in mobile.synonyms
    assert cfg.pipeline.layout.anchor.enabled is True
