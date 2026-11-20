"""Unit tests for torus.core pydantic models (P0.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from torus.core import (
    BBox,
    Candidate,
    Charset,
    Document,
    Evidence,
    EvidenceRequiredError,
    ExtractionResult,
    FieldStatus,
    FieldValue,
    ImageNotLoadedError,
    LabelAnchor,
    Page,
    PageTypePrediction,
    ReaderCaps,
    ReadHint,
    Reading,
    Region,
    RegionKind,
    ValidationContext,
    ValidationResult,
)


def _bbox() -> BBox:
    return BBox(x0=0.1, y0=0.2, x1=0.4, y1=0.35)


def _evidence(*, region_id: str = "r1") -> Evidence:
    return Evidence(region_id=region_id, page_index=0, bbox=_bbox())


def test_bbox_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=-0.1, y0=0.0, x1=0.5, y1=0.5)


def test_bbox_rejects_unordered() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=0.5, y0=0.1, x1=0.2, y1=0.4)


def test_page_lazy_image_loader() -> None:
    page = Page(index=0, width=100, height=200)
    assert page.has_image is False
    with pytest.raises(ImageNotLoadedError):
        _ = page.image

    calls = {"n": 0}

    def loader() -> str:
        calls["n"] += 1
        return "pixels"

    page.bind_image_loader(loader)
    assert page.has_image is True
    assert page.image == "pixels"
    assert page.image == "pixels"
    assert calls["n"] == 1


def test_page_bind_image() -> None:
    page = Page(index=1, width=10, height=10, orientation=90, rotated=True)
    page.bind_image(object())
    assert page.has_image is True
    assert page.image is not None


def test_document_round_trip() -> None:
    page = Page(index=0, width=800, height=1100, page_type="application_form")
    doc = Document(
        id="abc123",
        source_path="/tmp/synth.pdf",
        pages=[page],
        metadata={"source": "synth"},
    )
    restored = Document.model_validate(doc.model_dump())
    assert restored.id == "abc123"
    assert restored.pages[0].page_type == "application_form"
    assert restored.metadata["source"] == "synth"


def test_region_with_label_anchor() -> None:
    region = Region(
        id="reg-1",
        page_index=0,
        bbox=_bbox(),
        kind=RegionKind.COMB,
        label_anchor=LabelAnchor(
            text="PIN", bbox=BBox(x0=0.05, y0=0.2, x1=0.09, y1=0.3)
        ),
    )
    assert region.kind == RegionKind.COMB
    assert region.label_anchor is not None
    assert region.label_anchor.text == "PIN"


def test_reading_soft_error() -> None:
    reading = Reading(
        region_id="r1",
        reader_name="stub",
        reader_version="0.0.1",
        text="",
        error="model unavailable",
        time_ms=1.5,
    )
    assert reading.error is not None
    assert reading.text == ""


def test_reading_rejects_bad_char_confidence() -> None:
    with pytest.raises(ValidationError):
        Reading(
            region_id="r1",
            reader_name="stub",
            reader_version="1",
            text="A",
            char_confidences=[1.2],
        )


def test_candidate_and_read_hint() -> None:
    candidate = Candidate(
        region_id="r1",
        text="110001",
        agreeing_readers=["a", "b"],
        confidence=0.91,
    )
    hint = ReadHint(charset=Charset.DIGITS, expected_length=6, language="en")
    assert candidate.agreeing_readers == ["a", "b"]
    assert hint.charset == Charset.DIGITS


def test_field_value_requires_evidence_when_value_present() -> None:
    with pytest.raises(EvidenceRequiredError):
        FieldValue(
            column_key="pin",
            value="110001",
            status=FieldStatus.CONFIRMED,
            confidence=0.99,
            evidence=[],
        )


def test_field_value_allows_empty_without_evidence() -> None:
    fv = FieldValue(
        column_key="pin",
        value=None,
        status=FieldStatus.NOT_FOUND,
        confidence=0.0,
        reason="no region located",
    )
    assert fv.evidence == []


def test_field_value_with_evidence() -> None:
    fv = FieldValue(
        column_key="pin",
        value="110001",
        status=FieldStatus.CONFIRMED,
        confidence=0.95,
        evidence=[_evidence()],
        validator_results=[
            ValidationResult(validator_name="pin_lookup", passed=True),
        ],
    )
    assert fv.status == FieldStatus.CONFIRMED
    assert len(fv.evidence) == 1


def test_extraction_result() -> None:
    fv = FieldValue(
        column_key="pin",
        value="110001",
        status=FieldStatus.LIKELY,
        confidence=0.7,
        evidence=[_evidence()],
    )
    result = ExtractionResult(
        document_id="abc123",
        template_id="deb_v1",
        rows=[{"pin": fv}],
        timings_ms={"total": 12.0},
        model_versions={"stub": "0.0.1"},
        config_hash="deadbeef",
    )
    assert result.rows[0]["pin"].column_key == "pin"
    assert result.config_hash == "deadbeef"


def test_validation_context_and_page_type_prediction() -> None:
    ctx = ValidationContext(
        column_key="pin",
        row_values={"state": "XX"},
        document_id="abc",
        page_types=("application_form",),
    )
    pred = PageTypePrediction(
        page_type="application_form",
        confidence=0.88,
        alternatives=[("other", 0.1)],
    )
    caps = ReaderCaps(scripts=["latn"], handwriting=True, per_char_confidence=True)
    assert ctx.column_key == "pin"
    assert pred.page_type == "application_form"
    assert caps.handwriting is True
