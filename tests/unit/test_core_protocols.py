"""Unit tests that stub classes satisfy torus.core protocols (P0.3)."""

from __future__ import annotations

from torus.core import (
    BBox,
    Candidate,
    Document,
    Evidence,
    FieldMapper,
    FieldStatus,
    FieldValue,
    Page,
    PageClassifier,
    PageTypePrediction,
    Reader,
    ReaderCaps,
    ReadHint,
    Reading,
    Region,
    RegionKind,
    Row,
    TemplateSpec,
    ValidationContext,
    ValidationResult,
    Validator,
)


class _StubTemplate:
    def __init__(self, template_id: str) -> None:
        self._id = template_id

    @property
    def id(self) -> str:
        return self._id


class _StubReader:
    name = "stub"

    def capabilities(self) -> ReaderCaps:
        return ReaderCaps(printed=True)

    def read(self, image: object, region: Region, hint: ReadHint) -> Reading:
        _ = image, hint
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version="0.0.1",
            text="OK",
            raw_score=0.9,
        )


class _StubClassifier:
    def classify(self, page: Page) -> PageTypePrediction:
        return PageTypePrediction(
            page_type=page.page_type or "unknown",
            confidence=0.5,
        )


class _StubValidator:
    name = "non_empty"

    def check(self, value: str, ctx: ValidationContext) -> ValidationResult:
        return ValidationResult(
            validator_name=self.name,
            passed=bool(value),
            message=None if value else f"{ctx.column_key} empty",
        )


class _StubMapper:
    def map(
        self,
        doc: Document,
        candidates: list[Candidate],
        template: TemplateSpec,
    ) -> list[Row]:
        _ = doc
        if not candidates:
            return []
        cand = candidates[0]
        fv = FieldValue(
            column_key="col_a",
            value=cand.text,
            status=FieldStatus.LIKELY,
            confidence=cand.confidence,
            evidence=[
                Evidence(
                    region_id=cand.region_id,
                    page_index=0,
                    bbox=BBox(x0=0.0, y0=0.0, x1=0.1, y1=0.1),
                )
            ],
        )
        return [{fv.column_key: fv}]


def test_reader_protocol() -> None:
    reader: Reader = _StubReader()
    region = Region(
        id="r1",
        page_index=0,
        bbox=BBox(x0=0.0, y0=0.0, x1=0.2, y1=0.1),
        kind=RegionKind.FIELD_VALUE,
    )
    reading = reader.read(object(), region, ReadHint())
    assert isinstance(reader, Reader)
    assert reading.text == "OK"
    assert reader.capabilities().printed is True


def test_page_classifier_protocol() -> None:
    classifier: PageClassifier = _StubClassifier()
    page = Page(index=0, width=10, height=10, page_type="form")
    pred = classifier.classify(page)
    assert isinstance(classifier, PageClassifier)
    assert pred.page_type == "form"


def test_validator_protocol() -> None:
    validator: Validator = _StubValidator()
    ctx = ValidationContext(column_key="col_a")
    assert isinstance(validator, Validator)
    assert validator.check("x", ctx).passed is True
    assert validator.check("", ctx).passed is False


def test_field_mapper_and_template_spec_protocols() -> None:
    template = _StubTemplate("deb_v1")
    assert isinstance(template, TemplateSpec)

    mapper: FieldMapper = _StubMapper()
    doc = Document(id="d1", source_path="synth.pdf")
    candidates = [
        Candidate(
            region_id="r1", text="VALUE", agreeing_readers=["stub"], confidence=0.8
        )
    ]
    rows = mapper.map(doc, candidates, template)
    assert isinstance(mapper, FieldMapper)
    assert len(rows) == 1
    assert rows[0]["col_a"].value == "VALUE"
