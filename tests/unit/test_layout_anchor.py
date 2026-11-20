"""Unit tests for label matching, embeddings, and relative value search (P4.2)."""

from __future__ import annotations

from torus.config.schemas import FieldSpec, LabelAnchorConfig
from torus.core import BBox, Region, RegionKind
from torus.core.types import FieldType
from torus.layout import (
    CharNgramEmbedder,
    LabelMatchKind,
    RelativePlacement,
    find_value_near_label,
    match_label_to_fields,
    normalize_label,
)


def _spec(
    key: str,
    header: str,
    *,
    synonyms: list[str] | None = None,
) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=header,
        description=f"Synthetic field {key}",
        type=FieldType.TEXT,
        synonyms=synonyms or [],
    )


def test_normalize_label_strips_punct_and_case() -> None:
    assert normalize_label("Mob. No.") == "mob no"
    assert normalize_label("  PIN-Code  ") == "pin code"


def test_exact_header_match() -> None:
    fields = [_spec("mobile_no", "Mobile No", synonyms=["Cell"])]
    hit = match_label_to_fields("Mobile No:", fields)
    assert hit is not None
    assert hit.field_key == "mobile_no"
    assert hit.kind is LabelMatchKind.EXACT
    assert hit.score == 1.0


def test_synonym_match() -> None:
    fields = [_spec("mobile_no", "Mobile No", synonyms=["Mob. No.", "Cell"])]
    hit = match_label_to_fields("Mob. No.", fields)
    assert hit is not None
    assert hit.field_key == "mobile_no"
    assert hit.kind is LabelMatchKind.SYNONYM


def test_embedding_near_miss() -> None:
    fields = [_spec("mobile_no", "Mobile No", synonyms=["Mobile"])]
    embedder = CharNgramEmbedder()
    # Close enough to "Mobile No" without being an exact synonym.
    hit = match_label_to_fields(
        "Mobile Number",
        fields,
        embedder=embedder,
        embedding_min=0.55,
    )
    assert hit is not None
    assert hit.field_key == "mobile_no"
    assert hit.kind is LabelMatchKind.EMBEDDING
    assert hit.score >= 0.55


def test_no_match_below_threshold() -> None:
    fields = [_spec("mobile_no", "Mobile No")]
    hit = match_label_to_fields(
        "Date of Birth",
        fields,
        embedder=CharNgramEmbedder(),
        embedding_min=0.95,
    )
    assert hit is None


def test_find_value_right_of_label() -> None:
    label = BBox(x0=0.05, y0=0.20, x1=0.18, y1=0.24)
    value = Region(
        id="v1",
        page_index=0,
        bbox=BBox(x0=0.22, y0=0.20, x1=0.40, y1=0.24),
        kind=RegionKind.TEXT_LINE,
    )
    decoy = Region(
        id="d1",
        page_index=0,
        bbox=BBox(x0=0.05, y0=0.50, x1=0.20, y1=0.54),
        kind=RegionKind.TEXT_LINE,
    )
    found = find_value_near_label(label, [value, decoy])
    assert found is not None
    region, placement = found
    assert region.id == "v1"
    assert placement is RelativePlacement.RIGHT


def test_find_value_below_label() -> None:
    cfg = LabelAnchorConfig(prefer=["below", "right", "same_row"])
    label = BBox(x0=0.10, y0=0.30, x1=0.25, y1=0.34)
    value = Region(
        id="v2",
        page_index=0,
        bbox=BBox(x0=0.10, y0=0.36, x1=0.35, y1=0.40),
        kind=RegionKind.TEXT_LINE,
    )
    found = find_value_near_label(label, [value], config=cfg)
    assert found is not None
    region, placement = found
    assert region.id == "v2"
    assert placement is RelativePlacement.BELOW


def test_find_value_respects_exclude_ids() -> None:
    label = BBox(x0=0.05, y0=0.20, x1=0.18, y1=0.24)
    value = Region(
        id="v1",
        page_index=0,
        bbox=BBox(x0=0.22, y0=0.20, x1=0.40, y1=0.24),
        kind=RegionKind.TEXT_LINE,
    )
    found = find_value_near_label(label, [value], exclude_ids=frozenset({"v1"}))
    assert found is None


def test_read_region_texts_with_mock_reader() -> None:
    from torus.core.models import ReaderCaps, ReadHint, Reading
    from torus.layout import read_region_texts

    class _MockReader:
        name = "mock_print"

        def capabilities(self) -> ReaderCaps:
            return ReaderCaps(printed=True)

        def read(self, image: object, region: Region, hint: ReadHint) -> Reading:
            return Reading(
                region_id=region.id,
                reader_name=self.name,
                reader_version="0",
                text="Cell",
                raw_score=1.0,
                time_ms=0.0,
            )

    region = Region(
        id="lab1",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.15),
        kind=RegionKind.TEXT_LINE,
    )
    texts = read_region_texts(object(), [region], _MockReader())
    assert texts == {"lab1": "Cell"}
