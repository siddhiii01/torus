"""Unit tests for torus.eval scoring (P1.2) — synthetic expected vs predicted."""

from __future__ import annotations

from pathlib import Path

import pytest

from torus.config import load_template
from torus.config.schemas import (
    FieldSpec,
    MatchRule,
    NormalizeKind,
    RowKeyConfig,
    RowKeyPart,
    TemplateConfig,
)
from torus.core import BBox, Evidence, FieldStatus, FieldValue
from torus.core.types import Charset, FieldType
from torus.eval import (
    aggregate_scores,
    align_rows,
    norm_address_source,
    norm_date,
    score_expected,
    score_rows,
    values_match,
)
from torus.eval.schema import ExpectedDocument

_REPO = Path(__file__).resolve().parents[2]
_STARTER = _REPO / "config" / "templates" / "starter_v1.yaml"
_DEB = _REPO / "config" / "templates" / "deb_v1.yaml"


def _evidence() -> Evidence:
    return Evidence(
        region_id="r1",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.4, y1=0.2),
    )


def _fv(
    key: str,
    value: str | None,
    status: FieldStatus,
    *,
    confidence: float = 0.9,
) -> FieldValue:
    evidence = [_evidence()] if value else []
    return FieldValue(
        column_key=key,
        value=value,
        status=status,
        confidence=confidence,
        evidence=evidence,
    )


def _mini_template() -> TemplateConfig:
    return TemplateConfig(
        id="mini_score_v1",
        version=1,
        description="Synthetic scoring template",
        row_key=RowKeyConfig(
            primary=[
                RowKeyPart(sources=["person_name"], normalize=NormalizeKind.PERSON),
                RowKeyPart(
                    sources=["address_type"], normalize=NormalizeKind.ADDRESS_TYPE
                ),
                RowKeyPart(
                    sources=["address_source"],
                    normalize=NormalizeKind.ADDRESS_SOURCE,
                    aliases={
                        "applicationform": ["application form", "Application Form"],
                        "voterid": ["voter id", "Voter ID"],
                    },
                ),
            ],
            disambiguator=RowKeyPart(
                sources=["address_line1"], normalize=NormalizeKind.TEXT
            ),
            fuzzy_threshold=0.8,
        ),
        fields=[
            FieldSpec(
                key="person_name",
                excel_header="Name",
                description="Person name",
                type=FieldType.NAME,
                charset=Charset.LETTERS,
                match=MatchRule.NORMALISED,
                critical=True,
            ),
            FieldSpec(
                key="address_type",
                excel_header="Type",
                description="Address type code",
                type=FieldType.ENUM,
                match=MatchRule.NORMALISED,
                normalize=NormalizeKind.ADDRESS_TYPE,
                critical=True,
            ),
            FieldSpec(
                key="address_source",
                excel_header="Source",
                description="Address source label",
                type=FieldType.ENUM,
                match=MatchRule.NORMALISED,
                normalize=NormalizeKind.ADDRESS_SOURCE,
                critical=True,
            ),
            FieldSpec(
                key="address_line1",
                excel_header="Line1",
                description="Address line 1",
                type=FieldType.ADDRESS,
                match=MatchRule.FUZZY,
            ),
            FieldSpec(
                key="mobile_no",
                excel_header="Mobile",
                description="Mobile number",
                type=FieldType.PHONE_IN,
                charset=Charset.DIGITS,
                length=10,
                match=MatchRule.NORMALISED,
                critical=True,
            ),
            FieldSpec(
                key="ops_note",
                excel_header="Ops",
                description="Passthrough ops column",
                type=FieldType.TEXT,
                match=MatchRule.EXACT,
                passthrough=True,
            ),
        ],
    )


def test_starter_template_loads_with_row_key() -> None:
    template = load_template(_STARTER)
    assert template.row_key is not None
    assert template.row_key.primary[0].sources == ["person_name"]
    assert template.field("mobile_no") is not None
    assert template.field("mobile_no").match is MatchRule.NORMALISED


def test_deb_template_row_key_and_scored_fields() -> None:
    template = load_template(_DEB)
    assert template.row_key is not None
    assert len(template.row_key.primary) == 3
    assert template.row_key.disambiguator is not None
    scored = {f.key for f in template.scored_fields()}
    assert "person_name" in scored
    assert "bank_lan" not in scored  # passthrough
    addr = template.field("address_source")
    assert addr is not None
    assert addr.normalize is NormalizeKind.ADDRESS_SOURCE


def test_norm_date_and_address_source_aliases() -> None:
    assert norm_date("16-Mar-1971") == "1971-03-16"
    assert norm_date("16/03/1971") == "1971-03-16"
    aliases = {"applicationform": ["application form", "Application Form"]}
    assert norm_address_source("Application Form", aliases=aliases) == "applicationform"
    assert norm_address_source("unknown-doc", aliases=aliases) == "unknown_doc"


def test_match_rules_exact_normalised_fuzzy() -> None:
    phone = FieldSpec(
        key="mobile_no",
        excel_header="M",
        description="m",
        type=FieldType.PHONE_IN,
        match=MatchRule.NORMALISED,
    )
    assert values_match(phone, "98765 43210", "9876543210")
    assert not values_match(
        FieldSpec(
            key="mobile_no",
            excel_header="M",
            description="m",
            type=FieldType.PHONE_IN,
            match=MatchRule.EXACT,
        ),
        "98765 43210",
        "9876543210",
    )
    addr = FieldSpec(
        key="address_line1",
        excel_header="A",
        description="a",
        type=FieldType.ADDRESS,
        match=MatchRule.FUZZY,
    )
    # Overlap is |pred ∩ gold| / |gold|; pred must cover gold tokens.
    assert values_match(
        addr,
        "12 market street near park",
        "12 market street",
        fuzzy_threshold=0.8,
    )
    assert not values_match(
        addr,
        "12 market street",
        "12 market street near park",
        fuzzy_threshold=0.8,
    )


def test_align_rows_primary_and_disambiguator() -> None:
    template = _mini_template()
    gold = [
        {
            "person_name": "Ada Lovelace",
            "address_type": "H1",
            "address_source": "Application Form",
            "address_line1": "10 Synthetic Road",
            "mobile_no": "9000000001",
        },
        {
            "person_name": "Ada Lovelace",
            "address_type": "H1",
            "address_source": "Application Form",
            "address_line1": "20 Other Lane",
            "mobile_no": "9000000002",
        },
    ]
    pred = [
        {
            "person_name": "ada lovelace",
            "address_type": "H1",
            "address_source": "applicationform",
            "address_line1": "20 other lane",
            "mobile_no": "9000000002",
        },
        {
            "person_name": "Ada Lovelace",
            "address_type": "H1",
            "address_source": "Application Form",
            "address_line1": "10 Synthetic Road",
            "mobile_no": "9000000001",
        },
    ]
    result = align_rows(gold, pred, template.row_key)
    assert len(result.pairs) == 2
    assert len(result.unmatched_gold) == 0
    assert len(result.unmatched_pred) == 0
    # First pred (line 20) pairs with second gold
    assert result.pairs[0][0]["mobile_no"] == "9000000002"
    assert result.pairs[1][0]["mobile_no"] == "9000000001"


def test_score_rows_accuracy_and_passthrough_ignored() -> None:
    template = _mini_template()
    gold = [
        {
            "person_name": "Synth Person",
            "address_type": "H1",
            "address_source": "Voter ID",
            "address_line1": "1 Test Avenue",
            "mobile_no": "9876543210",
            "ops_note": "ignore-me",
        }
    ]
    pred = [
        {
            "person_name": "synth person",
            "address_type": "H1",
            "address_source": "voterid",
            "address_line1": "1 Test Avenue Block A",
            "mobile_no": "98765-43210",
            "ops_note": "wrong-ops",  # passthrough — not scored
        }
    ]
    score = score_rows(gold, pred, template, case_id="synth_01")
    assert score.rows.aligned == 1
    assert score.fields_micro.accuracy == 1.0
    assert score.silent_errors == 0
    assert "ops_note" not in score.per_field
    assert score.per_field["mobile_no"].correct == 1


def test_silent_errors_and_status_rates() -> None:
    template = _mini_template()
    gold = [
        {
            "person_name": "Synth Person",
            "address_type": "H1",
            "address_source": "applicationform",
            "address_line1": "9 Road",
            "mobile_no": "9111111111",
        }
    ]
    pred = [
        {
            "person_name": _fv("person_name", "Synth Person", FieldStatus.CONFIRMED),
            "address_type": _fv("address_type", "H1", FieldStatus.CONFIRMED),
            "address_source": _fv(
                "address_source", "applicationform", FieldStatus.CONFIRMED
            ),
            "address_line1": _fv(
                "address_line1", "9 Road", FieldStatus.LIKELY, confidence=0.7
            ),
            # Wrong value but confirmed → silent error
            "mobile_no": _fv("mobile_no", "9000000000", FieldStatus.CONFIRMED),
        }
    ]
    score = score_rows(gold, pred, template, case_id="synth_silent")
    assert score.silent_errors == 1
    assert score.per_field["mobile_no"].silent_errors == 1
    assert score.status_rates.total >= 1
    assert score.status_rates.counts[FieldStatus.CONFIRMED.value] >= 1
    assert 0.0 <= score.status_rates.automatic_rate <= 1.0
    # Confirmed wrong + other confirmed/likely → automatic_rate is confirmed share
    assert score.status_rates.automatic_rate == pytest.approx(
        score.status_rates.counts[FieldStatus.CONFIRMED.value]
        / score.status_rates.total
    )


def test_score_expected_document() -> None:
    template = load_template(_STARTER)
    expected = ExpectedDocument(
        template_id="starter_v1",
        case_id="case_a",
        rows=[
            {
                "person_name": "Synthetic A",
                "mobile_no": "9876543210",
                "address_pincode": "560001",
                "notes": "hello world",
            }
        ],
    )
    pred = [
        {
            "person_name": "synthetic a",
            "mobile_no": "9876543210",
            "address_pincode": "560001",
            "notes": "hello world street",
        }
    ]
    score = score_expected(expected, pred, template)
    assert score.case_id == "case_a"
    assert score.rows.aligned == 1
    assert score.fields_micro.correct == score.fields_micro.compared
    assert score.silent_errors == 0


def test_unmatched_confirmed_pred_is_silent_error() -> None:
    template = _mini_template()
    gold = [
        {
            "person_name": "Alpha",
            "address_type": "H1",
            "address_source": "dl",
            "address_line1": "1 A",
            "mobile_no": "9000000001",
        }
    ]
    pred = [
        {
            "person_name": _fv("person_name", "Beta", FieldStatus.CONFIRMED),
            "address_type": _fv("address_type", "W", FieldStatus.CONFIRMED),
            "address_source": _fv("address_source", "dpn", FieldStatus.CONFIRMED),
            "address_line1": _fv("address_line1", "2 B", FieldStatus.CONFIRMED),
            "mobile_no": _fv("mobile_no", "9000000009", FieldStatus.CONFIRMED),
        }
    ]
    score = score_rows(gold, pred, template)
    assert score.rows.aligned == 0
    assert score.silent_errors >= 1


def test_aggregate_scores() -> None:
    template = _mini_template()
    gold = [
        {
            "person_name": "One",
            "address_type": "H1",
            "address_source": "dl",
            "address_line1": "1",
            "mobile_no": "9000000001",
        }
    ]
    pred_ok = [
        {
            "person_name": "One",
            "address_type": "H1",
            "address_source": "dl",
            "address_line1": "1",
            "mobile_no": "9000000001",
        }
    ]
    pred_bad = [
        {
            "person_name": "One",
            "address_type": "H1",
            "address_source": "dl",
            "address_line1": "1",
            "mobile_no": "9111111111",
        }
    ]
    s1 = score_rows(gold, pred_ok, template, case_id="a")
    s2 = score_rows(gold, pred_bad, template, case_id="b")
    agg = aggregate_scores([s1, s2])
    assert agg.case_id == "aggregate"
    assert agg.rows.gold == 2
    assert agg.per_field["mobile_no"].compared == 2
    assert agg.per_field["mobile_no"].correct == 1


def test_align_by_index_when_no_row_key() -> None:
    template = TemplateConfig(
        id="positional_v1",
        fields=[
            FieldSpec(
                key="person_name",
                excel_header="N",
                description="n",
                type=FieldType.NAME,
                match=MatchRule.EXACT,
            )
        ],
        row_key=None,
    )
    gold = [{"person_name": "A"}, {"person_name": "B"}]
    pred = [{"person_name": "A"}]
    score = score_rows(gold, pred, template)
    assert score.rows.aligned == 1
    assert score.rows.unmatched_gold == 1
