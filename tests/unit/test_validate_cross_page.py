"""P7.3 cross-page agreement on synthetic observations."""

from __future__ import annotations

import pytest

from torus.core.errors import ConfigError
from torus.core.models import BBox, Evidence
from torus.validate.cross_page import (
    AgreementOutcome,
    FactKind,
    FactObservation,
    check_cross_page,
    normalise_fact,
)


def obs(
    text: str, page: int, conf: float = 0.9, rid: str | None = None
) -> FactObservation:
    return FactObservation(
        text=text,
        confidence=conf,
        evidence=Evidence(
            region_id=rid or f"r{page}-{abs(hash(text)) % 97}",
            page_index=page,
            bbox=BBox(x0=0.1, y0=0.1, x1=0.4, y1=0.2),
        ),
    )


def test_agree_across_pages_passes_validator() -> None:
    res = check_cross_page(
        "name", [obs("Test  Person", 0), obs("TEST PERSON.", 3)], FactKind.TEXT
    )
    assert res.outcome is AgreementOutcome.AGREE
    assert res.pages == [0, 3]
    vr = res.validation_result
    assert vr is not None and vr.passed and vr.message is None


def test_disagree_fails_validator_and_keeps_both_values() -> None:
    res = check_cross_page(
        "phone",
        [obs("98765 43210", 0, 0.8), obs("9876543211", 2, 0.95)],
        FactKind.DIGITS,
    )
    assert res.outcome is AgreementOutcome.DISAGREE
    vr = res.validation_result
    assert vr is not None and not vr.passed and vr.message == "cross_page_disagreement"
    assert {g.text for g in res.groups} == {"98765 43210", "9876543211"}
    assert len(res.review_evidence) == 2
    assert "9876543211" not in (vr.message or "")


def test_same_page_duplicates_are_not_independent() -> None:
    res = check_cross_page(
        "id", [obs("AB12", 1, rid="a"), obs("ab-12", 1, rid="b")], FactKind.ALNUM
    )
    assert res.outcome is AgreementOutcome.SINGLE
    assert res.validation_result is None


def test_single_observation_has_no_signal() -> None:
    res = check_cross_page("id", [obs("AB12", 0)], FactKind.ALNUM)
    assert res.outcome is AgreementOutcome.SINGLE
    assert res.validation_result is None


def test_disagreement_on_one_page_still_counts() -> None:
    res = check_cross_page(
        "id", [obs("AB12", 1, rid="a"), obs("AB13", 1, rid="b")], FactKind.ALNUM
    )
    assert res.outcome is AgreementOutcome.DISAGREE


def test_ocr_lookalikes_are_not_merged() -> None:
    res = check_cross_page("id", [obs("AB10", 0), obs("ABIO", 1)], FactKind.ALNUM)
    assert res.outcome is AgreementOutcome.DISAGREE


def test_blank_readings_ignored_and_all_blank_raises() -> None:
    res = check_cross_page(
        "n", [obs("Alpha", 0), obs("  ", 1), obs("alpha", 2)], FactKind.TEXT
    )
    assert res.outcome is AgreementOutcome.AGREE
    with pytest.raises(ConfigError):
        check_cross_page("n", [obs(" ", 0)], FactKind.TEXT)


def test_group_order_is_deterministic_most_supported_first() -> None:
    res = check_cross_page(
        "n",
        [obs("Beta", 0, 0.99), obs("Alpha", 1), obs("alpha", 2)],
        FactKind.TEXT,
    )
    assert [g.normalised for g in res.groups] == ["alpha", "beta"]


@pytest.mark.parametrize(
    ("kind", "a", "b"),
    [
        (FactKind.TEXT, "A. B-Name", "a b name"),
        (FactKind.ALNUM, "AB-12 34", "ab1234"),
        (FactKind.DIGITS, "+91 98765-43210", "919876543210"),
    ],
)
def test_normalisation_kinds(kind: FactKind, a: str, b: str) -> None:
    assert normalise_fact(a, kind) == normalise_fact(b, kind)
