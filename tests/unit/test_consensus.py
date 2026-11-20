"""P6.1 consensus: alignment, weighted voting, weight fitting (synthetic strings)."""

from __future__ import annotations

import pytest

from torus.consensus import (
    ReaderWeights,
    WeightSample,
    align_texts,
    edit_distance,
    fit_reader_weights,
    vote,
)
from torus.core.models import Reading


def _r(name: str, text: str, **kw: object) -> Reading:
    return Reading(
        region_id="r1",
        reader_name=name,
        reader_version="1",
        text=text,
        **kw,  # type: ignore[arg-type]
    )


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


@pytest.mark.parametrize(
    "texts",
    [["12345", "12345"], ["12345", "1245"], ["abc", "xabcx", "bc"], ["a", "", "bb"]],
)
def test_alignment_is_lossless_and_rectangular(texts: list[str]) -> None:
    rows = align_texts(texts)
    assert len({len(r) for r in rows}) == 1
    for row, text in zip(rows, texts, strict=True):
        assert "".join(c for c in row if c is not None) == text


def test_alignment_places_deletion_as_gap() -> None:
    rows = align_texts(["12345", "1245"])
    assert rows[1] == ["1", "2", None, "4", "5"]


def test_majority_corrects_substitution_and_deletion() -> None:
    sub = vote("r1", [_r("a", "AB1234"), _r("b", "AB1234"), _r("c", "AB1284")])
    assert sub.candidate.text == "AB1234"
    assert sub.dissenting_readers == ["c"]
    dele = vote("r1", [_r("a", "AB1234"), _r("b", "AB1234"), _r("c", "AB124")])
    assert dele.candidate.text == "AB1234"
    assert dele.min_column_agreement == pytest.approx(2 / 3)


def test_unanimous() -> None:
    res = vote("r1", [_r("a", "9876"), _r("b", "9876")])
    assert res.candidate.text == "9876"
    assert res.candidate.confidence == 1.0
    assert res.min_column_agreement == 1.0
    assert res.dissenting_readers == []


def test_weights_break_a_two_way_split() -> None:
    readings = [_r("strong", "7"), _r("weak", "1")]
    res = vote("r1", readings, weights={"strong": 3.0, "weak": 1.0})
    assert res.candidate.text == "7"
    assert res.candidate.confidence == pytest.approx(0.75)
    assert res.dissenting_readers == ["weak"]


def test_tie_is_deterministic_regardless_of_input_order() -> None:
    a = vote("r1", [_r("a", "1"), _r("b", "2")])
    b = vote("r1", [_r("b", "2"), _r("a", "1")])
    assert a.candidate.text == b.candidate.text
    assert a.min_column_agreement == pytest.approx(0.5)


def test_errors_and_empty_readings_abstain() -> None:
    res = vote(
        "r1",
        [_r("a", "55"), _r("b", "", error="timeout"), _r("c", "")],
    )
    assert res.n_readings == 1
    assert res.candidate.text == "55"


def test_no_usable_readings_is_empty_zero_confidence() -> None:
    res = vote("r1", [_r("a", "", error="boom")])
    assert res.candidate.text == ""
    assert res.candidate.confidence == 0.0
    assert res.n_readings == 0


def test_char_confidences_can_overrule_a_vote_split() -> None:
    sure = _r("a", "5", char_confidences=[0.99])
    unsure = _r("b", "6", char_confidences=[0.10])
    assert vote("r1", [sure, unsure]).candidate.text == "5"
    # With char confidence ignored it is a tie, broken deterministically.
    flat = vote("r1", [sure, unsure], use_char_confidence=False)
    assert flat.min_column_agreement == pytest.approx(0.5)


def test_mismatched_char_confidence_length_is_ignored() -> None:
    bad = _r("a", "55", char_confidences=[0.1], raw_score=0.9)
    res = vote("r1", [bad, _r("b", "55")])
    assert res.candidate.text == "55"


def test_agreeing_families_counts_distinct_families() -> None:
    res = vote(
        "r1",
        [_r("p1", "42"), _r("p2", "42"), _r("t1", "42"), _r("v1", "43")],
        families={"p1": "paddle", "p2": "paddle", "t1": "tesseract", "v1": "vlm"},
    )
    assert res.agreeing_families == ["paddle", "tesseract"]
    assert res.dissenting_readers == ["v1"]


def test_fit_weights_rank_accurate_reader_higher() -> None:
    samples = [
        WeightSample(
            field_type="pin",
            gold="123456",
            readings=[_r("good", "123456"), _r("bad", "123459")],
        )
        for _ in range(5)
    ]
    weights = fit_reader_weights(samples)
    w = weights.for_field("pin")
    assert w["good"] > w["bad"] > 0.0
    assert weights.for_field("name") == {}


def test_fit_skips_errors_and_small_samples() -> None:
    samples = [
        WeightSample("pin", "12", [_r("a", "12"), _r("b", "", error="x")]),
    ]
    assert fit_reader_weights(samples).by_field_type == {}
    big = [
        WeightSample("pin", "123456", [_r("a", "123456"), _r("b", "", error="x")])
    ] * 5
    assert set(fit_reader_weights(big).for_field("pin")) == {"a"}


def test_fitted_weights_drive_vote_per_field_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    weights = ReaderWeights(
        by_field_type={"pin": {"a": 5.0, "b": 1.0}, "name": {"a": 1.0, "b": 5.0}}
    )
    readings = [_r("a", "1"), _r("b", "2")]
    assert vote("r1", readings, weights=weights, field_type="pin").candidate.text == "1"
    assert (
        vote("r1", readings, weights=weights, field_type="name").candidate.text == "2"
    )

    path = tmp_path / "w.json"
    weights.save(path)
    assert ReaderWeights.load(path) == weights
