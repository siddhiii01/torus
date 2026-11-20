"""P7.5 per-column normalisation (synthetic values only)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torus.config.schemas import (
    AddressSplit,
    FieldSpec,
    NameCase,
    OutputFormat,
    TemplateConfig,
)
from torus.core.types import FieldType
from torus.validate import normalise_row, normalise_value, split_address_lines


def _spec(ftype: FieldType, key: str = "col", **kw: Any) -> FieldSpec:
    return FieldSpec(
        key=key, excel_header=key, description="synthetic", type=ftype, **kw
    )


def _date_spec(**out: Any) -> FieldSpec:
    return _spec(FieldType.DATE, output=OutputFormat(**out))


# --- dates -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("05/03/1990", "1990-03-05"),
        ("5-3-1990", "1990-03-05"),
        ("05.03.1990", "1990-03-05"),
        ("1990-03-05", "1990-03-05"),
        ("05031990", "1990-03-05"),
        ("5 Mar 1990", "1990-03-05"),
        ("05-March-1990", "1990-03-05"),
        ("  05 / 03 / 1990 ", "1990-03-05"),
    ],
)
def test_date_to_iso_day_first(raw: str, expected: str) -> None:
    res = normalise_value(_spec(FieldType.DATE), raw)
    assert res.ok
    assert res.value == expected


def test_date_month_first_when_configured() -> None:
    res = normalise_value(_date_spec(date_order="mdy"), "03/05/1990")
    assert res.value == "1990-03-05"


def test_date_custom_output_format() -> None:
    res = normalise_value(_date_spec(date_format="%d/%m/%Y"), "1990-03-05")
    assert res.value == "05/03/1990"


@pytest.mark.parametrize("raw", ["31/02/1990", "00/01/1990", "13/13/1990", "abc"])
def test_date_invalid_is_left_alone_and_flagged(raw: str) -> None:
    res = normalise_value(_spec(FieldType.DATE), raw)
    assert not res.ok
    assert res.value == raw
    assert res.reason is not None
    assert raw not in res.reason


def test_two_digit_year_not_guessed_without_pivot() -> None:
    res = normalise_value(_spec(FieldType.DATE), "05/03/90")
    assert not res.ok
    assert res.value == "05/03/90"


def test_two_digit_year_with_pivot() -> None:
    spec = _date_spec(two_digit_year_pivot=30)
    assert normalise_value(spec, "05/03/90").value == "1990-03-05"
    assert normalise_value(spec, "05/03/05").value == "2005-03-05"


def test_unparseable_month_name_flagged() -> None:
    res = normalise_value(_spec(FieldType.DATE), "05 Foo 1990")
    assert not res.ok


# --- names -----------------------------------------------------------------


def test_name_default_preserves_case_collapses_space() -> None:
    res = normalise_value(_spec(FieldType.NAME), "  Asha   kumari ")
    assert res.value == "Asha kumari"
    assert res.ok


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (NameCase.UPPER, "ASHA O'NEIL-RAO"),
        (NameCase.LOWER, "asha o'neil-rao"),
        (NameCase.TITLE, "Asha O'Neil-Rao"),
        (NameCase.PRESERVE, "aSHA o'neil-RAO"),
    ],
)
def test_name_case_modes(case: NameCase, expected: str) -> None:
    spec = _spec(FieldType.NAME, output=OutputFormat(name_case=case))
    assert normalise_value(spec, "aSHA o'neil-RAO").value == expected


# --- other types -------------------------------------------------------------


def test_phone_and_pin_strip_presentation_separators() -> None:
    assert normalise_value(_spec(FieldType.PHONE_IN), "98765 43210").value == (
        "9876543210"
    )
    assert normalise_value(_spec(FieldType.PIN), "123-456").value == "123456"


def test_digit_types_do_not_map_lookalikes() -> None:
    res = normalise_value(_spec(FieldType.AADHAAR), "1234 5O78 9012")
    assert res.value == "12345O789012"


def test_id_types_upper_and_strip() -> None:
    assert normalise_value(_spec(FieldType.PAN), "abcde 1234f").value == "ABCDE1234F"
    assert normalise_value(_spec(FieldType.IFSC), "abcd0123456").value == "ABCD0123456"


def test_email_lowercased_and_stripped() -> None:
    assert normalise_value(_spec(FieldType.EMAIL), " A.B@Example.COM ").value == (
        "a.b@example.com"
    )


def test_amount_thousands_separators() -> None:
    assert normalise_value(_spec(FieldType.AMOUNT), "1,23,456.50").value == "123456.50"
    bad = normalise_value(_spec(FieldType.AMOUNT), "12x4")
    assert not bad.ok
    assert bad.value == "12x4"


def test_text_collapses_whitespace_and_zero_width() -> None:
    res = normalise_value(_spec(FieldType.TEXT), "a​  b\n c")
    assert res.value == "a b c"
    assert res.changed


def test_unchanged_value_reports_unchanged() -> None:
    res = normalise_value(_spec(FieldType.TEXT), "abc")
    assert not res.changed
    assert res.ok


def test_empty_value_ok_and_untouched() -> None:
    res = normalise_value(_spec(FieldType.DATE), "   ")
    assert res.ok
    assert res.value == ""


def test_idempotent() -> None:
    spec = _date_spec(date_format="%d/%m/%Y")
    once = normalise_value(spec, "5 Mar 1990")
    twice = normalise_value(spec, once.value)
    assert once.value == twice.value == "05/03/1990"


# --- address splitting -----------------------------------------------------


def test_split_prefers_comma_boundaries() -> None:
    lines = split_address_lines(
        "12 Alpha Road, Beta Colony, Testpur District", max_chars=20, n_lines=3
    )
    assert lines == ["12 Alpha Road,", "Beta Colony,", "Testpur District"]


def test_split_never_drops_or_cuts_words() -> None:
    text = "one two three four five six seven eight"
    lines = split_address_lines(text, max_chars=12, n_lines=4)
    assert lines is not None
    assert " ".join(lines).split() == text.split()
    assert all(len(line) <= 12 for line in lines)


def test_split_overflow_returns_none() -> None:
    assert split_address_lines("aaa bbb ccc ddd", max_chars=5, n_lines=1) is None


def test_split_word_longer_than_line_returns_none() -> None:
    assert split_address_lines("x" * 30, max_chars=10, n_lines=3) is None


def test_split_short_text_is_single_line() -> None:
    assert split_address_lines("short", max_chars=40, n_lines=3) == ["short"]


def _template(**kw: Any) -> TemplateConfig:
    fields = [_spec(FieldType.ADDRESS, key=f"address_line{i}") for i in (1, 2, 3)] + [
        _spec(FieldType.DATE, key="dob")
    ]
    return TemplateConfig(id="t", fields=fields, **kw)


def test_row_splits_long_line1_into_empty_siblings() -> None:
    tmpl = _template(
        address_split=[
            AddressSplit(
                keys=["address_line1", "address_line2", "address_line3"],
                max_chars=20,
            )
        ]
    )
    out = normalise_row(
        tmpl,
        {
            "address_line1": "12 Alpha Road, Beta Colony, Testpur District",
            "address_line2": None,
            "address_line3": "",
            "dob": "05/03/1990",
        },
    )
    assert out["address_line1"].value == "12 Alpha Road,"
    assert out["address_line2"].value == "Beta Colony,"
    assert out["address_line3"].value == "Testpur District"
    assert out["dob"].value == "1990-03-05"
    assert all(r.ok for r in out.values())


def test_row_does_not_resplit_when_siblings_already_filled() -> None:
    tmpl = _template(
        address_split=[
            AddressSplit(keys=["address_line1", "address_line2"], max_chars=10)
        ]
    )
    out = normalise_row(
        tmpl,
        {"address_line1": "A very long first line", "address_line2": "Second"},
    )
    assert out["address_line1"].value == "A very long first line"
    assert out["address_line2"].value == "Second"


def test_row_split_overflow_flags_all_keys_and_keeps_text() -> None:
    tmpl = _template(
        address_split=[
            AddressSplit(keys=["address_line1", "address_line2"], max_chars=5)
        ]
    )
    text = "alpha beta gamma delta epsilon"
    out = normalise_row(tmpl, {"address_line1": text, "address_line2": None})
    assert not out["address_line1"].ok
    assert out["address_line1"].value == text
    assert out["address_line2"].value == ""


def test_row_skips_missing_and_unknown_keys() -> None:
    out = normalise_row(_template(), {"dob": "05/03/1990", "nope": "x"})
    assert set(out) == {"dob"}


def test_row_none_value_stays_empty() -> None:
    out = normalise_row(_template(), {"dob": None})
    assert out["dob"].value == ""
    assert out["dob"].ok


# --- config validation -----------------------------------------------------


def test_bad_date_format_rejected() -> None:
    with pytest.raises(ValidationError):
        OutputFormat(date_format="no directives")


def test_address_split_keys_must_exist_in_template() -> None:
    with pytest.raises(ValidationError):
        _template(address_split=[AddressSplit(keys=["a", "b"], max_chars=10)])


def test_address_split_needs_two_keys() -> None:
    with pytest.raises(ValidationError):
        AddressSplit(keys=["only"], max_chars=10)
