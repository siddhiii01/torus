"""P7.4 plausibility validators; all dates and numbers are made up."""

from __future__ import annotations

from datetime import date

import pytest

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.validate import (
    make_age_range_validator,
    make_date_order_validator,
    make_differs_from_validator,
    register_validator,
    run_validators,
)

TODAY = date(2026, 6, 15)


def today() -> date:
    return TODAY


def ctx(column: str = "col", **row: str | None) -> ValidationContext:
    return ValidationContext(column_key=column, row_values=row)


# --- age --------------------------------------------------------------------


def test_age_in_range_passes() -> None:
    v = make_age_range_validator("age_ok", min_age=18, max_age=100, today=today)
    assert v.check("15/06/2008", ctx()).passed  # exactly 18 today
    assert v.check("01-01-1980", ctx()).passed
    assert v.check("1980-01-01", ctx()).passed


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("16/06/2008", "age_below_minimum"),  # turns 18 tomorrow
        ("01/01/1900", "age_above_maximum"),
        ("01/01/2030", "date_in_future"),
        ("31/02/1990", "not_a_calendar_date"),
        ("garbage", "bad_format"),
        ("", "bad_format"),
    ],
)
def test_age_failures(value: str, reason: str) -> None:
    v = make_age_range_validator("age_ok", min_age=18, max_age=100, today=today)
    res = v.check(value, ctx())
    assert not res.passed
    assert res.message == reason


def test_age_leap_day_birthday() -> None:
    v = make_age_range_validator("age_ok", min_age=18, max_age=100, today=today)
    assert v.check("29/02/2008", ctx()).passed  # 18 by June 2026


def test_age_bad_config() -> None:
    with pytest.raises(ConfigError):
        make_age_range_validator("x", min_age=50, max_age=10)
    with pytest.raises(ConfigError):
        make_age_range_validator("x", min_age=-1)


# --- date order -------------------------------------------------------------


def test_date_after_reference() -> None:
    v = make_date_order_validator("app_after_dob", reference_key="dob", today=today)
    assert v.check("01/01/2020", ctx(dob="05/05/1990")).passed


@pytest.mark.parametrize(
    ("value", "dob", "reason"),
    [
        ("01/01/1980", "05/05/1990", "date_order"),
        ("05/05/1990", "05/05/1990", "date_order"),  # equal not allowed by default
        ("01/01/2020", "nonsense", "reference_unreadable"),
        ("01/01/2020", "30/02/1990", "reference_unreadable"),
        ("01/01/2030", "05/05/1990", "date_in_future"),
        ("x", "05/05/1990", "bad_format"),
    ],
)
def test_date_after_failures(value: str, dob: str, reason: str) -> None:
    v = make_date_order_validator("app_after_dob", reference_key="dob", today=today)
    res = v.check(value, ctx(dob=dob))
    assert not res.passed
    assert res.message == reason


def test_date_order_missing_reference_is_not_a_conflict() -> None:
    v = make_date_order_validator("app_after_dob", reference_key="dob", today=today)
    assert v.check("01/01/2020", ctx()).passed
    assert v.check("01/01/2020", ctx(dob=None)).passed
    assert v.check("01/01/2020", ctx(dob="  ")).passed


def test_date_order_allow_equal_and_before() -> None:
    eq = make_date_order_validator(
        "same_day_ok", reference_key="dob", allow_equal=True, today=today
    )
    assert eq.check("05/05/1990", ctx(dob="05/05/1990")).passed
    before = make_date_order_validator(
        "dob_before_app", reference_key="app", must_be="before", today=today
    )
    assert before.check("05/05/1990", ctx(app="01/01/2020")).passed
    assert not before.check("05/05/2021", ctx(app="01/01/2020")).passed
    # Future dates are only rejected for "after" rules.
    assert before.check("01/01/2000", ctx(app="01/01/2030")).passed


def test_date_order_bad_config() -> None:
    with pytest.raises(ConfigError):
        make_date_order_validator("x", reference_key="k", must_be="sideways")


# --- differs from -----------------------------------------------------------


def test_differs_passes_when_distinct_or_absent() -> None:
    v = make_differs_from_validator("phone_not_acct", other_keys=["account"])
    assert v.check("9123456780", ctx(account="123456789012")).passed
    assert v.check("9123456780", ctx()).passed
    assert v.check("9123456780", ctx(account=None)).passed


@pytest.mark.parametrize(
    "phone",
    ["9123456780", "91234 56780", "+91 91234-56780", "09123456780", "919123456780"],
)
def test_differs_catches_presentation_variants(phone: str) -> None:
    v = make_differs_from_validator("phone_not_acct", other_keys=["account"])
    res = v.check(phone, ctx(account="9123456780"))
    assert not res.passed
    assert res.message == "duplicates_other_field"


def test_differs_multiple_keys_and_empty_value() -> None:
    v = make_differs_from_validator("p", other_keys=["a", "b"])
    assert not v.check("5551234567", ctx(a="1", b="555 123 4567")).passed
    res = v.check("  - ", ctx())
    assert not res.passed
    assert res.message == "empty_value"


def test_differs_does_not_fold_lookalikes() -> None:
    v = make_differs_from_validator("p", other_keys=["a"])
    assert v.check("O123456789", ctx(a="0123456789")).passed


def test_differs_bad_config() -> None:
    with pytest.raises(ConfigError):
        make_differs_from_validator("x", other_keys=[])


# --- registry / privacy -------------------------------------------------------


def test_registered_by_name_and_runs_via_registry() -> None:
    register_validator(
        make_age_range_validator("p74_age", min_age=18, today=today), replace=True
    )
    results = run_validators(["p74_age"], "01/01/1990", ctx())
    assert [r.passed for r in results] == [True]
    failed = run_validators(["p74_age"], "01/01/2020", ctx())[0]
    assert not failed.passed


def test_reasons_never_echo_value() -> None:
    v = make_differs_from_validator("p", other_keys=["a"])
    res = v.check("9123456780", ctx(a="9123456780"))
    assert "9123456780" not in (res.message or "")
