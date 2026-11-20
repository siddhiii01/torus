"""P7.2 lookup validators on a synthetic PIN master (made-up PIN/place data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.validate import get_validator
from torus.validate.lookup import (
    LazyLookups,
    load_pin_master,
    load_state_list,
    make_pin_exists_validator,
    make_pin_location_validator,
    make_state_validator,
    normalise_place,
)

MASTER_CSV = (
    "pincode,office,district,state\n"
    "123456,Alpha Nagar,Testpur,Odisha\n"
    "123456,Beta Colony,Testpur,Odisha\n"
    "234567,Gamma Road,Sampleganj,Tamil Nadu\n"
)


@pytest.fixture
def lookups(tmp_path: Path) -> LazyLookups:
    (tmp_path / "india_post_pincodes.csv").write_text(MASTER_CSV, encoding="utf-8")
    real = load_state_list()
    (tmp_path / "india_states_uts.txt").write_text(
        "\n".join(sorted(real.official)), encoding="utf-8"
    )
    (tmp_path / "india_state_aliases.csv").write_text(
        "alias,official\norissa,Odisha\n", encoding="utf-8"
    )
    return LazyLookups(directory=tmp_path)


def ctx(**row: str | None) -> ValidationContext:
    return ValidationContext(column_key="pin", row_values=row)


def test_pin_exists_pass_and_fail(lookups: LazyLookups) -> None:
    v = make_pin_exists_validator(lookups)
    assert v.check("123456", ctx()).passed
    assert v.check("123 456", ctx()).passed  # presentation grouping only
    r = v.check("999999", ctx())
    assert not r.passed and r.message == "pin_not_found"


@pytest.mark.parametrize("bad", ["", "12345", "1234567", "012345", "12345O", "abcdef"])
def test_pin_format_failures_never_repaired(lookups: LazyLookups, bad: str) -> None:
    r = make_pin_exists_validator(lookups).check(bad, ctx())
    assert not r.passed and r.message == "pin_format"


def test_missing_master_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "india_states_uts.txt").write_text("Odisha\n", encoding="utf-8")
    missing = LazyLookups(directory=tmp_path)
    r = make_pin_exists_validator(missing).check("123456", ctx())
    assert not r.passed and r.message == "lookup_unavailable"
    loc = make_pin_location_validator(missing, state_key="s", city_key=None)
    assert loc.check("123456", ctx(s="Odisha")).message == "lookup_unavailable"


def test_state_closed_list_and_aliases(lookups: LazyLookups) -> None:
    v = make_state_validator(lookups)
    assert v.check("  tamil   NADU ", ctx()).passed
    assert v.check("Orissa", ctx()).passed
    assert v.check("Dadra & Nagar Haveli and Daman & Diu", ctx()).passed
    r = v.check("Atlantis", ctx())
    assert not r.passed and r.message == "state_not_in_closed_list"
    assert not v.check("", ctx()).passed


def test_location_agreement(lookups: LazyLookups) -> None:
    v = make_pin_location_validator(lookups, state_key="state", city_key="city")
    assert v.check("123456", ctx(state="Odisha", city="Testpur")).passed
    assert v.check("123456", ctx(state="Orissa", city="alpha nagar")).passed
    assert v.check("123456", ctx(state="Odisha", city=None)).passed
    assert v.check("123456", ctx(state=None, city="Testpur")).passed


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"state": "Tamil Nadu", "city": "Testpur"}, "pin_state_mismatch"),
        ({"state": "Odisha", "city": "Sampleganj"}, "pin_city_mismatch"),
        ({"state": "Atlantis", "city": "Testpur"}, "state_not_in_closed_list"),
        ({"state": "", "city": "  "}, "location_missing"),
        ({}, "location_missing"),
    ],
)
def test_location_disagreement_fails(
    lookups: LazyLookups, row: dict[str, str], reason: str
) -> None:
    v = make_pin_location_validator(lookups, state_key="state", city_key="city")
    r = v.check("123456", ctx(**row))
    assert not r.passed and r.message == reason


def test_location_unknown_pin(lookups: LazyLookups) -> None:
    v = make_pin_location_validator(lookups, state_key="state", city_key=None)
    assert v.check("999999", ctx(state="Odisha")).message == "pin_not_found"


def test_location_requires_a_key(lookups: LazyLookups) -> None:
    with pytest.raises(ConfigError):
        make_pin_location_validator(lookups, state_key=None, city_key=None)


def test_messages_never_echo_value(lookups: LazyLookups) -> None:
    r = make_pin_exists_validator(lookups).check("999999", ctx())
    assert "999999" not in (r.message or "")


def test_master_rejects_untrusted_rows(tmp_path: Path) -> None:
    states = load_state_list()
    bad_pin = tmp_path / "a.csv"
    bad_pin.write_text("pincode,office,district,state\n12,x,y,Odisha\n", "utf-8")
    with pytest.raises(ConfigError):
        load_pin_master(bad_pin, states)
    bad_state = tmp_path / "b.csv"
    bad_state.write_text("pincode,office,district,state\n123456,x,y,Nowhere\n", "utf-8")
    with pytest.raises(ConfigError):
        load_pin_master(bad_state, states)
    bad_cols = tmp_path / "c.csv"
    bad_cols.write_text("pincode,state\n123456,Odisha\n", "utf-8")
    with pytest.raises(ConfigError):
        load_pin_master(bad_cols, states)


def test_repo_state_list_is_closed_set() -> None:
    states = load_state_list()
    assert len(states.official) == 36
    assert states.canonical("Orissa") == "odisha"


def test_normalise_place() -> None:
    assert normalise_place(" A  &  B. ") == "a and b"


def test_registered_by_name() -> None:
    assert get_validator("indian_pin").name == "indian_pin"
    assert get_validator("indian_state").name == "indian_state"
