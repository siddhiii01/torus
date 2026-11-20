"""P7.1 format validators. Made-up values only; never real identifiers."""

from __future__ import annotations

import pytest

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.core.protocols import Validator
from torus.validate import (
    available_validators,
    get_validator,
    run_validators,
    verhoeff_check_digit,
    verhoeff_valid,
)

CTX = ValidationContext(column_key="c")


def _aadhaar(prefix11: str) -> str:
    return prefix11 + str(verhoeff_check_digit(prefix11))


def _passes(name: str, value: str, ctx: ValidationContext = CTX) -> bool:
    return get_validator(name).check(value, ctx).passed


GOOD_AADHAAR = _aadhaar("23456789012")


def test_verhoeff_published_vector() -> None:
    assert verhoeff_check_digit("236") == 3
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2364")
    assert not verhoeff_valid("")
    assert not verhoeff_valid("23a3")


def test_verhoeff_detects_every_single_digit_error() -> None:
    for pos in range(12):
        for d in "0123456789":
            if d != GOOD_AADHAAR[pos]:
                bad = GOOD_AADHAAR[:pos] + d + GOOD_AADHAAR[pos + 1 :]
                assert not verhoeff_valid(bad)


def test_verhoeff_detects_adjacent_transposition() -> None:
    s = GOOD_AADHAAR
    for i in range(11):
        if s[i] != s[i + 1]:
            assert not verhoeff_valid(s[:i] + s[i + 1] + s[i] + s[i + 2 :])


CASES: dict[str, tuple[list[str], list[str]]] = {
    "indian_mobile": (
        ["9876543210", "+91 98765 43210", "919876543210", "09876543210", "6000000001"],
        [
            "5876543210",  # bad first digit
            "987654321",  # 9 digits
            "98765432100",  # 11 digits, no trunk 0
            "98765 4321O",  # letter O instead of zero: never folded
            "",
            "abcdefghij",
        ],
    ),
    "indian_landline": (
        [
            "02212345678",
            "022-2345 6789",
            "(0124) 2345678",
            "+91 22 12345678",
            "2212345678",
        ],
        [
            "12345678",  # no STD code
            "9876543210",  # mobile range
            "0221234567",  # 10 digits with trunk 0 → only 9 significant
            "022 1234 567O",
            "",
        ],
    ),
    "email": (
        ["a.b@example.com", "x_y+z@mail.example.co.in", " a@b.io "],
        [
            "a@b",
            "a@@b.com",
            "a b@example.com",
            ".a@example.com",
            "a..b@example.com",
            "a@-example.com",
            "a@example.c",
            "a@example.c0m",
            "",
        ],
    ),
    "pan": (
        ["ABCPE1234F", "abcpe1234f", "ABCPE 1234 F"],
        [
            "ABCZE1234F",  # invalid holder category
            "ABCPE12345",  # last must be a letter
            "ABCPE1234",
            "ABCPEI234F",  # I instead of 1
            "0BCPE1234F",
        ],
    ),
    "aadhaar": (
        [GOOD_AADHAAR, f"{GOOD_AADHAAR[:4]} {GOOD_AADHAAR[4:8]} {GOOD_AADHAAR[8:]}"],
        [
            GOOD_AADHAAR[:-1] + str((int(GOOD_AADHAAR[-1]) + 1) % 10),  # bad checksum
            GOOD_AADHAAR[:11],
            GOOD_AADHAAR + "0",
            _aadhaar("03456789012"),  # valid checksum, forbidden first digit
            _aadhaar("13456789012"),
            GOOD_AADHAAR.replace("0", "O"),
            "",
        ],
    ),
    "epic": (
        ["ABC1234567", "abc1234567"],
        ["AB12345678", "ABC123456", "ABC12345678", "ABC123456O", ""],
    ),
    "driving_licence": (
        ["AB0120151234567", "AB01-2015-1234567", "ab01 2015 1234567"],
        [
            "AB0130151234567",  # year outside 19xx/20xx
            "AB012015123456",
            "A00120151234567",
            "AB01201512345678",
            "",
        ],
    ),
    "passport": (
        ["A1234567", "p7654321", "A 1234567"],
        ["Q1234567", "X1234567", "Z1234567", "A0234567", "A123456", "12345678", ""],
    ),
    "ifsc": (
        ["ABCD0123456", "abcd0a1b2c3", "ABCD 0123456"],
        ["ABCD1123456", "ABC00123456", "ABCD012345", "ABCD01234567", "AB1D0123456", ""],
    ),
    "account_number": (
        ["123456789", "123456789012345678", "1234 5678 9012", "100000000"],
        ["12345678", "1234567890123456789", "000000000", "12345678A", ""],
    ),
    "date": (
        [
            "01/02/2000",
            "1-2-2000",
            "29.02.2000",
            "01022000",
            "2000-02-29",
            " 31/12/1999 ",
        ],
        [
            "29/02/2001",  # not a leap year
            "31/04/2000",
            "00/01/2000",
            "13/13/2000",
            "01/02/00",  # two-digit year is ambiguous
            "01/02-2000",  # mixed separators
            "2000/02/01",
            "0102200O",
            "",
        ],
    ),
}


@pytest.mark.parametrize(
    ("name", "value"),
    [(n, v) for n, (good, _) in CASES.items() for v in good],
)
def test_valid_values_pass(name: str, value: str) -> None:
    result = get_validator(name).check(value, CTX)
    assert result.passed, result.message
    assert result.message is None
    assert result.validator_name == name


@pytest.mark.parametrize(
    ("name", "value"),
    [(n, v) for n, (_, bad) in CASES.items() for v in bad],
)
def test_invalid_values_fail_with_reason(name: str, value: str) -> None:
    result = get_validator(name).check(value, CTX)
    assert not result.passed
    assert result.message


def test_reason_never_echoes_value() -> None:
    secret = "ZZ-private-9988"
    for name in CASES:
        result = get_validator(name).check(secret, CTX)
        assert secret not in (result.message or "")


def test_account_length_bounds_from_extras() -> None:
    ctx = ValidationContext(
        column_key="c", extras={"min_length": "11", "max_length": "12"}
    )
    assert not _passes("account_number", "1234567890", ctx)
    assert _passes("account_number", "12345678901", ctx)
    assert _passes("account_number", "123456789012", ctx)
    assert not _passes("account_number", "1234567890123", ctx)


def test_account_bad_bound_is_config_error() -> None:
    ctx = ValidationContext(column_key="c", extras={"min_length": "ten"})
    with pytest.raises(ConfigError):
        get_validator("account_number").check("123456789", ctx)


def test_validators_do_not_mutate_or_fold_lookalikes() -> None:
    # A passing PAN is reported as passing, but the validator returns no new
    # text; O-for-0 confusion fails instead of being repaired.
    assert not _passes("aadhaar", "2345 6789 O123")
    assert not _passes("account_number", "12345678O")


def test_registry_has_all_p71_validators_and_protocol() -> None:
    expected = {
        "indian_mobile",
        "indian_landline",
        "email",
        "pan",
        "aadhaar",
        "epic",
        "driving_licence",
        "passport",
        "ifsc",
        "account_number",
        "date",
    }
    assert expected <= set(available_validators())
    for name in expected:
        assert isinstance(get_validator(name), Validator)


def test_run_validators_in_order_and_unknown_name_raises() -> None:
    results = run_validators(["pan", "epic"], "ABCPE1234F", CTX)
    assert [r.validator_name for r in results] == ["pan", "epic"]
    assert [r.passed for r in results] == [True, False]
    with pytest.raises(ConfigError):
        run_validators(["no_such_validator"], "x", CTX)


def test_register_duplicate_rejected() -> None:
    from torus.validate import register_validator

    with pytest.raises(ConfigError):
        register_validator(get_validator("pan"))
