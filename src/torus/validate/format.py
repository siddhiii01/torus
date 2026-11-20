"""P7.1 format validators for Indian identifiers and contact fields.

Only public standards live here (ID layouts, number plans, checksums). Nothing
describes a client or a document. Each rule returns ``None`` when the value is
well-formed or a short reason code; reason codes never echo the value.

Upper/lower case is ignored where the standard is case-free, because the
validator does not rewrite the value (casing is normalised in P7.5). Look-alike
characters (O/0, I/1, S/5) are *not* folded: they fail and go to review.
"""

from __future__ import annotations

import re
from datetime import date

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.validate.base import RuleValidator, strip_separators
from torus.validate.verhoeff import verhoeff_valid

_DIGITS = re.compile(r"\d+")
_PHONE_PUNCT = re.compile(r"[\s\-()]+")


def _all_digits(s: str) -> bool:
    return _DIGITS.fullmatch(s) is not None and s.isascii()


# --- phones -----------------------------------------------------------------

_MOBILE = re.compile(r"[6-9]\d{9}")
# Landline = STD code + subscriber number = 10 digits, area codes start 1-8
# (9 is the mobile range).
_LANDLINE = re.compile(r"[1-8]\d{9}")


def _strip_country_and_trunk(digits: str, *, plus: bool) -> str:
    """Drop +91 / 91 country code and the single leading trunk 0."""
    if plus:
        digits = digits.removeprefix("91")
    elif len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def _phone_digits(value: str) -> tuple[str, bool] | None:
    s = _PHONE_PUNCT.sub("", value.strip())
    plus = s.startswith("+")
    body = s[1:] if plus else s
    if not _all_digits(body):
        return None
    return body, plus


def _indian_mobile(value: str, _ctx: ValidationContext) -> str | None:
    parsed = _phone_digits(value)
    if parsed is None:
        return "not_digits"
    digits = _strip_country_and_trunk(parsed[0], plus=parsed[1])
    if len(digits) != 10:
        return "bad_length"
    if _MOBILE.fullmatch(digits) is None:
        return "bad_prefix"
    return None


def _indian_landline(value: str, _ctx: ValidationContext) -> str | None:
    parsed = _phone_digits(value)
    if parsed is None:
        return "not_digits"
    digits = _strip_country_and_trunk(parsed[0], plus=parsed[1])
    if len(digits) != 10:
        return "bad_length"
    if _LANDLINE.fullmatch(digits) is None:
        return "bad_prefix"
    return None


# --- email ------------------------------------------------------------------

_EMAIL_LOCAL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+")
_EMAIL_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
_EMAIL_TLD = re.compile(r"[A-Za-z]{2,24}")


def _email(value: str, _ctx: ValidationContext) -> str | None:
    s = value.strip()
    if s.count("@") != 1:
        return "bad_at_sign"
    local, domain = s.split("@")
    if not local or len(local) > 64 or _EMAIL_LOCAL.fullmatch(local) is None:
        return "bad_local_part"
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return "bad_local_part"
    labels = domain.split(".")
    if len(domain) > 253 or len(labels) < 2:
        return "bad_domain"
    if any(_EMAIL_LABEL.fullmatch(lab) is None for lab in labels):
        return "bad_domain"
    if _EMAIL_TLD.fullmatch(labels[-1]) is None:
        return "bad_tld"
    return None


# --- government IDs ---------------------------------------------------------

# 4th character = holder category (public PAN layout).
_PAN = re.compile(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]")


def _pan(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value).upper()
    if len(s) != 10:
        return "bad_length"
    return None if _PAN.fullmatch(s) else "bad_pattern"


def _aadhaar(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value)
    if not _all_digits(s):
        return "not_digits"
    if len(s) != 12:
        return "bad_length"
    if s[0] in "01":  # UIDAI never issues numbers starting 0 or 1
        return "bad_prefix"
    return None if verhoeff_valid(s) else "bad_checksum"


_EPIC = re.compile(r"[A-Z]{3}\d{7}")


def _epic(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value).upper()
    if len(s) != 10:
        return "bad_length"
    return None if _EPIC.fullmatch(s) else "bad_pattern"


# State code, RTO code, 4-digit issue year, 7-digit serial.
_DL = re.compile(r"[A-Z]{2}\d{2}(?:19|20)\d{2}\d{7}")


def _driving_licence(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value).upper()
    if len(s) != 15:
        return "bad_length"
    return None if _DL.fullmatch(s) else "bad_pattern"


# One series letter (never Q, X, Z) + 7 digits, first digit non-zero.
_PASSPORT = re.compile(r"[A-PR-WY][1-9]\d{6}")


def _passport(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value).upper()
    if len(s) != 8:
        return "bad_length"
    return None if _PASSPORT.fullmatch(s) else "bad_pattern"


# --- banking ----------------------------------------------------------------

_IFSC = re.compile(r"[A-Z]{4}0[A-Z0-9]{6}")


def _ifsc(value: str, _ctx: ValidationContext) -> str | None:
    s = strip_separators(value).upper()
    if len(s) != 11:
        return "bad_length"
    return None if _IFSC.fullmatch(s) else "bad_pattern"


_ACCOUNT_MIN, _ACCOUNT_MAX = 9, 18  # public range across Indian banks


def _length_bound(ctx: ValidationContext, key: str, default: int) -> int:
    raw = ctx.extras.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"validator extra {key!r} must be an integer"
        raise ConfigError(msg) from exc


def _account_number(value: str, ctx: ValidationContext) -> str | None:
    s = strip_separators(value)
    if not _all_digits(s):
        return "not_digits"
    lo = _length_bound(ctx, "min_length", _ACCOUNT_MIN)
    hi = _length_bound(ctx, "max_length", _ACCOUNT_MAX)
    if not lo <= len(s) <= hi:
        return "bad_length"
    if set(s) == {"0"}:
        return "all_zero"
    return None


# --- dates ------------------------------------------------------------------

_DATE_DMY = re.compile(r"(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})")
_DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_COMPACT = re.compile(r"(\d{2})(\d{2})(\d{4})")  # comb-box DDMMYYYY


def parse_date(value: str) -> date | str:
    """Parse day-first (Indian convention), ISO year-first or compact DDMMYYYY.

    Returns the date, or a reason code. Two-digit years are rejected as
    ambiguous. Shared with the plausibility rules (P7.4).
    """
    s = value.strip()
    if m := _DATE_ISO.fullmatch(s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := _DATE_DMY.fullmatch(s):
        d, mo, y = int(m[1]), int(m[3]), int(m[4])
    elif m := _DATE_COMPACT.fullmatch(s):
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    else:
        return "bad_format"
    try:
        return date(y, mo, d)
    except ValueError:
        return "not_a_calendar_date"


def _date(value: str, _ctx: ValidationContext) -> str | None:
    """Real calendar date. Plausibility (age, ordering) is P7.4, not format."""
    parsed = parse_date(value)
    return parsed if isinstance(parsed, str) else None


FORMAT_VALIDATORS: tuple[RuleValidator, ...] = (
    RuleValidator("indian_mobile", _indian_mobile),
    RuleValidator("indian_landline", _indian_landline),
    RuleValidator("email", _email),
    RuleValidator("pan", _pan),
    RuleValidator("aadhaar", _aadhaar),
    RuleValidator("epic", _epic),
    RuleValidator("driving_licence", _driving_licence),
    RuleValidator("passport", _passport),
    RuleValidator("ifsc", _ifsc),
    RuleValidator("account_number", _account_number),
    RuleValidator("date", _date),
)
