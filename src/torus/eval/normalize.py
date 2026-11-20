"""Value normalisation helpers for evaluation scoring and row alignment."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from torus.config.schemas import NormalizeKind
from torus.core.types import FieldType
from torus.eval.schema import ExpectedCell

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_KEEP_SPACE = re.compile(r"[^a-z0-9\s]")
_DIGITS = re.compile(r"\D")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_ISO_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_DMY = re.compile(r"^(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})$")
_DMON_Y = re.compile(r"^(\d{1,2})[\s\-./]+([A-Za-z]{3,})[\s\-./]+(\d{2,4})$")

_MONTHS: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Field types that compare as digits-only under match=normalised.
_DIGIT_TYPES: frozenset[FieldType] = frozenset(
    {
        FieldType.PHONE_IN,
        FieldType.PIN,
        FieldType.AADHAAR,
        FieldType.ACCOUNT,
        FieldType.AMOUNT,
    }
)

# Field types that strip to alphanumeric under match=normalised.
_ALNUM_TYPES: frozenset[FieldType] = frozenset(
    {
        FieldType.PAN,
        FieldType.EPIC,
        FieldType.IFSC,
    }
)


def is_empty(value: ExpectedCell | Any) -> bool:
    """True when the cell is missing or blank after string conversion."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


def as_text(value: ExpectedCell | Any) -> str:
    """Stringify a scalar cell; empty for None."""
    if value is None:
        return ""
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())[:10]
    return str(value)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def norm_text(value: ExpectedCell | Any) -> str:
    return collapse_ws(as_text(value)).casefold()


def norm_person(value: ExpectedCell | Any) -> str:
    text = norm_text(value)
    text = _NON_ALNUM_KEEP_SPACE.sub(" ", text)
    return collapse_ws(text)


def norm_address_type(value: ExpectedCell | Any) -> str:
    return _WS.sub("", as_text(value)).upper()


def digits_only(value: ExpectedCell | Any) -> str:
    return _DIGITS.sub("", as_text(value))


def norm_alnum(value: ExpectedCell | Any) -> str:
    return _NON_ALNUM.sub("", norm_text(value))


def norm_date(value: ExpectedCell | Any) -> str:
    if is_empty(value):
        return ""
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())[:10]
    text = as_text(value).strip()
    match = _ISO_DATE.match(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = _ISO_PREFIX.match(text)
    if match:
        return match.group(1)
    match = _DMY.match(text)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return f"{year:04d}-{month:02d}-{day:02d}"
    match = _DMON_Y.match(text)
    if match:
        day = int(match.group(1))
        month_num = _MONTHS.get(match.group(2)[:3].lower())
        year = int(match.group(3))
        if month_num is not None:
            if year < 100:
                year += 2000 if year < 50 else 1900
            return f"{year:04d}-{month_num:02d}-{day:02d}"
    return norm_text(text)


def _build_alias_lookup(
    aliases: Mapping[str, list[str]] | None,
) -> dict[str, str]:
    """Map every accepted surface form (and the canonical) → canonical."""
    lookup: dict[str, str] = {}
    if not aliases:
        return lookup
    for canonical, forms in aliases.items():
        canon = collapse_ws(canonical).casefold()
        lookup[canon] = canon
        for form in forms:
            lookup[collapse_ws(form).casefold()] = canon
    return lookup


def norm_address_source(
    value: ExpectedCell | Any,
    *,
    aliases: Mapping[str, list[str]] | None = None,
) -> str:
    """Canonicalise an address-source label using optional template aliases."""
    raw = collapse_ws(as_text(value)).casefold()
    if not raw:
        return ""
    lookup = _build_alias_lookup(aliases)
    if raw in lookup:
        return lookup[raw]
    cleaned = collapse_ws(_NON_ALNUM.sub(" ", raw))
    if cleaned in lookup:
        return lookup[cleaned]
    # Unknown sources: casefold + collapse separators to underscores.
    return cleaned.replace(" ", "_")


def apply_kind(
    kind: NormalizeKind,
    value: ExpectedCell | Any,
    *,
    aliases: Mapping[str, list[str]] | None = None,
) -> str:
    """Apply a named normalisation kind from row_key / scoring config."""
    if kind is NormalizeKind.NONE:
        return as_text(value).strip()
    if kind is NormalizeKind.TEXT:
        return norm_text(value)
    if kind is NormalizeKind.PERSON:
        return norm_person(value)
    if kind is NormalizeKind.ADDRESS_TYPE:
        return norm_address_type(value)
    if kind is NormalizeKind.ADDRESS_SOURCE:
        return norm_address_source(value, aliases=aliases)
    if kind is NormalizeKind.DIGITS:
        return digits_only(value)
    if kind is NormalizeKind.ALNUM:
        return norm_alnum(value)
    if kind is NormalizeKind.DATE:
        return norm_date(value)
    return norm_text(value)


def kind_for_field_type(field_type: FieldType) -> NormalizeKind:
    """Default normalisation kind for ``match=normalised`` from field type."""
    if field_type in _DIGIT_TYPES:
        return NormalizeKind.DIGITS
    if field_type in _ALNUM_TYPES:
        return NormalizeKind.ALNUM
    if field_type is FieldType.DATE:
        return NormalizeKind.DATE
    if field_type is FieldType.NAME:
        return NormalizeKind.PERSON
    if field_type is FieldType.EMAIL:
        return NormalizeKind.TEXT
    return NormalizeKind.TEXT


def tokens(text: str) -> set[str]:
    return {part for part in _NON_ALNUM.split(text) if part}
