"""Shared enums and literal types used by core models and field specs."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

OrientationDegrees = Literal[0, 90, 180, 270]


class RegionKind(StrEnum):
    """Kind of layout region located on a page."""

    TEXT_LINE = "text_line"
    FIELD_VALUE = "field_value"
    COMB = "comb"
    CHECKBOX = "checkbox"
    TABLE_CELL = "table_cell"
    STAMP = "stamp"
    SIGNATURE = "signature"
    PHOTO = "photo"


class FieldStatus(StrEnum):
    """Pipeline decision for a mapped Excel column value."""

    CONFIRMED = "confirmed"
    LIKELY = "likely"
    NEEDS_REVIEW = "needs_review"
    NOT_FOUND = "not_found"


class Charset(StrEnum):
    """Expected character set for a field (from the field spec / ReadHint)."""

    DIGITS = "digits"
    LETTERS = "letters"
    ALPHANUMERIC = "alphanumeric"
    ANY = "any"


class FieldType(StrEnum):
    """Semantic field types referenced by template field specs."""

    TEXT = "text"
    NAME = "name"
    ADDRESS = "address"
    PHONE_IN = "phone_in"
    EMAIL = "email"
    DATE = "date"
    PIN = "pin"
    PAN = "pan"
    AADHAAR = "aadhaar"
    EPIC = "epic"
    ACCOUNT = "account"
    IFSC = "ifsc"
    AMOUNT = "amount"
    ENUM = "enum"
    CHECKBOX = "checkbox"
    OTHER = "other"
