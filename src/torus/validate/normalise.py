"""Per-column output normalisation (P7.5), driven by the template.

Normalisation changes *presentation* only: separators, whitespace, letter case,
date layout, address line breaks. It never repairs a value. Anything that cannot
be re-formatted without guessing (an impossible date, a two-digit year with no
configured pivot, an address that does not fit its lines) is returned exactly as
read with ``ok=False`` and a reason code, so the caller can send it to review.

Reasons never echo the value (document text must not reach logs or errors).

Run normalisation once per value: a date written in a day/month order that
differs from ``date_order`` would be re-read differently on a second pass.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from torus.config.schemas import (
    DateOrder,
    FieldSpec,
    NameCase,
    OutputFormat,
    TemplateConfig,
)
from torus.core.types import FieldType
from torus.validate.base import strip_separators

_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_LETTER_RUN = re.compile(r"[^\W\d_]+")

_ISO_DATE = re.compile(r"(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})")
_NUM_DATE = re.compile(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2}|\d{4})")
_COMPACT_DATE = re.compile(r"(\d{2})(\d{2})(\d{4})")
_NAMED_DATE = re.compile(r"(\d{1,2})[\s\-./]+([A-Za-z]+)[\s\-./,]*(\d{2}|\d{4})")
_AMOUNT = re.compile(r"\d+(?:\.\d+)?")

# Public calendar names; "sept" is a common abbreviation.
_MONTHS: dict[str, int] = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}

_SEPARATOR_DIGIT_TYPES = frozenset(
    {FieldType.PHONE_IN, FieldType.PIN, FieldType.AADHAAR, FieldType.ACCOUNT}
)
_UPPER_ID_TYPES = frozenset({FieldType.PAN, FieldType.EPIC, FieldType.IFSC})


@dataclass(frozen=True)
class NormalisationResult:
    """Outcome of normalising one value.

    ``value`` is the normalised text when ``ok`` and the cleaned original when
    not. ``reason`` is a short code, never the value itself.
    """

    value: str
    changed: bool
    ok: bool = True
    reason: str | None = None


def _clean(text: str) -> str:
    """NFC, drop zero-width characters, collapse every whitespace run to one space."""
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


def _title(text: str) -> str:
    return _LETTER_RUN.sub(lambda m: m.group(0).capitalize(), text)


def _apply_name_case(text: str, case: NameCase) -> str:
    if case is NameCase.UPPER:
        return text.upper()
    if case is NameCase.LOWER:
        return text.lower()
    if case is NameCase.TITLE:
        return _title(text)
    return text


def _expand_year(raw: str, pivot: int | None) -> int | None:
    year = int(raw)
    if len(raw) == 4:
        return year
    if pivot is None:
        return None
    return 2000 + year if year < pivot else 1900 + year


def _parse_date(text: str, out: OutputFormat) -> tuple[date | None, str | None]:
    """Parse *text* into a date, or return a reason code. Never guesses."""
    day: int
    month: int
    year_raw: str
    iso = _ISO_DATE.fullmatch(text)
    named = _NAMED_DATE.fullmatch(text)
    numeric = _NUM_DATE.fullmatch(text) or _COMPACT_DATE.fullmatch(text)
    if iso:
        year_raw, month, day = iso[1], int(iso[2]), int(iso[3])
    elif named:
        found = _MONTHS.get(named[2].lower())
        if found is None:
            return None, "date_unknown_month_name"
        day, month, year_raw = int(named[1]), found, named[3]
    elif numeric:
        first, second = int(numeric[1]), int(numeric[2])
        day, month = (
            (second, first) if out.date_order is DateOrder.MDY else (first, second)
        )
        year_raw = numeric[3]
    else:
        return None, "date_unrecognised_format"
    year = _expand_year(year_raw, out.two_digit_year_pivot)
    if year is None:
        return None, "date_two_digit_year_ambiguous"
    try:
        return date(year, month, day), None
    except ValueError:
        return None, "date_invalid"


def _normalise_date(cleaned: str, out: OutputFormat) -> tuple[str, str | None]:
    parsed, reason = _parse_date(cleaned, out)
    if parsed is None:
        return cleaned, reason
    return parsed.strftime(out.date_format), None


def _normalise_amount(cleaned: str) -> tuple[str, str | None]:
    compact = cleaned.replace(",", "").replace(" ", "")
    if _AMOUNT.fullmatch(compact):
        return compact, None
    return cleaned, "amount_not_numeric"


def normalise_value(spec: FieldSpec, value: str) -> NormalisationResult:
    """Normalise *value* for the column *spec* describes."""
    cleaned = _clean(value)
    if not cleaned:
        return NormalisationResult(value="", changed=value != "")
    ftype = spec.type
    reason: str | None = None
    result = cleaned
    if ftype is FieldType.DATE:
        result, reason = _normalise_date(cleaned, spec.output)
    elif ftype is FieldType.NAME:
        result = _apply_name_case(cleaned, spec.output.name_case)
    elif ftype in _SEPARATOR_DIGIT_TYPES:
        result = strip_separators(cleaned)
    elif ftype in _UPPER_ID_TYPES:
        result = strip_separators(cleaned).upper()
    elif ftype is FieldType.EMAIL:
        result = _WS.sub("", cleaned).lower()
    elif ftype is FieldType.AMOUNT:
        result, reason = _normalise_amount(cleaned)
    return NormalisationResult(
        value=result, changed=result != value, ok=reason is None, reason=reason
    )


def split_address_lines(text: str, *, max_chars: int, n_lines: int) -> list[str] | None:
    """Re-flow *text* over at most *n_lines* lines of at most *max_chars*.

    Breaks fall between words, preferring just after a comma. No word is cut and
    no text is dropped: when it does not fit, returns ``None``.
    """
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_chars:
            return None
        while current and len(" ".join([*current, word])) > max_chars:
            cut = _break_after(current)
            lines.append(" ".join(current[:cut]))
            current = current[cut:]
        current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines if len(lines) <= n_lines else None


def _break_after(words: list[str]) -> int:
    """Index to cut at: after the last comma-ended word, else keep all words."""
    for i in range(len(words) - 2, -1, -1):
        if words[i].endswith(","):
            return i + 1
    return len(words)


def normalise_row(
    template: TemplateConfig, values: Mapping[str, str | None]
) -> dict[str, NormalisationResult]:
    """Normalise one output row; keys the template does not define are skipped.

    An ``address_split`` group re-flows its first line over the group's columns,
    but only when the other columns are empty: a row that already carries
    separate lines is never re-cut.
    """
    results: dict[str, NormalisationResult] = {}
    for key, raw in values.items():
        spec = template.field(key)
        if spec is not None:
            results[key] = normalise_value(spec, raw or "")
    for group in template.address_split:
        _apply_split(group.keys, group.max_chars, values, results)
    return results


def _apply_split(
    keys: list[str],
    max_chars: int,
    values: Mapping[str, str | None],
    results: dict[str, NormalisationResult],
) -> None:
    first = results.get(keys[0])
    if first is None or len(first.value) <= max_chars or not first.ok:
        return
    if any((values.get(k) or "").strip() for k in keys[1:]):
        return
    lines = split_address_lines(first.value, max_chars=max_chars, n_lines=len(keys))
    if lines is None:
        results[keys[0]] = NormalisationResult(
            value=first.value,
            changed=first.changed,
            ok=False,
            reason="address_overflow",
        )
        return
    for i, key in enumerate(keys):
        line = lines[i] if i < len(lines) else ""
        if key in results or line:
            results[key] = NormalisationResult(value=line, changed=True)
