"""P7.4 plausibility validators: age range, date order, "must differ" rules.

Format validators say a value is well-formed; these say it is believable next to
the rest of the row. Column keys and bounds come from the field spec / template
config (never from code), so the pipeline builds them at start-up with the
``make_*`` factories and registers them under per-column names.

Rules never repair or reinterpret a value. A value that cannot be parsed fails
(``bad_format``) and goes to review. Reasons are codes only and never echo the
value. A *reference* column that is absent on the row is not a conflict, so the
rule passes; a reference that is present but unreadable fails closed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.validate.base import RuleValidator, strip_separators
from torus.validate.format import parse_date

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _present(ctx: ValidationContext, key: str) -> str | None:
    v = ctx.row_values.get(key)
    return v if v is not None and v.strip() else None


def _age_on(born: date, as_of: date) -> int:
    return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))


def make_age_range_validator(
    name: str,
    *,
    min_age: int = 0,
    max_age: int = 120,
    today: Callable[[], date] = date.today,
) -> RuleValidator:
    """Value is a date of birth; age today must lie in [min_age, max_age].

    ``today`` is injectable so evaluation runs and tests are reproducible.
    """
    if not 0 <= min_age <= max_age:
        raise ConfigError("age range needs 0 <= min_age <= max_age")

    def rule(value: str, ctx: ValidationContext) -> str | None:
        born = parse_date(value)
        if isinstance(born, str):
            return born
        now = today()
        if born > now:
            return "date_in_future"
        age = _age_on(born, now)
        if age < min_age:
            return "age_below_minimum"
        if age > max_age:
            return "age_above_maximum"
        return None

    return RuleValidator(name, rule)


def make_date_order_validator(
    name: str,
    *,
    reference_key: str,
    must_be: str = "after",
    allow_equal: bool = False,
    today: Callable[[], date] = date.today,
) -> RuleValidator:
    """This column's date must be ``after`` or ``before`` the reference column's.

    Example (config-driven): application date after date of birth. Dates later
    than today are rejected for ``after`` rules, as a recorded event cannot be
    in the future.
    """
    if must_be not in ("after", "before"):
        raise ConfigError("must_be is 'after' or 'before'")

    def rule(value: str, ctx: ValidationContext) -> str | None:
        mine = parse_date(value)
        if isinstance(mine, str):
            return mine
        raw = _present(ctx, reference_key)
        if must_be == "after" and mine > today():
            return "date_in_future"
        if raw is None:
            return None
        ref = parse_date(raw)
        if isinstance(ref, str):
            return "reference_unreadable"
        if mine == ref:
            return None if allow_equal else "date_order"
        ok = mine > ref if must_be == "after" else mine < ref
        return None if ok else "date_order"

    return RuleValidator(name, rule)


def _canonical(value: str) -> str:
    """Compare ignoring presentation only (separators, case, +91/trunk 0)."""
    s = _NON_ALNUM.sub("", strip_separators(value).casefold())
    if s.isdigit():
        if len(s) == 12 and s.startswith("91"):
            s = s[2:]
        elif len(s) == 11 and s.startswith("0"):
            s = s[1:]
    return s


def make_differs_from_validator(
    name: str, *, other_keys: Sequence[str]
) -> RuleValidator:
    """Value must not equal any of the other columns (phone vs account number).

    A phone number is not an account number: when OCR copies one into the other
    the pair is equal, which is never a legitimate answer.
    """
    if not other_keys:
        raise ConfigError("differs_from needs at least one other column key")
    keys = tuple(other_keys)

    def rule(value: str, ctx: ValidationContext) -> str | None:
        mine = _canonical(value)
        if not mine:
            return "empty_value"
        for key in keys:
            raw = _present(ctx, key)
            if raw is not None and _canonical(raw) == mine:
                return "duplicates_other_field"
        return None

    return RuleValidator(name, rule)
