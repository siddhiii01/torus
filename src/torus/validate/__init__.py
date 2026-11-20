"""Validators: format, checksum, lookup, cross-page agreement."""

from __future__ import annotations

from torus.validate.base import RuleValidator, strip_separators
from torus.validate.cross_page import (
    CrossPageResult,
    FactKind,
    FactObservation,
    check_cross_page,
)
from torus.validate.normalise import (
    NormalisationResult,
    normalise_row,
    normalise_value,
    split_address_lines,
)
from torus.validate.plausibility import (
    make_age_range_validator,
    make_date_order_validator,
    make_differs_from_validator,
)
from torus.validate.registry import (
    available_validators,
    get_validator,
    register_validator,
    run_validators,
)
from torus.validate.verhoeff import verhoeff_check_digit, verhoeff_valid

__all__ = [
    "CrossPageResult",
    "FactKind",
    "FactObservation",
    "NormalisationResult",
    "check_cross_page",
    "normalise_row",
    "normalise_value",
    "split_address_lines",
    "RuleValidator",
    "available_validators",
    "get_validator",
    "make_age_range_validator",
    "make_date_order_validator",
    "make_differs_from_validator",
    "register_validator",
    "run_validators",
    "strip_separators",
    "verhoeff_check_digit",
    "verhoeff_valid",
]
