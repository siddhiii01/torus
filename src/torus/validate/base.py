"""Shared validator plumbing.

Validators check; they never rewrite or "repair" a value (AGENTS.md §3.4).
Normalisation helpers here only strip *presentation* separators so that a value
printed in groups ("1234 5678 9012") can be checked. They never map look-alike
characters (O/0, I/1): an OCR confusion must fail and go to review.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from torus.core.models import ValidationContext, ValidationResult

# Result of a rule: None when the value is fine, else a short reason code.
# Reasons never echo the value (document content must not reach logs/errors).
Rule = Callable[[str, ValidationContext], str | None]

_SEPARATORS = re.compile(r"[\s\-]+")


def strip_separators(value: str) -> str:
    """Remove whitespace and hyphens (presentation grouping only)."""
    return _SEPARATORS.sub("", value)


@dataclass
class RuleValidator:
    """Adapts a pure rule function to the ``Validator`` protocol.

    Not frozen: the protocol declares ``name`` as a settable attribute.
    """

    name: str
    rule: Rule

    def check(self, value: str, ctx: ValidationContext) -> ValidationResult:
        reason = self.rule(value, ctx)
        return ValidationResult(
            validator_name=self.name, passed=reason is None, message=reason
        )
