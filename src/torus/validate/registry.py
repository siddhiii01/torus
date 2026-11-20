"""Name → validator registry so field specs can reference validators by name."""

from __future__ import annotations

from collections.abc import Iterable

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext, ValidationResult
from torus.core.protocols import Validator
from torus.validate.format import FORMAT_VALIDATORS
from torus.validate.lookup import LOOKUP_VALIDATORS

_REGISTRY: dict[str, Validator] = {}


def register_validator(validator: Validator, *, replace: bool = False) -> None:
    """Register *validator* under its ``name``."""
    if validator.name in _REGISTRY and not replace:
        msg = f"validator {validator.name!r} is already registered"
        raise ConfigError(msg)
    _REGISTRY[validator.name] = validator


def get_validator(name: str) -> Validator:
    try:
        return _REGISTRY[name]
    except KeyError:
        msg = f"unknown validator {name!r}; known: {', '.join(available_validators())}"
        raise ConfigError(msg) from None


def available_validators() -> list[str]:
    return sorted(_REGISTRY)


def run_validators(
    names: Iterable[str], value: str, ctx: ValidationContext
) -> list[ValidationResult]:
    """Run the validators a field spec lists; one result per name, in order.

    An unknown name raises ``ConfigError``: silently skipping it would let a
    value reach ``confirmed`` without the check the spec asked for.
    """
    return [get_validator(n).check(value, ctx) for n in names]


for _v in (*FORMAT_VALIDATORS, *LOOKUP_VALIDATORS):
    register_validator(_v)
