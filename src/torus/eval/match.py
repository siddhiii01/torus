"""Per-column match rules: exact, normalised, fuzzy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torus.config.schemas import FieldSpec, MatchRule, NormalizeKind
from torus.eval.normalize import (
    apply_kind,
    as_text,
    is_empty,
    kind_for_field_type,
    norm_text,
    tokens,
)
from torus.eval.schema import ExpectedCell


def resolve_normalize_kind(spec: FieldSpec) -> NormalizeKind:
    """Normalisation kind for a field: explicit override, else from type."""
    if spec.normalize is not None:
        return spec.normalize
    return kind_for_field_type(spec.type)


def canonicalize(
    spec: FieldSpec,
    value: ExpectedCell | Any,
    *,
    rule: MatchRule | None = None,
    aliases: Mapping[str, list[str]] | None = None,
) -> str:
    """Canonical string for comparison under *rule* (defaults to spec.match)."""
    effective = rule if rule is not None else spec.match
    if is_empty(value):
        return ""
    if effective is MatchRule.EXACT:
        return as_text(value).strip()
    kind = resolve_normalize_kind(spec)
    return apply_kind(kind, value, aliases=aliases)


def fuzzy_equal(pred: str, gold: str, *, threshold: float) -> bool:
    """Token-overlap match: |pred ∩ gold| / |gold| >= threshold."""
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pred_tokens, gold_tokens = tokens(pred), tokens(gold)
    if not gold_tokens:
        return not pred_tokens
    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)
    return overlap >= threshold


def values_match(
    spec: FieldSpec,
    pred: ExpectedCell | Any,
    gold: ExpectedCell | Any,
    *,
    fuzzy_threshold: float = 0.8,
    aliases: Mapping[str, list[str]] | None = None,
) -> bool:
    """Return True when predicted and gold agree under the field's match rule."""
    rule = spec.match
    if rule is MatchRule.EXACT:
        return as_text(pred).strip() == as_text(gold).strip()

    pred_n = canonicalize(spec, pred, rule=MatchRule.NORMALISED, aliases=aliases)
    gold_n = canonicalize(spec, gold, rule=MatchRule.NORMALISED, aliases=aliases)

    if rule is MatchRule.NORMALISED:
        return pred_n == gold_n

    left = pred_n if pred_n else norm_text(pred)
    right = gold_n if gold_n else norm_text(gold)
    return fuzzy_equal(left, right, threshold=fuzzy_threshold)
