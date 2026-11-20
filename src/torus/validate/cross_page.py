"""Cross-page agreement (P7.3): one fact found on several pages must agree.

The caller groups observations that claim to be the *same fact* (same column,
same person/row) and passes them in. This module only compares them:

* two or more distinct pages, all equal after normalisation -> ``AGREE``;
* any two different normalised values -> ``DISAGREE`` (both values kept, with
  their evidence, for the review screen);
* fewer than two distinct pages -> ``SINGLE``: no signal either way.

The outcome is exposed as a ``ValidationResult``. A disagreement is a failed
validator, so P6.4 already caps the field at ``needs_review``; an agreement is a
passed validator, which feeds the calibrated confidence through the existing
``validator_passed`` feature. No confidence formula is invented here. Normalisation
only removes presentation differences (case, spacing, separators) and never maps
look-alike characters. Messages are codes; the values live only in the result's
groups, never in logs or exceptions.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from torus.core.errors import ConfigError
from torus.core.models import Evidence, ValidationResult

CROSS_PAGE_VALIDATOR_NAME = "cross_page_agreement"
REASON_DISAGREE = "cross_page_disagreement"

_WS = re.compile(r"\s+")
_TEXT_PUNCT = re.compile(r"[.,;:\-_/\\'\"()]")
_NON_ALNUM = re.compile(r"[\W_]+")
_NON_DIGIT = re.compile(r"\D+")


class FactKind(StrEnum):
    """How two printed forms of one fact are canonicalised before comparing."""

    TEXT = "text"  # names, addresses: case, spacing, light punctuation
    ALNUM = "alnum"  # ID numbers, codes: letters and digits only
    DIGITS = "digits"  # phone, PIN, account: digits only


class AgreementOutcome(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    SINGLE = "single"


def normalise_fact(text: str, kind: FactKind) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    if kind is FactKind.DIGITS:
        return _NON_DIGIT.sub("", t)
    if kind is FactKind.ALNUM:
        return _NON_ALNUM.sub("", t)
    return _WS.sub(" ", _TEXT_PUNCT.sub(" ", t)).strip()


class FactObservation(BaseModel):
    """One reading of the fact on one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    evidence: Evidence
    confidence: float = Field(ge=0.0, le=1.0)


class FactGroup(BaseModel):
    """Observations that agree with each other after normalisation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    normalised: str
    text: str = Field(description="Printed form of the most confident observation.")
    observations: list[FactObservation]

    @property
    def pages(self) -> list[int]:
        return sorted({o.evidence.page_index for o in self.observations})


class CrossPageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str
    outcome: AgreementOutcome
    groups: list[FactGroup]
    pages: list[int]

    @property
    def validation_result(self) -> ValidationResult | None:
        """None when there is no cross-page signal (``SINGLE``)."""
        if self.outcome is AgreementOutcome.SINGLE:
            return None
        agree = self.outcome is AgreementOutcome.AGREE
        return ValidationResult(
            validator_name=CROSS_PAGE_VALIDATOR_NAME,
            passed=agree,
            message=None if agree else REASON_DISAGREE,
        )

    @property
    def review_evidence(self) -> list[Evidence]:
        """Evidence of every competing value, so a reviewer sees them all."""
        return [o.evidence for g in self.groups for o in g.observations]


def check_cross_page(
    column_key: str,
    observations: Sequence[FactObservation],
    kind: FactKind,
) -> CrossPageResult:
    """Compare observations of one fact. Empty readings are ignored."""
    groups: dict[str, list[FactObservation]] = {}
    for obs in observations:
        norm = normalise_fact(obs.text, kind)
        if norm:
            groups.setdefault(norm, []).append(obs)
    if not groups:
        raise ConfigError(f"no usable observations for column {column_key!r}")

    fact_groups = [
        FactGroup(
            normalised=norm,
            text=max(obs_list, key=lambda o: o.confidence).text,
            observations=list(obs_list),
        )
        for norm, obs_list in groups.items()
    ]
    # Most-supported value first, ties broken by confidence then text for
    # determinism.
    fact_groups.sort(
        key=lambda g: (
            -len(g.pages),
            -max(o.confidence for o in g.observations),
            g.normalised,
        )
    )
    pages = sorted({o.evidence.page_index for g in fact_groups for o in g.observations})

    if len(fact_groups) > 1:
        outcome = AgreementOutcome.DISAGREE
    elif len(pages) >= 2:
        outcome = AgreementOutcome.AGREE
    else:
        outcome = AgreementOutcome.SINGLE
    return CrossPageResult(
        column_key=column_key, outcome=outcome, groups=fact_groups, pages=pages
    )
