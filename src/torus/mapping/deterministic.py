"""Deterministic mapping (P8.4): fill directly anchored, validated columns.

A column is filled without the LLM only when there is nothing to judge:

- every candidate was located by a label anchor, and
- all candidates agree on one value (case/whitespace-insensitive), and
- every validator the field spec lists ran and passed.

The value, status and confidence come from consensus (the weakest candidate
status and confidence); this stage never raises them. A candidate that fails a
validator is kept for the reviewer but marked ``needs_review``; it is never
corrected and never sent to the model to "fix". Anything else (several distinct
values, no anchor) returns ``None``: that is judgement, left to the LLM.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from torus.config.schemas import FieldSpec
from torus.core.models import Evidence, FieldValue
from torus.core.types import FieldStatus
from torus.mapping.rows import FieldObservation

REASON_VALIDATOR_FAILED = "validator_failed"
REASON_NOT_VALIDATED = "validation_missing"

_STATUS_RANK = {
    FieldStatus.CONFIRMED: 3,
    FieldStatus.LIKELY: 2,
    FieldStatus.NEEDS_REVIEW: 1,
    FieldStatus.NOT_FOUND: 0,
}


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def decide_direct(
    spec: FieldSpec, candidates: Sequence[FieldObservation]
) -> FieldValue | None:
    """Return the filled value, or ``None`` when the column needs judgement."""
    if not candidates or not all(c.anchored for c in candidates):
        return None
    if len({_normalise(c.text) for c in candidates}) != 1:
        return None

    results = [r for c in candidates for r in c.validator_results]
    failed = sorted({r.validator_name for r in results if not r.passed})
    ran = {r.validator_name for r in results}
    missing = sorted(set(spec.validators) - ran)

    weakest = min(candidates, key=lambda c: _STATUS_RANK[c.status])
    status = weakest.status
    reason: str | None = None
    if failed:
        status = FieldStatus.NEEDS_REVIEW
        reason = f"{REASON_VALIDATOR_FAILED}:{','.join(failed)}"
    elif missing:
        status = FieldStatus.NEEDS_REVIEW
        reason = f"{REASON_NOT_VALIDATED}:{','.join(missing)}"

    first = candidates[0]
    return FieldValue(
        column_key=spec.key,
        value=first.text,
        status=status,
        confidence=min(c.confidence for c in candidates),
        evidence=[
            Evidence(region_id=c.region_id, page_index=c.page_index, bbox=c.bbox)
            for c in candidates
        ],
        validator_results=results,
        reason=reason,
    )
