"""Character-level weighted voting over aligned readings (P6.1).

``Candidate.confidence`` here is the weighted share of readers whose whole text
equals the voted text. It is a raw agreement feature, not a probability:
calibration (P6.3) maps it, with the other features on ``VoteResult``, to a
measured probability of being right.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from torus.consensus.align import align_texts
from torus.consensus.weights import ReaderWeights, resolve_weights
from torus.core.models import Candidate, Reading


class VoteResult(BaseModel):
    """Voted candidate plus the agreement features calibration needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: Candidate
    n_readings: int = Field(ge=0, description="Readings that took part in the vote.")
    agreeing_families: list[str] = Field(default_factory=list)
    min_column_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description="Weakest per-character weighted agreement (1.0 when unanimous).",
    )
    dissenting_readers: list[str] = Field(default_factory=list)


def vote(
    region_id: str,
    readings: Sequence[Reading],
    *,
    weights: ReaderWeights | Mapping[str, float] | None = None,
    field_type: str | None = None,
    families: Mapping[str, str] | None = None,
    use_char_confidence: bool = True,
) -> VoteResult:
    """Vote one text for ``region_id`` from its readings.

    Readings with ``error`` set or empty text abstain. With no usable reading the
    candidate is empty with confidence 0 (never a crash).
    """
    usable = sorted(
        (r for r in readings if r.error is None and r.text),
        key=lambda r: r.reader_name,
    )
    if not usable:
        return VoteResult(
            candidate=Candidate(region_id=region_id, text="", confidence=0.0),
            n_readings=0,
            min_column_agreement=0.0,
        )

    w = resolve_weights(weights, field_type, [r.reader_name for r in usable])
    ordered = sorted(usable, key=lambda r: (-w[r.reader_name], r.reader_name))
    rows = align_texts([r.text for r in ordered])

    chars: list[str] = []
    min_agreement = 1.0
    for col in range(len(rows[0])):
        scores: dict[str | None, float] = defaultdict(float)
        top_weight: dict[str | None, float] = defaultdict(float)
        for row, reading in zip(rows, ordered, strict=True):
            sym = row[col]
            wt = w[reading.reader_name] * _symbol_conf(
                reading, row, col, use_char_confidence
            )
            scores[sym] += wt
            top_weight[sym] = max(top_weight[sym], w[reading.reader_name])
        total = sum(scores.values())
        # Ties: heavier single reader, then a real character over a gap, then
        # code point, so output never depends on dict order.
        winner = max(
            scores,
            key=lambda s: (scores[s], top_weight[s], s is not None, s or ""),
        )
        min_agreement = min(min_agreement, scores[winner] / total if total > 0 else 0.0)
        if winner is not None:
            chars.append(winner)

    text = "".join(chars)
    agreeing = [r for r in ordered if r.text == text]
    dissenting = [r.reader_name for r in ordered if r.text != text]
    total_w = sum(w[r.reader_name] for r in ordered)
    share = sum(w[r.reader_name] for r in agreeing) / total_w if total_w > 0 else 0.0
    fam_map = families or {}
    agreeing_families = sorted(
        {fam_map.get(r.reader_name, r.reader_name) for r in agreeing}
    )

    return VoteResult(
        candidate=Candidate(
            region_id=region_id,
            text=text,
            agreeing_readers=[r.reader_name for r in agreeing],
            confidence=min(max(share, 0.0), 1.0),
        ),
        n_readings=len(ordered),
        agreeing_families=agreeing_families,
        min_column_agreement=min_agreement,
        dissenting_readers=dissenting,
    )


def _symbol_conf(
    reading: Reading, row: list[str | None], col: int, use_char_confidence: bool
) -> float:
    """Confidence scale for this reader's symbol (or gap) in one column."""
    fallback = reading.raw_score if reading.raw_score is not None else 1.0
    sym = row[col]
    confs = reading.char_confidences
    if (
        not use_char_confidence
        or sym is None
        or confs is None
        or len(confs) != len(reading.text)
    ):
        return fallback
    index = sum(1 for s in row[: col + 1] if s is not None) - 1
    return confs[index]
