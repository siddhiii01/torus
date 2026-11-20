"""Reader weights per field type, learned from the dev set.

Weight = log-odds of the reader's smoothed character accuracy, the naive-Bayes
weight for an independent voter. Only counts and weights are stored, never text.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from torus.consensus.align import edit_distance
from torus.core.models import Reading

DEFAULT_WEIGHT = 1.0


class ReaderWeights(BaseModel):
    """Per-field-type reader weights with a fallback for unseen combinations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: float = Field(default=DEFAULT_WEIGHT, ge=0.0)
    by_field_type: dict[str, dict[str, float]] = Field(default_factory=dict)

    def for_field(self, field_type: str | None) -> dict[str, float]:
        """Reader name → weight for one field type (empty when not fitted)."""
        if field_type is None:
            return {}
        return dict(self.by_field_type.get(field_type, {}))

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReaderWeights:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WeightSample:
    """One dev-set region: its field type, the answer key text and the readings."""

    field_type: str
    gold: str
    readings: Sequence[Reading]


def fit_reader_weights(
    samples: Iterable[WeightSample],
    *,
    min_chars: int = 20,
    max_weight: float = 10.0,
) -> ReaderWeights:
    """Fit log-odds weights per (field type, reader).

    Failed or empty readings are skipped: availability is not accuracy. A
    (field type, reader) pair with fewer than ``min_chars`` gold characters is
    left out so the default applies instead of an unreliable estimate.
    """
    correct: dict[tuple[str, str], int] = defaultdict(int)
    total: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        n = len(sample.gold)
        if n == 0:
            continue
        for reading in sample.readings:
            if reading.error is not None or not reading.text:
                continue
            key = (sample.field_type, reading.reader_name)
            total[key] += n
            correct[key] += max(n - edit_distance(sample.gold, reading.text), 0)

    by_type: dict[str, dict[str, float]] = defaultdict(dict)
    for (field_type, reader), n_chars in sorted(total.items()):
        if n_chars < min_chars:
            continue
        acc = (correct[(field_type, reader)] + 1) / (n_chars + 2)
        weight = math.log(acc / (1.0 - acc))
        by_type[field_type][reader] = min(max(weight, 0.0), max_weight)
    return ReaderWeights(by_field_type=dict(by_type))


def resolve_weights(
    weights: ReaderWeights | Mapping[str, float] | None,
    field_type: str | None,
    readers: Iterable[str],
) -> dict[str, float]:
    """Weight for each reader name, falling back to the default."""
    if weights is None:
        return {r: DEFAULT_WEIGHT for r in readers}
    if isinstance(weights, ReaderWeights):
        fitted = weights.for_field(field_type)
        return {r: fitted.get(r, weights.default) for r in readers}
    return {r: weights.get(r, DEFAULT_WEIGHT) for r in readers}
