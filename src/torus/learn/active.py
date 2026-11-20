"""Active learning (P10.5): order the review queue so each correction teaches most.

This only *orders* items; every ``needs_review`` / ``likely`` item still reaches
a person. Score = how unsure the calibrated confidence is, boosted for columns
people often correct and for items where readers disagree. A per-column decay
keeps one column from filling the top of the queue. Pure and deterministic.
Weights live in ``config/active_learning.yaml``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.learn.dataset import LearnError
from torus.review.queue import ReviewItem
from torus.review.store import CorrectionRecord


class ActiveLearningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uncertainty_weight: float = Field(default=1.0, ge=0.0)
    column_error_weight: float = Field(default=0.5, ge=0.0)
    disagreement_weight: float = Field(default=0.5, ge=0.0)
    column_decay: float = Field(default=0.25, ge=0.0)
    prior_corrections: float = Field(default=1.0, gt=0.0, description="Smoothing.")


def load_active_config(path: Path) -> ActiveLearningConfig:
    try:
        return ActiveLearningConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        )
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise LearnError(f"invalid active-learning config: {path.name}") from exc


def column_correction_rates(
    records: Iterable[CorrectionRecord], *, prior: float = 1.0
) -> dict[str, float]:
    """Smoothed share of human decisions per column that changed the value."""
    corrected: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for rec in records:
        if rec.action == "skip":
            continue
        total[rec.column_key] += 1
        corrected[rec.column_key] += rec.action == "correct"
    return {key: (corrected[key] + prior) / (total[key] + 2 * prior) for key in total}


def disagreement(item: ReviewItem) -> float:
    """0 when readers agree (or there is one), up to 1 when all differ."""
    texts = [
        r.text.strip() for r in item.readings if r.error is None and r.text.strip()
    ]
    if len(texts) < 2:
        return 0.0
    return (len(set(texts)) - 1) / (len(texts) - 1)


def prioritise(
    items: Sequence[ReviewItem],
    rates: dict[str, float],
    config: ActiveLearningConfig | None = None,
) -> list[ReviewItem]:
    """Reorder ``items`` most-informative first; never drops or adds items."""
    cfg = config or ActiveLearningConfig()
    base = {
        it.item_id: cfg.uncertainty_weight * (1.0 - it.confidence)
        + cfg.column_error_weight * rates.get(it.column_key, 0.0)
        + cfg.disagreement_weight * disagreement(it)
        for it in items
    }
    remaining = sorted(items, key=lambda i: (-base[i.item_id], i.row, i.column_key))
    picked: Counter[str] = Counter()
    out: list[ReviewItem] = []
    while remaining:
        best = max(
            remaining,
            key=lambda i: (
                base[i.item_id] / (1.0 + cfg.column_decay * picked[i.column_key]),
                -i.row,
                i.column_key,
                i.item_id,
            ),
        )
        remaining.remove(best)
        picked[best.column_key] += 1
        out.append(best)
    return out
