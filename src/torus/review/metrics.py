"""Adapt stored review decisions to eval review metrics (P9.5)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torus.eval.review_metrics import (
    DecisionEvent,
    PackReviewInput,
    ReviewMetrics,
    aggregate_review_metrics,
)
from torus.review.bundle import load_bundle
from torus.review.store import CorrectionRecord, CorrectionStore


def _event(rec: CorrectionRecord) -> DecisionEvent:
    return DecisionEvent(rec.column_key, rec.action, rec.decided_at)


def review_metrics_from_bundles(
    bundles: Sequence[Path], store: CorrectionStore, *, max_gap_s: float
) -> ReviewMetrics:
    packs: list[PackReviewInput] = []
    for path in bundles:
        result = load_bundle(path).result
        packs.append(
            PackReviewInput(
                result=result,
                final=[_event(r) for r in store.latest(result.document_id).values()],
                history=[_event(r) for r in store.load(result.document_id)],
            )
        )
    return aggregate_review_metrics(packs, max_gap_s=max_gap_s)
