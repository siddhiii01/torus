"""Combine two PageClassifier strategies when config strategy is ``vote``."""

from __future__ import annotations

from collections.abc import Sequence

from torus.config.schemas import ClassifyVoteConfig
from torus.core.models import Page, PageTypePrediction
from torus.core.protocols import PageClassifier
from torus.logging import get_logger

logger = get_logger(__name__)


class VotingPageClassifier:
    """Runs enabled strategies and merges their predictions."""

    name = "vote"

    def __init__(
        self,
        classifiers: Sequence[tuple[str, PageClassifier]],
        vote_config: ClassifyVoteConfig | None = None,
        *,
        fallback_page_type: str = "other",
    ) -> None:
        if not classifiers:
            msg = "VotingPageClassifier requires at least one strategy"
            raise ValueError(msg)
        self._classifiers = list(classifiers)
        self._vote = vote_config or ClassifyVoteConfig()
        self._fallback = fallback_page_type

    def classify(self, page: Page) -> PageTypePrediction:
        predictions: list[tuple[str, PageTypePrediction]] = []
        for name, clf in self._classifiers:
            predictions.append((name, clf.classify(page)))

        if len(predictions) == 1:
            return predictions[0][1]

        # Average confidence per page_type across voters.
        totals: dict[str, list[float]] = {}
        for _name, pred in predictions:
            totals.setdefault(pred.page_type, []).append(pred.confidence)
            for alt_type, alt_conf in pred.alternatives:
                totals.setdefault(alt_type, []).append(alt_conf)

        means = {type_id: sum(vals) / len(vals) for type_id, vals in totals.items()}
        if not means:
            return PageTypePrediction(
                page_type=self._fallback,
                confidence=0.0,
            )

        ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
        winner_type, winner_conf = ranked[0]

        strategy_labels = [pred.page_type for _n, pred in predictions]
        all_agree = len(set(strategy_labels)) == 1

        if all_agree:
            confidence = min(1.0, winner_conf + self._vote.agree_boost)
        else:
            confidence = max(0.0, winner_conf - self._vote.disagree_penalty)

        alternatives = [
            (type_id, conf) for type_id, conf in ranked[1:4] if type_id != winner_type
        ]

        logger.info(
            "page_classified",
            classifier_name=self.name,
            page_index=page.index,
            page_type=winner_type,
            confidence=confidence,
            strategy="vote",
        )
        return PageTypePrediction(
            page_type=winner_type,
            confidence=confidence,
            alternatives=alternatives,
        )
