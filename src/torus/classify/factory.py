"""Build a ``PageClassifier`` from pipeline classify config."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from torus.classify.trained import TrainedImageTextClassifier
from torus.classify.vlm import ZeroShotVlmClassifier
from torus.classify.vlm_client import VlmClient, build_default_vlm_client
from torus.classify.vote import VotingPageClassifier
from torus.config.schemas import (
    ClassifyConfig,
    ClassifyStrategy,
    PageTypesConfig,
)
from torus.core.errors import ClassifyError, ConfigError
from torus.core.models import Page, PageTypePrediction
from torus.core.protocols import PageClassifier

PageTextLookup = Callable[[Page], str]


class DisabledPageClassifier:
    """No-op classifier when ``classify.enabled`` is false."""

    name = "disabled"

    def __init__(self, fallback_page_type: str = "other") -> None:
        self._fallback = fallback_page_type

    def classify(self, page: Page) -> PageTypePrediction:
        _ = page
        return PageTypePrediction(page_type=self._fallback, confidence=0.0)


def build_page_classifier(
    page_types: PageTypesConfig,
    config: ClassifyConfig | None = None,
    *,
    page_text_lookup: PageTextLookup | Mapping[int, str] | None = None,
    vlm_client: VlmClient | None = None,
) -> PageClassifier:
    """Construct the configured classifier (vlm, trained, or vote).

    Missing VLM endpoint/weights are soft-skipped inside each strategy; the
    factory never crashes solely because a model is unavailable.
    """
    cfg = config if config is not None else ClassifyConfig()
    _validate_fallback(page_types, cfg.fallback_page_type)

    if not cfg.enabled:
        return DisabledPageClassifier(cfg.fallback_page_type)

    if not page_types.page_types:
        raise ConfigError("page_types catalogue is empty; cannot build classifier")

    strategies: list[tuple[str, PageClassifier]] = []

    want_vlm = cfg.strategy in (ClassifyStrategy.VLM, ClassifyStrategy.VOTE)
    want_trained = cfg.strategy in (
        ClassifyStrategy.TRAINED,
        ClassifyStrategy.VOTE,
    )

    if want_vlm and cfg.vlm.enabled:
        client = vlm_client
        if client is None:
            client = build_default_vlm_client(cfg.vlm)
        strategies.append(
            (
                "vlm",
                ZeroShotVlmClassifier(
                    page_types,
                    cfg.vlm,
                    client=client,
                    fallback_page_type=cfg.fallback_page_type,
                    page_text_lookup=page_text_lookup,
                ),
            )
        )

    if want_trained and cfg.trained.enabled:
        strategies.append(
            (
                "trained",
                TrainedImageTextClassifier(
                    page_types,
                    cfg.trained,
                    fallback_page_type=cfg.fallback_page_type,
                    page_text_lookup=page_text_lookup,
                ),
            )
        )

    if not strategies:
        raise ClassifyError(
            "no classify strategies enabled for "
            f"strategy={cfg.strategy.value!r}; enable vlm and/or trained"
        )

    if cfg.strategy == ClassifyStrategy.VOTE and len(strategies) > 1:
        return VotingPageClassifier(
            strategies,
            cfg.vote,
            fallback_page_type=cfg.fallback_page_type,
        )

    # Single strategy (vlm-only, trained-only, or vote with one survivor).
    return strategies[0][1]


def _validate_fallback(page_types: PageTypesConfig, fallback: str) -> None:
    ids = page_types.ids()
    if ids and fallback not in ids:
        raise ConfigError(
            f"classify.fallback_page_type {fallback!r} is not in page_types.yaml"
        )
