"""Zero-shot VLM page classifier using ``page_types.yaml`` descriptions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from PIL import Image

from torus.classify.prompt import build_zero_shot_messages, parse_vlm_response
from torus.classify.vlm_client import VlmClient
from torus.config.schemas import ClassifyVlmConfig, PageTypesConfig
from torus.core.errors import ClassifyError
from torus.core.models import Page, PageTypePrediction
from torus.logging import get_logger

logger = get_logger(__name__)

PageTextLookup = Callable[[Page], str]


class ZeroShotVlmClassifier:
    """PageClassifier strategy (a): zero-shot VLM over type descriptions."""

    name = "vlm_zero_shot"

    def __init__(
        self,
        page_types: PageTypesConfig,
        config: ClassifyVlmConfig,
        *,
        client: VlmClient | None = None,
        fallback_page_type: str = "other",
        page_text_lookup: PageTextLookup | Mapping[int, str] | None = None,
    ) -> None:
        self._page_types = page_types
        self._config = config
        self._client = client
        self._fallback = fallback_page_type
        self._page_text_lookup = page_text_lookup
        allowed = page_types.ids()
        if fallback_page_type not in allowed and allowed:
            # Keep going; factory validates. Soft path uses first id only if empty.
            pass
        self._allowed = allowed if allowed else frozenset({fallback_page_type})

    def classify(self, page: Page) -> PageTypePrediction:
        if page.blank:
            blank_id = "blank" if "blank" in self._allowed else self._fallback
            return PageTypePrediction(page_type=blank_id, confidence=0.95)

        client = self._client
        if client is None:
            logger.info(
                "classifier_skipped",
                classifier_name=self.name,
                page_index=page.index,
                strategy="vlm",
                error_code="no_client",
            )
            return PageTypePrediction(
                page_type=self._fallback,
                confidence=0.0,
            )

        page_text = _resolve_page_text(page, self._page_text_lookup)
        messages = build_zero_shot_messages(
            self._page_types,
            fallback_page_type=self._fallback,
            page_text_excerpt=page_text or None,
        )
        image: Image.Image | None = None
        if page.has_image:
            image = cast(Image.Image, page.image)

        try:
            raw = client.complete(messages, image=image)
            page_type, confidence = parse_vlm_response(
                raw,
                self._allowed,
                fallback_page_type=self._fallback,
            )
        except ClassifyError as exc:
            logger.info(
                "classifier_soft_fail",
                classifier_name=self.name,
                page_index=page.index,
                strategy="vlm",
                error_type=type(exc).__name__,
            )
            return PageTypePrediction(
                page_type=self._fallback,
                confidence=0.0,
            )

        alternatives: list[tuple[str, float]] = []
        if page_type != self._fallback:
            alternatives.append((self._fallback, max(0.0, 1.0 - confidence)))

        logger.info(
            "page_classified",
            classifier_name=self.name,
            page_index=page.index,
            page_type=page_type,
            confidence=confidence,
            strategy="vlm",
        )
        return PageTypePrediction(
            page_type=page_type,
            confidence=confidence,
            alternatives=alternatives,
        )


def _resolve_page_text(
    page: Page,
    lookup: PageTextLookup | Mapping[int, str] | None,
) -> str:
    if lookup is None:
        return ""
    if callable(lookup):
        return lookup(page) or ""
    return lookup.get(page.index, "") or ""
