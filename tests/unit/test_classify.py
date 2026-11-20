"""Unit tests for page classifiers (P3.2). No model weights required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from torus.classify import (
    LinearWeights,
    TrainedImageTextClassifier,
    VotingPageClassifier,
    ZeroShotVlmClassifier,
    build_page_classifier,
    build_zero_shot_messages,
    classify_page,
    parse_vlm_response,
    save_weights,
    train_linear_classifier,
)
from torus.classify.features import extract_features, feature_names
from torus.classify.prompt import build_type_catalogue
from torus.classify.text_cues import cue_match_score, normalize_text
from torus.config import load_config
from torus.config.schemas import (
    ClassifyConfig,
    ClassifyStrategy,
    ClassifyTrainedConfig,
    ClassifyVlmConfig,
    ClassifyVoteConfig,
    PageTypesConfig,
    PageTypeSpec,
)
from torus.core import (
    ClassifyError,
    ConfigError,
    Page,
    PageClassifier,
    PageTypePrediction,
)


def _mini_types() -> PageTypesConfig:
    return PageTypesConfig(
        page_types=[
            PageTypeSpec(
                id="checklist",
                description="Document checklist with ticks.",
                examples=["KYC list"],
                visual_cues=["vertical list of checkboxes"],
                text_cues=["checklist", "documents required", "pending"],
            ),
            PageTypeSpec(
                id="blank",
                description="Blank page.",
                examples=["white sheet"],
                visual_cues=["uniform background"],
                text_cues=[],
            ),
            PageTypeSpec(
                id="other",
                description="Catch-all.",
                examples=["unknown"],
                visual_cues=["none"],
                text_cues=["unrelated"],
            ),
        ]
    )


def _page_with_image(
    *,
    blank: bool = False,
    size: tuple[int, int] = (200, 280),
    draw_lines: bool = False,
) -> Page:
    img = Image.new("RGB", size, color=(255, 255, 255))
    if draw_lines:
        draw = ImageDraw.Draw(img)
        for y in range(40, size[1] - 20, 18):
            draw.rectangle([20, y, 30, y + 10], outline=(0, 0, 0), width=1)
            draw.text((40, y), "item", fill=(0, 0, 0))
    page = Page(index=0, width=size[0], height=size[1], blank=blank)
    page.bind_image(img)
    return page


class _FixedVlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        image: Image.Image | None = None,
    ) -> str:
        _ = messages, image
        self.calls += 1
        return self.reply


def test_repo_pipeline_loads_classify_defaults() -> None:
    cfg = load_config()
    classify = cfg.pipeline.classify
    assert classify.enabled is True
    assert classify.strategy == ClassifyStrategy.VOTE
    assert classify.fallback_page_type == "other"
    assert classify.vlm.enabled is True
    assert classify.trained.enabled is True


def test_normalize_and_cue_match() -> None:
    assert normalize_text("Documents Required!") == "documents required"
    score = cue_match_score(
        "This is a checklist of documents required",
        ["checklist", "documents required", "missing cue"],
    )
    assert score == pytest.approx(2 / 3)


def test_parse_vlm_response_accepts_fenced_json() -> None:
    allowed = frozenset({"checklist", "other"})
    page_type, conf = parse_vlm_response(
        '```json\n{"page_type": "checklist", "confidence": 0.91}\n```',
        allowed,
    )
    assert page_type == "checklist"
    assert conf == pytest.approx(0.91)


def test_parse_vlm_response_rejects_unknown_id() -> None:
    allowed = frozenset({"checklist", "other"})
    page_type, conf = parse_vlm_response(
        '{"page_type": "not_in_catalogue", "confidence": 0.99}',
        allowed,
        fallback_page_type="other",
    )
    assert page_type == "other"
    assert conf == 0.0


def test_zero_shot_prompt_uses_catalogue_ids() -> None:
    types = _mini_types()
    catalogue = build_type_catalogue(types)
    assert {row["id"] for row in catalogue} == {"checklist", "blank", "other"}
    messages = build_zero_shot_messages(types, fallback_page_type="other")
    assert messages[0]["role"] == "system"
    assert "checklist" in messages[1]["content"]


def test_vlm_classifier_uses_injected_client() -> None:
    types = _mini_types()
    client = _FixedVlm('{"page_type": "checklist", "confidence": 0.88}')
    clf: PageClassifier = ZeroShotVlmClassifier(
        types,
        ClassifyVlmConfig(model="stub"),
        client=client,
        fallback_page_type="other",
    )
    pred = clf.classify(_page_with_image(draw_lines=True))
    assert client.calls == 1
    assert pred.page_type == "checklist"
    assert pred.confidence == pytest.approx(0.88)


def test_vlm_classifier_soft_fails_without_client() -> None:
    types = _mini_types()
    clf = ZeroShotVlmClassifier(
        types,
        ClassifyVlmConfig(model=None),
        client=None,
        fallback_page_type="other",
    )
    pred = clf.classify(_page_with_image())
    assert pred.page_type == "other"
    assert pred.confidence == 0.0


def test_vlm_blank_short_circuit() -> None:
    types = _mini_types()
    client = _FixedVlm('{"page_type": "other", "confidence": 0.1}')
    clf = ZeroShotVlmClassifier(
        types,
        ClassifyVlmConfig(model="stub"),
        client=client,
        fallback_page_type="other",
    )
    pred = clf.classify(_page_with_image(blank=True))
    assert pred.page_type == "blank"
    assert client.calls == 0


def test_trained_prior_uses_text_cues() -> None:
    types = _mini_types()
    clf = TrainedImageTextClassifier(
        types,
        ClassifyTrainedConfig(weights=None),
        fallback_page_type="other",
        page_text_lookup={0: "KYC checklist documents required pending"},
    )
    pred = clf.classify(_page_with_image(draw_lines=True))
    assert pred.page_type == "checklist"
    assert pred.confidence > 0.3


def test_trained_weights_roundtrip_and_predict(tmp_path: Path) -> None:
    types = _mini_types()
    type_ids = [s.id for s in types.page_types]
    names = feature_names(type_ids)
    page = _page_with_image(draw_lines=True)
    text = "checklist documents required"
    x = extract_features(page, types, text)
    # Two samples of checklist, one other.
    samples = [
        (x, "checklist"),
        (x + 0.01, "checklist"),
        (np.zeros_like(x), "other"),
    ]
    weights = train_linear_classifier(
        samples, type_ids, names, epochs=80, learning_rate=0.8, seed=1
    )
    path = tmp_path / "page_clf.json"
    save_weights(weights, path)

    clf = TrainedImageTextClassifier(
        types,
        ClassifyTrainedConfig(weights=str(path)),
        fallback_page_type="other",
        page_text_lookup={0: text},
    )
    pred = clf.classify(page)
    assert pred.page_type == "checklist"
    assert isinstance(clf, PageClassifier)


def test_vote_agree_boosts_confidence() -> None:
    class _Stub:
        def __init__(self, page_type: str, confidence: float) -> None:
            self.page_type = page_type
            self.confidence = confidence

        def classify(self, page: Page) -> PageTypePrediction:
            _ = page
            return PageTypePrediction(
                page_type=self.page_type, confidence=self.confidence
            )

    voter = VotingPageClassifier(
        [("a", _Stub("checklist", 0.7)), ("b", _Stub("checklist", 0.6))],
        ClassifyVoteConfig(agree_boost=0.1, disagree_penalty=0.15),
        fallback_page_type="other",
    )
    pred = voter.classify(_page_with_image())
    assert pred.page_type == "checklist"
    assert pred.confidence == pytest.approx(0.75)  # mean 0.65 + 0.1


def test_vote_disagree_applies_penalty() -> None:
    class _Stub:
        def __init__(self, page_type: str, confidence: float) -> None:
            self._page_type = page_type
            self._confidence = confidence

        def classify(self, page: Page) -> PageTypePrediction:
            _ = page
            return PageTypePrediction(
                page_type=self._page_type, confidence=self._confidence
            )

    voter = VotingPageClassifier(
        [("a", _Stub("checklist", 0.9)), ("b", _Stub("other", 0.4))],
        ClassifyVoteConfig(agree_boost=0.1, disagree_penalty=0.15),
        fallback_page_type="other",
    )
    pred = voter.classify(_page_with_image())
    assert pred.page_type == "checklist"
    assert pred.confidence == pytest.approx(0.75)  # 0.9 - 0.15


def test_factory_vote_with_injected_vlm() -> None:
    types = _mini_types()
    client = _FixedVlm('{"page_type": "checklist", "confidence": 0.8}')
    cfg = ClassifyConfig(
        strategy=ClassifyStrategy.VOTE,
        fallback_page_type="other",
        vlm=ClassifyVlmConfig(enabled=True, model="stub"),
        trained=ClassifyTrainedConfig(enabled=True, weights=None),
    )
    clf = build_page_classifier(
        types,
        cfg,
        vlm_client=client,
        page_text_lookup={0: "checklist documents required"},
    )
    page = _page_with_image(draw_lines=True)
    pred = classify_page(clf, page)
    assert page.page_type == pred.page_type
    assert pred.page_type == "checklist"


def test_factory_rejects_unknown_fallback() -> None:
    types = _mini_types()
    with pytest.raises(ConfigError, match="fallback_page_type"):
        build_page_classifier(
            types,
            ClassifyConfig(fallback_page_type="not_a_type"),
        )


def test_linear_weights_shape_guard() -> None:
    with pytest.raises(ClassifyError):
        LinearWeights(
            ["a", "b"],
            ["f1"],
            np.zeros((2, 2)),
            np.zeros(2),
        )
