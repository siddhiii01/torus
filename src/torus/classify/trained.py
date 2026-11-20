"""Small trained image+text page classifier (NumPy softmax / linear model)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torus.classify.features import extract_features, feature_names
from torus.classify.text_cues import text_cue_scores
from torus.config.schemas import ClassifyTrainedConfig, PageTypesConfig
from torus.core.errors import ClassifyError
from torus.core.models import Page, PageTypePrediction
from torus.logging import get_logger

logger = get_logger(__name__)

PageTextLookup = Callable[[Page], str]
WEIGHTS_VERSION = 1


class TrainedImageTextClassifier:
    """PageClassifier strategy (b): image features + text cues → type scores."""

    name = "trained_image_text"

    def __init__(
        self,
        page_types: PageTypesConfig,
        config: ClassifyTrainedConfig | None = None,
        *,
        weights: LinearWeights | None = None,
        fallback_page_type: str = "other",
        page_text_lookup: PageTextLookup | Mapping[int, str] | None = None,
    ) -> None:
        self._page_types = page_types
        self._config = config or ClassifyTrainedConfig()
        self._fallback = fallback_page_type
        self._page_text_lookup = page_text_lookup
        self._type_ids = [spec.id for spec in page_types.page_types]
        self._allowed = (
            frozenset(self._type_ids)
            if self._type_ids
            else frozenset({fallback_page_type})
        )
        self._weights: LinearWeights | None
        if weights is not None:
            self._weights = weights
        else:
            self._weights = _load_weights_optional(
                self._config.weights, expected_ids=self._type_ids
            )

    def classify(self, page: Page) -> PageTypePrediction:
        if page.blank:
            blank_id = "blank" if "blank" in self._allowed else self._fallback
            return PageTypePrediction(page_type=blank_id, confidence=0.95)

        page_text = _resolve_page_text(page, self._page_text_lookup)
        features = extract_features(page, self._page_types, page_text)

        if self._weights is not None:
            scores = self._weights.score(features)
        else:
            scores = _prior_scores(page, page_text, self._page_types, self._type_ids)

        page_type, confidence, alternatives = _topk_prediction(
            scores, self._type_ids, fallback=self._fallback
        )
        logger.info(
            "page_classified",
            classifier_name=self.name,
            page_index=page.index,
            page_type=page_type,
            confidence=confidence,
            strategy="trained",
        )
        return PageTypePrediction(
            page_type=page_type,
            confidence=confidence,
            alternatives=alternatives,
        )


class LinearWeights:
    """Multinomial logistic weights: score = softmax(W @ x + b)."""

    def __init__(
        self,
        page_type_ids: Sequence[str],
        feature_name_list: Sequence[str],
        weight_matrix: NDArray[np.float64],
        bias: NDArray[np.float64],
    ) -> None:
        self.page_type_ids = list(page_type_ids)
        self.feature_names = list(feature_name_list)
        self.weights = np.asarray(weight_matrix, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.weights.shape != (len(self.page_type_ids), len(self.feature_names)):
            raise ClassifyError(
                "weight matrix shape does not match page types / features"
            )
        if self.bias.shape != (len(self.page_type_ids),):
            raise ClassifyError("bias shape does not match page types")

    def score(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if features.shape != (len(self.feature_names),):
            raise ClassifyError(
                f"expected {len(self.feature_names)} features, got {features.shape[0]}"
            )
        logits = self.weights @ features + self.bias
        return _softmax(logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": WEIGHTS_VERSION,
            "page_type_ids": self.page_type_ids,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearWeights:
        version = int(data.get("version", 0))
        if version != WEIGHTS_VERSION:
            raise ClassifyError(f"unsupported weights version {version}")
        ids = list(data["page_type_ids"])
        names = list(data["feature_names"])
        weights = np.asarray(data["weights"], dtype=np.float64)
        bias = np.asarray(data["bias"], dtype=np.float64)
        return cls(ids, names, weights, bias)


def resolve_weights_path(path_str: str | None) -> Path | None:
    """Resolve weights path; relative paths use ``TORUS_MODEL_DIR``."""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    model_dir = os.environ.get("TORUS_MODEL_DIR", "").strip()
    if not model_dir:
        return path
    return Path(model_dir) / path


def load_weights(path: Path) -> LinearWeights:
    """Load trained weights from a JSON file under the model dir."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClassifyError(f"cannot read weights: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ClassifyError(f"invalid weights JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ClassifyError(f"weights root must be an object: {path}")
    return LinearWeights.from_dict(data)


def save_weights(weights: LinearWeights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(weights.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_linear_classifier(
    samples: Sequence[tuple[NDArray[np.float64], str]],
    page_type_ids: Sequence[str],
    feature_name_list: Sequence[str],
    *,
    epochs: int = 200,
    learning_rate: float = 0.5,
    seed: int = 0,
) -> LinearWeights:
    """Fit a multinomial logistic model with plain NumPy (no new deps)."""
    if not samples:
        raise ClassifyError("no training samples")
    id_to_idx = {tid: i for i, tid in enumerate(page_type_ids)}
    unknown = sorted({label for _, label in samples if label not in id_to_idx})
    if unknown:
        raise ClassifyError(f"unknown page_type labels in samples: {unknown}")

    rng = np.random.default_rng(seed)
    n_classes = len(page_type_ids)
    n_features = len(feature_name_list)
    w = rng.normal(0.0, 0.01, size=(n_classes, n_features))
    b = np.zeros(n_classes, dtype=np.float64)

    xs = np.stack([x for x, _ in samples], axis=0)
    ys = np.array([id_to_idx[label] for _, label in samples], dtype=np.int64)

    for _ in range(epochs):
        logits = xs @ w.T + b
        probs = _softmax_rows(logits)
        # One-hot targets
        y_oh = np.zeros_like(probs)
        y_oh[np.arange(len(ys)), ys] = 1.0
        grad = (probs - y_oh) / max(len(ys), 1)
        w -= learning_rate * (grad.T @ xs)
        b -= learning_rate * grad.sum(axis=0)

    return LinearWeights(page_type_ids, feature_name_list, w, b)


def build_training_sample(
    page: Page,
    page_types: PageTypesConfig,
    label: str,
    page_text: str = "",
) -> tuple[NDArray[np.float64], str]:
    """Extract features for one labelled page (training data outside the repo)."""
    return extract_features(page, page_types, page_text), label


def _load_weights_optional(
    path_str: str | None,
    *,
    expected_ids: Sequence[str],
) -> LinearWeights | None:
    path = resolve_weights_path(path_str)
    if path is None or not path.is_file():
        return None
    weights = load_weights(path)
    if list(weights.page_type_ids) != list(expected_ids):
        raise ClassifyError(
            "trained weights page_type_ids do not match config/page_types.yaml"
        )
    expected_names = feature_names(expected_ids)
    if list(weights.feature_names) != expected_names:
        raise ClassifyError(
            "trained weights feature_names do not match current feature layout"
        )
    return weights


def _prior_scores(
    page: Page,
    page_text: str,
    page_types: PageTypesConfig,
    type_ids: Sequence[str],
) -> NDArray[np.float64]:
    """Untrained fallback: text-cue scores + light visual priors (config cues)."""
    cues = text_cue_scores(page_text, page_types)
    scores = np.array([cues.get(tid, 0.0) for tid in type_ids], dtype=np.float64)

    if page.has_image:
        features = extract_features(page, page_types, page_text)
        ink = float(features[0])
        hline = float(features[2])
        large_blob = float(features[3])
        text_dens = float(features[4])
        # Soft visual nudges keyed by catalogue ids when present.
        for i, tid in enumerate(type_ids):
            if tid == "blank" and ink < 0.002:
                scores[i] += 0.8
            elif tid == "bank_statement" and hline > 0.55:
                scores[i] += 0.25
            elif tid == "photo_signature" and large_blob > 0.08 and text_dens < 0.2:
                scores[i] += 0.3
            elif tid == "brochure_terms" and text_dens > 0.4 and large_blob < 0.05:
                scores[i] += 0.15
            elif tid == "checklist" and text_dens > 0.15 and hline < 0.45:
                scores[i] += 0.1

    if float(scores.max()) <= 0.0:
        # Uniform over types except prefer fallback if listed.
        scores = np.ones(len(type_ids), dtype=np.float64)
        if "other" in type_ids:
            scores[list(type_ids).index("other")] = 2.0
    return _softmax(scores)


def _topk_prediction(
    scores: NDArray[np.float64],
    type_ids: Sequence[str],
    *,
    fallback: str,
) -> tuple[str, float, list[tuple[str, float]]]:
    if len(scores) == 0 or len(type_ids) == 0:
        return fallback, 0.0, []
    order = np.argsort(-scores)
    best_i = int(order[0])
    page_type = type_ids[best_i]
    confidence = float(scores[best_i])
    alternatives: list[tuple[str, float]] = []
    for idx in order[1:4]:
        alternatives.append((type_ids[int(idx)], float(scores[int(idx)])))
    return page_type, confidence, alternatives


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom <= 0.0:
        return np.full_like(logits, 1.0 / max(len(logits), 1))
    return exps / denom


def _softmax_rows(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    denom = np.sum(exps, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return exps / denom


def _resolve_page_text(
    page: Page,
    lookup: PageTextLookup | Mapping[int, str] | None,
) -> str:
    if lookup is None:
        return ""
    if callable(lookup):
        return lookup(page) or ""
    return lookup.get(page.index, "") or ""
