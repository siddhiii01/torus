"""Small NumPy MLP for one-character classification (no torch at inference)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from torus.core.errors import ReaderError
from torus.core.types import Charset
from torus.readers.char_cell.alphabet import DEFAULT_CLASSES, allowed_indices
from torus.readers.char_cell.preprocess import DEFAULT_INPUT_SIZE

WEIGHTS_VERSION = 1
DEFAULT_HIDDEN = 128

MatF64 = NDArray[np.float64]
VecF64 = NDArray[np.float64]


@dataclass(frozen=True)
class CharCellPrediction:
    """One classifier output: character, softmax probability, full distribution."""

    character: str
    probability: float
    probabilities: tuple[float, ...]


@dataclass
class CharCellWeights:
    """Two-layer MLP: softmax(W2 @ relu(W1 @ x + b1) + b2)."""

    classes: tuple[str, ...]
    input_size: int
    hidden_size: int
    w1: MatF64
    b1: VecF64
    w2: MatF64
    b2: VecF64

    def __post_init__(self) -> None:
        n_in = self.input_size * self.input_size
        n_cls = len(self.classes)
        if self.w1.shape != (self.hidden_size, n_in):
            raise ReaderError(
                f"w1 shape {self.w1.shape} != ({self.hidden_size}, {n_in})"
            )
        if self.b1.shape != (self.hidden_size,):
            raise ReaderError(f"b1 shape {self.b1.shape} != ({self.hidden_size},)")
        if self.w2.shape != (n_cls, self.hidden_size):
            raise ReaderError(
                f"w2 shape {self.w2.shape} != ({n_cls}, {self.hidden_size})"
            )
        if self.b2.shape != (n_cls,):
            raise ReaderError(f"b2 shape {self.b2.shape} != ({n_cls},)")

    def predict(
        self,
        features: NDArray[np.floating[Any]],
        *,
        charset: Charset = Charset.ANY,
    ) -> CharCellPrediction:
        """Return the best class under ``charset``; probability is the model softmax."""
        x = np.asarray(features, dtype=np.float64).reshape(-1)
        expected = self.input_size * self.input_size
        if x.shape != (expected,):
            raise ReaderError(f"expected {expected} features, got {x.shape[0]}")

        hidden = np.maximum(0.0, self.w1 @ x + self.b1)
        logits = self.w2 @ hidden + self.b2
        probs = _masked_softmax(logits, allowed_indices(self.classes, charset))
        idx = int(np.argmax(probs))
        return CharCellPrediction(
            character=self.classes[idx],
            probability=float(probs[idx]),
            probabilities=tuple(float(p) for p in probs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": WEIGHTS_VERSION,
            "classes": list(self.classes),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CharCellWeights:
        version = int(data.get("version", 0))
        if version != WEIGHTS_VERSION:
            raise ReaderError(f"unsupported char-cell weights version {version}")
        classes = tuple(str(c) for c in data["classes"])
        if not classes:
            raise ReaderError("char-cell weights have empty classes")
        return cls(
            classes=classes,
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            w1=np.asarray(data["w1"], dtype=np.float64),
            b1=np.asarray(data["b1"], dtype=np.float64),
            w2=np.asarray(data["w2"], dtype=np.float64),
            b2=np.asarray(data["b2"], dtype=np.float64),
        )


def train_mlp(
    samples: Sequence[tuple[NDArray[np.floating[Any]], str]],
    classes: Sequence[str] = DEFAULT_CLASSES,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    hidden_size: int = DEFAULT_HIDDEN,
    epochs: int = 40,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    seed: int = 0,
) -> CharCellWeights:
    """Fit the MLP with plain NumPy SGD (no new dependencies)."""
    if not samples:
        raise ReaderError("no training samples for char-cell model")

    class_list = tuple(classes)
    id_to_idx = {c: i for i, c in enumerate(class_list)}
    unknown = sorted({label for _, label in samples if label not in id_to_idx})
    if unknown:
        raise ReaderError(f"unknown labels in samples: {unknown}")

    n_in = input_size * input_size
    n_cls = len(class_list)
    rng = np.random.default_rng(seed)
    # He-ish init for ReLU.
    w1 = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(hidden_size, n_in))
    b1 = np.zeros(hidden_size, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(n_cls, hidden_size))
    b2 = np.zeros(n_cls, dtype=np.float64)

    xs = np.stack(
        [np.asarray(x, dtype=np.float64).reshape(-1) for x, _ in samples],
        axis=0,
    )
    ys = np.array([id_to_idx[label] for _, label in samples], dtype=np.int64)
    n = len(ys)
    order = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            xb = xs[batch]
            yb = ys[batch]
            # Forward
            z1 = xb @ w1.T + b1
            a1 = np.maximum(0.0, z1)
            logits = a1 @ w2.T + b2
            probs = _softmax_rows(logits)
            y_oh = np.zeros_like(probs)
            y_oh[np.arange(len(yb)), yb] = 1.0
            # Backward
            dlogits = (probs - y_oh) / max(len(yb), 1)
            dw2 = dlogits.T @ a1
            db2 = dlogits.sum(axis=0)
            da1 = dlogits @ w2
            dz1 = da1 * (z1 > 0.0)
            dw1 = dz1.T @ xb
            db1 = dz1.sum(axis=0)
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1

    return CharCellWeights(
        classes=class_list,
        input_size=input_size,
        hidden_size=hidden_size,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )


def save_weights(weights: CharCellWeights, path: Path) -> None:
    """Write weights JSON (or ``.npz`` sidecar arrays for large files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(
            path,
            version=np.asarray([WEIGHTS_VERSION], dtype=np.int32),
            input_size=np.asarray([weights.input_size], dtype=np.int32),
            hidden_size=np.asarray([weights.hidden_size], dtype=np.int32),
            classes=np.asarray(list(weights.classes)),
            w1=weights.w1,
            b1=weights.b1,
            w2=weights.w2,
            b2=weights.b2,
        )
        return
    path.write_text(
        json.dumps(weights.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_weights(path: Path) -> CharCellWeights:
    """Load weights from JSON or ``.npz`` under ``TORUS_MODEL_DIR``."""
    path = Path(path)
    if not path.is_file():
        raise ReaderError(f"char-cell weights not found: {path}")
    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReaderError(f"cannot read char-cell weights: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReaderError(f"invalid char-cell weights JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ReaderError(f"char-cell weights root must be an object: {path}")
    return CharCellWeights.from_dict(data)


def _load_npz(path: Path) -> CharCellWeights:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["version"]).reshape(-1)[0])
            if version != WEIGHTS_VERSION:
                raise ReaderError(f"unsupported char-cell weights version {version}")
            classes_arr = data["classes"]
            classes = tuple(str(c) for c in classes_arr.tolist())
            return CharCellWeights(
                classes=classes,
                input_size=int(np.asarray(data["input_size"]).reshape(-1)[0]),
                hidden_size=int(np.asarray(data["hidden_size"]).reshape(-1)[0]),
                w1=np.asarray(data["w1"], dtype=np.float64),
                b1=np.asarray(data["b1"], dtype=np.float64),
                w2=np.asarray(data["w2"], dtype=np.float64),
                b2=np.asarray(data["b2"], dtype=np.float64),
            )
    except OSError as exc:
        raise ReaderError(f"cannot read char-cell weights: {path}") from exc
    except KeyError as exc:
        raise ReaderError(f"incomplete char-cell weights archive: {path}") from exc


def _softmax_rows(logits: MatF64) -> MatF64:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _masked_softmax(logits: VecF64, allowed: Sequence[int]) -> VecF64:
    if not allowed:
        # No legal class under the hint — uniform over all (caller still picks argmax).
        n = logits.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    mask = np.full(logits.shape[0], -np.inf, dtype=np.float64)
    for i in allowed:
        mask[i] = 0.0
    masked = logits + mask
    shifted = masked - np.nanmax(masked)
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        out = np.zeros_like(logits)
        out[list(allowed)] = 1.0 / len(allowed)
        return out
    return exp / total
