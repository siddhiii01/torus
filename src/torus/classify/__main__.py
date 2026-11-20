"""CLI: ``python -m torus.classify train`` (S4).

Training labels and page images live under ``TORUS_DATA_DIR`` only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from torus.classify.features import feature_names
from torus.classify.trained import (
    build_training_sample,
    resolve_weights_path,
    save_weights,
    train_linear_classifier,
)
from torus.config import load_config
from torus.config.schemas import PageTypesConfig
from torus.core.errors import ClassifyError, ConfigError, TorusError
from torus.core.models import Page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torus.classify",
        description="Page-type classifier utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help=(
            "Train the small image+text classifier from a JSONL manifest "
            "(path, page_type, optional text). Paths must be under TORUS_DATA_DIR."
        ),
    )
    train.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL file: one object per line with path, page_type, optional text.",
    )
    train.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output weights JSON (relative paths resolve under TORUS_MODEL_DIR).",
    )
    train.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config_dir)
        page_types = cfg.page_types
        type_ids = [spec.id for spec in page_types.page_types]
        samples = _load_manifest_samples(args.manifest, page_types)
        names = feature_names(type_ids)
        weights = train_linear_classifier(
            samples,
            type_ids,
            names,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        out = resolve_weights_path(str(args.out)) or Path(args.out)
        save_weights(weights, out)
    except (ClassifyError, ConfigError, TorusError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote weights: {out}")
    print(f"samples: {len(samples)} types: {len(type_ids)}")
    return 0


def _load_manifest_samples(
    manifest: Path,
    page_types: PageTypesConfig,
) -> list[tuple[NDArray[np.float64], str]]:
    data_dir = _require_data_dir()
    samples: list[tuple[NDArray[np.float64], str]] = []
    for line_no, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text_line = line.strip()
        if not text_line:
            continue
        try:
            row = json.loads(text_line)
        except json.JSONDecodeError as exc:
            raise ClassifyError(f"manifest line {line_no}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ClassifyError(f"manifest line {line_no}: expected object")
        rel = str(row.get("path", "")).strip()
        label = str(row.get("page_type", "")).strip()
        page_text = str(row.get("text", "") or "")
        if not rel or not label:
            raise ClassifyError(
                f"manifest line {line_no}: path and page_type are required"
            )
        image_path = Path(rel)
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        _assert_under_data_dir(image_path, data_dir)
        if not image_path.is_file():
            raise ClassifyError(f"manifest line {line_no}: missing image {image_path}")
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            page = Page(index=0, width=rgb.width, height=rgb.height)
            page.bind_image(rgb.copy())
        samples.append(build_training_sample(page, page_types, label, page_text))
    if not samples:
        raise ClassifyError("manifest contained no samples")
    return samples


def _require_data_dir() -> Path:
    import os

    raw = os.environ.get("TORUS_DATA_DIR", "").strip()
    if not raw:
        raise ClassifyError("TORUS_DATA_DIR must be set to load training images")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise ClassifyError(f"TORUS_DATA_DIR is not a directory: {path}")
    return path


def _assert_under_data_dir(path: Path, data_dir: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(data_dir)
    except ValueError as exc:
        raise ClassifyError(
            f"training image must be under TORUS_DATA_DIR: {resolved}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
