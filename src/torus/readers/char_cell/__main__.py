"""CLI: ``python -m torus.readers.char_cell train``.

Train the character-cell MLP on synthetic glyphs and/or public EMNIST.
Weights go under ``TORUS_MODEL_DIR`` (never inside the git checkout).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from torus.core.errors import ReaderError, TorusError
from torus.readers.char_cell.alphabet import DEFAULT_CLASSES
from torus.readers.char_cell.model import (
    DEFAULT_HIDDEN,
    save_weights,
    train_mlp,
)
from torus.readers.char_cell.preprocess import DEFAULT_INPUT_SIZE
from torus.readers.char_cell.train_data import (
    Sample,
    load_emnist_samples,
    load_synth_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torus.readers.char_cell",
        description="Character-cell classifier utilities (P5.5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help=(
            "Train the small cell MLP. Default source is synthetic glyphs; "
            "add --source emnist when the train extra is installed."
        ),
    )
    train.add_argument(
        "--source",
        choices=("synth", "emnist", "both"),
        default="synth",
        help="Training data source (default: synth).",
    )
    train.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output weights path (.json or .npz). Relative paths resolve "
            "under TORUS_MODEL_DIR."
        ),
    )
    train.add_argument("--per-class", type=int, default=24, help="Synth samples/class.")
    train.add_argument(
        "--emnist-max-per-class",
        type=int,
        default=200,
        help="Cap EMNIST samples per class (default 200).",
    )
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    train.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
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
        samples = _collect_samples(args)
        weights = train_mlp(
            samples,
            DEFAULT_CLASSES,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        out = _resolve_out(args.out)
        save_weights(weights, out)
    except TorusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({len(weights.classes)} classes, {len(samples)} samples)")
    return 0


def _collect_samples(args: argparse.Namespace) -> list[Sample]:
    samples = []
    if args.source in ("synth", "both"):
        samples.extend(
            load_synth_samples(
                classes=DEFAULT_CLASSES,
                per_class=args.per_class,
                seed=args.seed,
                input_size=args.input_size,
            )
        )
    if args.source in ("emnist", "both"):
        samples.extend(
            load_emnist_samples(
                max_per_class=args.emnist_max_per_class,
                seed=args.seed,
                input_size=args.input_size,
            )
        )
    if not samples:
        raise ReaderError("no training samples collected")
    return samples


def _resolve_out(path: Path) -> Path:
    if path.is_absolute():
        return path
    root = os.environ.get("TORUS_MODEL_DIR", "").strip()
    if not root:
        raise ReaderError(
            "relative --out requires TORUS_MODEL_DIR (or pass an absolute path)"
        )
    return Path(root).expanduser().resolve() / path


if __name__ == "__main__":
    raise SystemExit(main())
