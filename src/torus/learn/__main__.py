"""``python -m torus.learn export``: corrections → training set (SKILLS S7)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from torus.core.errors import TorusError
from torus.learn.dataset import (
    TRAINING_SUBDIR,
    SampleKind,
    build_dataset,
    bundle_sources,
    ingest_pages,
    load_records,
)
from torus.review.store import CorrectionStore, data_dir


def _parse_since(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m torus.learn")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Build a training set from review corrections.")
    exp.add_argument("--kind", type=SampleKind, choices=list(SampleKind), required=True)
    exp.add_argument("--since", type=_parse_since, default=None, help="ISO date/time.")
    exp.add_argument("--name", default=None, help="Folder name under training/.")
    exp.add_argument("--bundles", type=Path, default=None, help="Bundle JSON folder.")
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--val-fraction", type=float, default=0.1)
    exp.add_argument("--test-fraction", type=float, default=0.1)
    exp.add_argument(
        "--exclude-document",
        action="append",
        default=[],
        help="Document id to keep out (repeat; use for hidden-set documents).",
    )
    exp.add_argument(
        "--corrected-only",
        action="store_true",
        help="Use only corrected values, not accepted suggestions.",
    )
    reg = sub.add_parser(
        "register", help="Register fine-tuned weights as a new version."
    )
    reg.add_argument("reader")
    reg.add_argument("weights", type=Path)
    reg.add_argument("--base-model", required=True)
    reg.add_argument("--recipe", required=True)
    reg.add_argument(
        "--data-hash", required=True, help="Hash from the dataset manifest."
    )
    reg.add_argument("--metrics", type=Path, default=None, help="JSON {name: number}.")
    lst = sub.add_parser("registry", help="List registered versions of a reader.")
    lst.add_argument("reader")
    pro = sub.add_parser(
        "promote", help="Promote a version if the hidden set improves."
    )
    pro.add_argument("reader")
    pro.add_argument("version")
    pro.add_argument(
        "--baseline", type=Path, required=True, help="Hidden-set report JSON."
    )
    pro.add_argument(
        "--candidate", type=Path, required=True, help="Hidden-set report JSON."
    )
    return parser


def _registry_command(args: argparse.Namespace) -> int:
    import json

    from torus.config import load_config
    from torus.eval.report import load_run_report
    from torus.learn.promotion import promote
    from torus.learn.registry import ModelRegistry

    registry = ModelRegistry.from_env()
    if args.command == "register":
        metrics = json.loads(args.metrics.read_text("utf-8")) if args.metrics else {}
        entry = registry.register(
            args.reader,
            args.weights,
            base_model=args.base_model,
            recipe=args.recipe,
            data_hash=args.data_hash,
            metrics=metrics,
        )
        print(f"registered {entry.reader} version {entry.version}")
    elif args.command == "registry":
        current = registry.current(args.reader)
        for version in registry.versions(args.reader):
            entry = registry.get(args.reader, version)
            mark = " (current)" if current and current.version == version else ""
            print(
                f"{version}{mark} data_hash={entry.data_hash} base={entry.base_model}"
            )
    else:
        decision = promote(
            registry,
            args.reader,
            args.version,
            load_run_report(args.baseline),
            load_run_report(args.candidate),
            gates=load_config().pipeline.regression,
        )
        print(f"promoted={decision.promoted} accuracy={decision.accuracy}")
        print(
            f"  automatic_rate={decision.automatic_rate} "
            f"silent={decision.silent_errors}"
        )
        for reason in decision.reasons:
            print(f"  - {reason}")
        if decision.calibration_pending:
            print("  next: re-fit the calibrator (P6.3) before using these weights")
        return 0 if decision.promoted else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command != "export":
            return _registry_command(args)
        root = data_dir()
        bundles = args.bundles or root / "bundles"
        sources = bundle_sources(bundles)
        store = CorrectionStore.from_env()
        corrections = root / "corrections"
        ids = sorted(p.stem for p in corrections.glob("*.jsonl"))
        name = args.name or f"{args.kind.value}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        out_dir = root / TRAINING_SUBDIR / name
        manifest = build_dataset(
            load_records(store, ids),
            ingest_pages(sources),
            out_dir,
            kind=args.kind,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            since=args.since,
            include_accepted=not args.corrected_only,
            exclude_documents=args.exclude_document,
        )
    except (TorusError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Counts and hash only: never label text.
    print(
        f"dataset {os.fspath(Path(TRAINING_SUBDIR) / name)} hash={manifest.data_hash}"
    )
    print(f"  counts={manifest.counts} skipped={manifest.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
