"""CLI entrypoint: ``python -m torus.eval``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from torus.config import default_config_dir, load_pipeline
from torus.core.errors import ConfigError, EvalError, TorusError
from torus.eval.diff import compare_runs, format_diff_markdown
from torus.eval.report import load_run_report
from torus.eval.review_metrics import ReviewMetrics
from torus.eval.schema import Split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m torus.eval",
        description=(
            "Torus evaluation harness (run, classify, locate, score report, "
            "regression diff, legacy promote, label)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help=(
            "Score a labelled test set against prediction JSON and write "
            "Markdown/HTML/JSON reports (metrics only)."
        ),
    )
    run.add_argument(
        "--set",
        dest="set_id",
        required=True,
        help="Test-set id under TORUS_DATA_DIR/testsets/<set>/.",
    )
    run.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory of {case_id}.json (or {case_id}_{variant}.json) preds.",
    )
    run.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Markdown report path (HTML + JSON sidecar written beside it).",
    )
    run.add_argument(
        "--split",
        choices=[s.value for s in Split],
        default=Split.DEV.value,
        help="Only score cases with this split (default: dev). Use '' for all.",
    )
    run.add_argument(
        "--all-splits",
        action="store_true",
        help="Score every case in the set regardless of split.",
    )
    run.add_argument(
        "--template",
        default=None,
        help="Template id (default: unique template_id from case meta).",
    )
    run.add_argument(
        "--pred-variant",
        default=None,
        help="Prefer {case_id}_{variant}.json when present (legacy bake-offs).",
    )
    run.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    run.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override TORUS_DATA_DIR for this run.",
    )
    run.add_argument(
        "--model-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="Record a model/version pair on the report (repeatable).",
    )
    run.add_argument(
        "--require-predictions",
        action="store_true",
        help="Fail if any selected case is missing a prediction file.",
    )
    run.add_argument(
        "--empty-pred-when-missing",
        action="store_true",
        help="Score missing predictions as zero rows instead of skipping.",
    )
    run.add_argument(
        "--review-bundle",
        dest="review_bundles",
        action="append",
        type=Path,
        default=[],
        help=(
            "Review bundle JSON (repeatable). Adds review effort from the stored "
            "decisions under TORUS_DATA_DIR to the report."
        ),
    )

    diff = sub.add_parser(
        "diff",
        help=(
            "Compare two eval run JSON reports; fail if silent errors rise "
            "or a column drops beyond the noise margin."
        ),
    )
    diff.add_argument(
        "baseline",
        type=Path,
        help="Baseline EvalRunReport JSON (e.g. last report on main).",
    )
    diff.add_argument(
        "candidate",
        type=Path,
        help="Candidate EvalRunReport JSON (current run).",
    )
    diff.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    diff.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the Markdown diff summary.",
    )

    promote = sub.add_parser(
        "promote-legacy",
        help=(
            "Copy legacy gold.json cases into testsets/<set>/ (local data dir "
            "only; for P1.6 baseline)."
        ),
    )
    promote.add_argument(
        "--set",
        dest="set_id",
        default="deb_dev",
        help="Destination test-set id (default: deb_dev).",
    )
    promote.add_argument(
        "--template",
        default="deb_v1",
        help="Template id written into meta/expected (default: deb_v1).",
    )
    promote.add_argument(
        "--legacy-cases-dir",
        type=Path,
        default=None,
        help="Override legacy cases root (default: $TORUS_DATA_DIR/legacy/deb/cases).",
    )
    promote.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override TORUS_DATA_DIR for this run.",
    )
    promote.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing promoted cases.",
    )

    classify = sub.add_parser(
        "classify",
        help=(
            "Score page-type predictions against gold pages.json and write "
            "a report with per-type P/R/F1, confusion matrix, and "
            "misclassifications by gold type (P3.5)."
        ),
    )
    classify.add_argument(
        "--set",
        dest="set_id",
        required=True,
        help="Test-set id under TORUS_DATA_DIR/testsets/<set>/.",
    )
    classify.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory of {case_id}.pages.json (or nested case/pages.json).",
    )
    classify.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Markdown report path (HTML + JSON sidecar written beside it).",
    )
    classify.add_argument(
        "--split",
        choices=[s.value for s in Split],
        default=Split.DEV.value,
        help="Only score cases with this split (default: dev).",
    )
    classify.add_argument(
        "--all-splits",
        action="store_true",
        help="Score every case in the set regardless of split.",
    )
    classify.add_argument(
        "--template",
        default=None,
        help="Template id (default: unique template_id from case meta).",
    )
    classify.add_argument(
        "--pred-variant",
        default=None,
        help="Prefer {case_id}_{variant}.pages.json when present.",
    )
    classify.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    classify.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override TORUS_DATA_DIR for this run.",
    )
    classify.add_argument(
        "--model-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="Record a model/version pair on the report (repeatable).",
    )
    classify.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip cases missing gold pages.json or predictions instead of failing.",
    )

    locate = sub.add_parser(
        "locate",
        help=(
            "Score field-location predictions against gold locations.json "
            "(IoU match per column, before reading; P4.8)."
        ),
    )
    locate.add_argument(
        "--set",
        dest="set_id",
        required=True,
        help="Test-set id under TORUS_DATA_DIR/testsets/<set>/.",
    )
    locate.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory of {case_id}.locations.json (or nested case/locations.json).",
    )
    locate.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Markdown report path (HTML + JSON sidecar written beside it).",
    )
    locate.add_argument(
        "--split",
        choices=[s.value for s in Split],
        default=Split.DEV.value,
        help="Only score cases with this split (default: dev).",
    )
    locate.add_argument(
        "--all-splits",
        action="store_true",
        help="Score every case in the set regardless of split.",
    )
    locate.add_argument(
        "--template",
        default=None,
        help="Template id (default: unique template_id from case meta).",
    )
    locate.add_argument(
        "--pred-variant",
        default=None,
        help="Prefer {case_id}_{variant}.locations.json when present.",
    )
    locate.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    locate.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override TORUS_DATA_DIR for this run.",
    )
    locate.add_argument(
        "--model-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="Record a model/version pair on the report (repeatable).",
    )
    locate.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Override layout.locate_iou from pipeline.yaml for this run.",
    )
    locate.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip cases missing gold locations.json or predictions instead of failing."
        ),
    )

    readers = sub.add_parser(
        "readers",
        help=(
            "Reader bake-off (P5.8): per reader x field type accuracy and speed "
            "on gold value boxes; proposes a default set (never edits config)."
        ),
    )
    readers.add_argument("--set", dest="set_id", required=True, help="Test-set id.")
    readers.add_argument(
        "--report", type=Path, required=True, help="Markdown report path (+ .json)."
    )
    readers.add_argument(
        "--reader",
        action="append",
        default=[],
        dest="reader_names",
        metavar="NAME",
        help="Reader to run (repeatable). Default: every reader that can run here.",
    )
    readers.add_argument(
        "--split",
        choices=[s.value for s in Split],
        default=Split.DEV.value,
        help="Only cases with this split (default: dev).",
    )
    readers.add_argument("--all-splits", action="store_true")
    readers.add_argument("--template", default=None, help="Template id.")
    readers.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Fewest readings per reader and field type to qualify for a proposal.",
    )
    readers.add_argument("--config-dir", default=None)
    readers.add_argument("--data-dir", type=Path, default=None)

    onboarding = sub.add_parser(
        "onboarding",
        help="Record or summarise time-to-onboard per new template (P12.4).",
    )
    onboarding.add_argument("action", choices=["record", "summary"])
    onboarding.add_argument("template_id", nargs="?")
    onboarding.add_argument("milestone", nargs="?")
    onboarding.add_argument("--log", type=Path, default=None, help="JSONL log path.")

    sub.add_parser(
        "label",
        help="Delegate to ``torus label`` (local PDF → expected.json).",
    )
    return parser


def _parse_model_versions(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise EvalError(f"invalid --model-version {item!r}; expected NAME=VERSION")
        name, _, version = item.partition("=")
        name, version = name.strip(), version.strip()
        if not name or not version:
            raise EvalError(f"invalid --model-version {item!r}; expected NAME=VERSION")
        out[name] = version
    return out


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    # ``label`` forwards remaining args to the main CLI so flags stay in one place.
    if argv_list is not None and argv_list and argv_list[0] == "label":
        from torus.cli import main as torus_main

        return torus_main(["label", *argv_list[1:]])
    if argv_list is None and len(sys.argv) > 1 and sys.argv[1] == "label":
        from torus.cli import main as torus_main

        return torus_main(["label", *sys.argv[2:]])

    parser = build_parser()
    args = parser.parse_args(argv_list)

    if args.command == "diff":
        return _cmd_diff(
            baseline=args.baseline,
            candidate=args.candidate,
            config_dir=args.config_dir,
            out=args.out,
        )
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "locate":
        return _cmd_locate(args)
    if args.command == "promote-legacy":
        return _cmd_promote(args)
    if args.command == "readers":
        return _cmd_readers(args)
    if args.command == "onboarding":
        return _cmd_onboarding(args)
    if args.command == "label":
        from torus.cli import main as torus_main

        return torus_main(["label"])

    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_onboarding(args: argparse.Namespace) -> int:
    from torus.eval import onboarding

    log = args.log if args.log is not None else onboarding.DEFAULT_LOG
    try:
        if args.action == "record":
            if not args.template_id or not args.milestone:
                print("error: record needs <template_id> <milestone>", file=sys.stderr)
                return 2
            onboarding.record(args.template_id, args.milestone, log=log)
            return 0
        print(onboarding.format_summary(onboarding.load_times(log)))
    except TorusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from torus.eval.run import run_evaluation

    split: Split | None = None if args.all_splits else Split(args.split)

    try:
        model_versions = _parse_model_versions(args.model_version)
        review = _review_metrics(args) if args.review_bundles else None
        result = run_evaluation(
            set_id=args.set_id,
            pred_dir=args.pred_dir,
            report_path=args.report,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            split=split,
            template_id=args.template,
            pred_variant=args.pred_variant,
            model_versions=model_versions,
            require_predictions=args.require_predictions,
            empty_pred_when_missing=args.empty_pred_when_missing,
            review=review,
        )
    except ConfigError as exc:
        print(f"torus.eval run: config error: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"torus.eval run: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus.eval run: {exc}", file=sys.stderr)
        return 2

    report = result.report
    micro = report.scores.fields_micro
    print(
        f"set={report.set_id} cases={report.case_count} "
        f"field_acc={micro.accuracy:.4f} field_f1={micro.f1:.4f} "
        f"silent_errors={report.scores.silent_errors} "
        f"row_f1={report.scores.rows.f1:.4f}"
    )
    if result.skipped_missing_pred:
        print(
            "skipped_missing_pred=" + ",".join(result.skipped_missing_pred),
            file=sys.stderr,
        )
    if result.paths.json_path is not None:
        print(f"report_json={result.paths.json_path}")
    print(f"report_md={result.paths.markdown}")
    return 0


def _review_metrics(args: argparse.Namespace) -> ReviewMetrics:
    from torus.config import load_config
    from torus.review.metrics import review_metrics_from_bundles
    from torus.review.store import CorrectionStore

    config = load_config(args.config_dir)
    return review_metrics_from_bundles(
        args.review_bundles,
        CorrectionStore.from_env("metrics"),
        max_gap_s=config.pipeline.review.max_gap_s,
    )


def _cmd_classify(args: argparse.Namespace) -> int:
    from torus.eval.run import run_classification_evaluation

    split: Split | None = None if args.all_splits else Split(args.split)

    try:
        model_versions = _parse_model_versions(args.model_version)
        result = run_classification_evaluation(
            set_id=args.set_id,
            pred_dir=args.pred_dir,
            report_path=args.report,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            split=split,
            template_id=args.template,
            pred_variant=args.pred_variant,
            model_versions=model_versions,
            require_page_labels=not args.allow_missing,
        )
    except ConfigError as exc:
        print(f"torus.eval classify: config error: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"torus.eval classify: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus.eval classify: {exc}", file=sys.stderr)
        return 2

    report = result.report
    clf = report.classification
    if clf is None:
        print(f"set={report.set_id} cases={report.case_count} classification=none")
    else:
        print(
            f"set={report.set_id} cases={report.case_count} "
            f"page_acc={clf.accuracy:.4f} "
            f"pages={clf.correct}/{clf.total} "
            f"misclass_types={len(clf.misclassifications)}"
        )
    if result.skipped_missing_pred:
        print(
            "skipped=" + ",".join(result.skipped_missing_pred),
            file=sys.stderr,
        )
    if result.paths.json_path is not None:
        print(f"report_json={result.paths.json_path}")
    print(f"report_md={result.paths.markdown}")
    return 0


def _cmd_locate(args: argparse.Namespace) -> int:
    from torus.eval.run import run_location_evaluation

    split: Split | None = None if args.all_splits else Split(args.split)

    try:
        model_versions = _parse_model_versions(args.model_version)
        result = run_location_evaluation(
            set_id=args.set_id,
            pred_dir=args.pred_dir,
            report_path=args.report,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            split=split,
            template_id=args.template,
            pred_variant=args.pred_variant,
            model_versions=model_versions,
            require_locations=not args.allow_missing,
            iou_threshold=args.iou,
        )
    except ConfigError as exc:
        print(f"torus.eval locate: config error: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"torus.eval locate: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus.eval locate: {exc}", file=sys.stderr)
        return 2

    report = result.report
    loc = report.location
    if loc is None:
        print(f"set={report.set_id} cases={report.case_count} location=none")
    else:
        print(
            f"set={report.set_id} cases={report.case_count} "
            f"located_rate={loc.located_rate:.4f} "
            f"located={loc.located}/{loc.expected} "
            f"mean_iou={loc.mean_iou:.4f} "
            f"iou_threshold={loc.iou_threshold:.2f}"
        )
    if result.skipped_missing_pred:
        print(
            "skipped=" + ",".join(result.skipped_missing_pred),
            file=sys.stderr,
        )
    if result.paths.json_path is not None:
        print(f"report_json={result.paths.json_path}")
    print(f"report_md={result.paths.markdown}")
    return 0


def _cmd_readers(args: argparse.Namespace) -> int:
    from torus.eval.readers_bakeoff import run_bakeoff

    split: Split | None = None if args.all_splits else Split(args.split)
    try:
        report, (md_path, json_path) = run_bakeoff(
            set_id=args.set_id,
            report_path=args.report,
            reader_names=args.reader_names or None,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            split=split,
            template_id=args.template,
            min_samples=args.min_samples,
        )
    except TorusError as exc:  # ConfigError and EvalError are TorusErrors
        print(f"torus.eval readers: {exc}", file=sys.stderr)
        return 2

    print(
        f"set={report.set_id} cases={report.case_count} "
        f"fields={report.field_samples} readers_run={len(report.readers_run)} "
        f"readers_skipped={len(report.readers_skipped)} "
        f"proposed={','.join(report.proposed_enabled) or 'none'}"
    )
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    from torus.eval.promote_legacy import promote_legacy_set

    try:
        results = promote_legacy_set(
            legacy_cases_dir=args.legacy_cases_dir,
            set_id=args.set_id,
            template_id=args.template,
            data_dir=args.data_dir,
            force=args.force,
        )
    except EvalError as exc:
        print(f"torus.eval promote-legacy: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus.eval promote-legacy: {exc}", file=sys.stderr)
        return 2

    created = sum(1 for r in results if r.created)
    print(
        f"promoted={len(results)} created={created} "
        f"set={args.set_id} template={args.template}"
    )
    for item in results:
        flag = "created" if item.created else "exists"
        print(f"  {item.case_id}: {flag}")
    return 0


def _cmd_diff(
    *,
    baseline: Path,
    candidate: Path,
    config_dir: str | None,
    out: Path | None,
) -> int:
    try:
        root = Path(config_dir).expanduser() if config_dir else default_config_dir()
        gates = load_pipeline(root / "pipeline.yaml").regression
        base_report = load_run_report(baseline)
        cand_report = load_run_report(candidate)
        result = compare_runs(base_report, cand_report, gates=gates)
    except ConfigError as exc:
        print(f"torus.eval diff: config error: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"torus.eval diff: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus.eval diff: {exc}", file=sys.stderr)
        return 2

    md = format_diff_markdown(result)
    print(md, end="")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
