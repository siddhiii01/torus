"""Command-line entrypoint for Torus (``torus`` / ``python -m torus``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from torus import __version__
from torus.core.errors import ConfigError, EvalError, TorusError
from torus.doctor.report import format_report, run_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torus",
        description="Torus local document engine CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor",
        help="Detect hardware, extras and which readers can run here.",
    )
    doctor.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )

    label = sub.add_parser(
        "label",
        help=(
            "Label a local PDF into TORUS_DATA_DIR/testsets/<set>/<case>/ "
            "(writes source.pdf, meta.json, expected.json). No upload."
        ),
    )
    label.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Local path to the source PDF (copied into the case folder).",
    )
    label.add_argument(
        "--set",
        dest="set_id",
        required=True,
        help="Test-set id (folder under TORUS_DATA_DIR/testsets/).",
    )
    label.add_argument(
        "--case",
        dest="case_id",
        required=True,
        help="Case id (folder name; must match meta.case_id).",
    )
    label.add_argument(
        "--template",
        dest="template_id",
        required=True,
        help="Template id (config/templates/<id>.yaml).",
    )
    label.add_argument(
        "--split",
        choices=["dev", "hidden"],
        default="dev",
        help="Eval split for meta.json (default: dev).",
    )
    label.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )
    label.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional tag: document year.",
    )
    label.add_argument(
        "--scan-type",
        choices=["digital", "scan", "photo", "fax", "mixed"],
        default=None,
        help="Optional tag: how the pack was captured.",
    )
    label.add_argument(
        "--handwriting",
        dest="handwriting_level",
        choices=["none", "light", "medium", "heavy"],
        default=None,
        help="Optional tag: handwriting density.",
    )
    label.add_argument(
        "--page-count",
        type=int,
        default=None,
        help="Optional expected page count for meta.json.",
    )
    label.add_argument(
        "--notes",
        default="",
        help="Operator notes (quality/split only — not field values).",
    )
    label.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of gold rows to prompt for (ignored with --rows-json).",
    )
    label.add_argument(
        "--rows-json",
        type=Path,
        default=None,
        help="Local JSON file: list of row objects (skips interactive prompts).",
    )
    label.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing case folder.",
    )
    label.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the PDF in the system viewer.",
    )

    run = sub.add_parser(
        "run",
        help="Run a file or folder through the pipeline and write the Excel.",
    )
    run.add_argument("source", type=Path, help="A PDF/image, or a folder of them.")
    run.add_argument(
        "--template",
        dest="template_id",
        required=True,
        help="Template id (config/templates/<id>.yaml).",
    )
    run.add_argument("--out", type=Path, required=True, help="Output .xlsx path.")
    run.add_argument(
        "--template-xlsx",
        type=Path,
        default=None,
        help="Client workbook to fill a copy of (default: headers from the template).",
    )
    run.add_argument(
        "--calibrator",
        type=Path,
        default=None,
        help=(
            "Fitted calibrator JSON (default: TORUS_MODEL_DIR/calibration/"
            "calibrator.json when present). Without one nothing is confirmed."
        ),
    )
    run.add_argument(
        "--config-dir",
        default=None,
        help="Config directory (default: TORUS_CONFIG_DIR or repo config/).",
    )

    review = sub.add_parser(
        "review",
        help="Open a review bundle in the local review UI (127.0.0.1 only).",
    )
    review.add_argument("bundle", type=Path, help="Review bundle JSON.")
    review.add_argument("--port", type=int, default=8765, help="Local port.")
    review.add_argument(
        "--reviewer",
        default=None,
        help="Reviewer stored with each decision (default: TORUS_REVIEWER, OS user).",
    )

    promote = sub.add_parser(
        "promote-review",
        help="Promote a reviewed pack into the test set (TORUS_DATA_DIR/testsets).",
    )
    promote.add_argument("bundle", type=Path, help="Review bundle JSON.")
    promote.add_argument(
        "--template", dest="template_id", required=True, help="Template id."
    )
    promote.add_argument("--set", dest="set_id", required=True, help="Test-set id.")
    promote.add_argument("--case", dest="case_id", required=True, help="Case id.")
    promote.add_argument("--split", choices=["dev", "hidden"], default="dev")
    promote.add_argument(
        "--include-unreviewed-confirmed",
        action="store_true",
        help="Also use engine-confirmed values nobody reviewed as gold (biases eval).",
    )
    promote.add_argument("--force", action="store_true", help="Overwrite the case.")
    promote.add_argument("--config-dir", default=None, help="Config directory.")

    reexport = sub.add_parser(
        "reexport",
        help="Regenerate the Excel from a review bundle with stored corrections.",
    )
    reexport.add_argument("bundle", type=Path, help="Review bundle JSON.")
    reexport.add_argument(
        "--template", dest="template_id", required=True, help="Template id."
    )
    reexport.add_argument("--out", type=Path, required=True, help="Output .xlsx path.")
    reexport.add_argument(
        "--template-xlsx",
        type=Path,
        default=None,
        help="Client workbook to fill a copy of (default: headers from the template).",
    )
    reexport.add_argument("--config-dir", default=None, help="Config directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "doctor":
        return _cmd_doctor(args.config_dir)
    if args.command == "label":
        return _cmd_label(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "reexport":
        return _cmd_reexport(args)
    if args.command == "promote-review":
        return _cmd_promote_review(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_review(args: argparse.Namespace) -> int:
    from torus.review.server import serve
    from torus.review.store import CorrectionStore

    try:
        store = CorrectionStore.from_env(args.reviewer)
        return serve(args.bundle, port=args.port, sink=store)
    except TorusError as exc:
        print(f"torus review: {exc}", file=sys.stderr)
        return 1


def _cmd_reexport(args: argparse.Namespace) -> int:
    from torus.config import load_config
    from torus.output.excel import write_blank_template
    from torus.review.bundle import load_bundle
    from torus.review.reexport import reexport
    from torus.review.store import CorrectionStore

    try:
        config = load_config(args.config_dir)
        try:
            template = config.template(args.template_id)
        except KeyError as exc:
            raise ConfigError(str(exc.args[0])) from exc
        bundle = load_bundle(args.bundle)
        template_xlsx = args.template_xlsx
        if template_xlsx is None:
            template_xlsx = write_blank_template(
                template, args.out.with_name(f"{args.out.stem}.template.xlsx")
            )
        summary = reexport(
            template_xlsx,
            args.out,
            template,
            [bundle.result],
            CorrectionStore.from_env(),
            config.pipeline.output,
        )
    except ConfigError as exc:
        print(f"torus reexport: config error: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus reexport: {exc}", file=sys.stderr)
        return 1

    # Counts and ids only — never field values.
    print(
        f"corrected={summary.corrected} accepted={summary.accepted} "
        f"skipped={summary.skipped} unapplied={len(summary.unapplied)}"
    )
    for item_id in summary.unapplied:
        print(f"  unapplied decision: {item_id}", file=sys.stderr)
    print(f"  {summary.write.path}")
    return 0


def _cmd_promote_review(args: argparse.Namespace) -> int:
    from torus.config import load_config
    from torus.eval.schema import Split
    from torus.review.bundle import load_bundle
    from torus.review.promote import promote_reviewed
    from torus.review.store import CorrectionStore

    try:
        config = load_config(args.config_dir)
        try:
            template = config.template(args.template_id)
        except KeyError as exc:
            raise ConfigError(str(exc.args[0])) from exc
        bundle = load_bundle(args.bundle)
        written = promote_reviewed(
            bundle.result,
            Path(bundle.source_path),
            CorrectionStore.from_env("promote"),
            template,
            set_id=args.set_id,
            case_id=args.case_id,
            split=Split(args.split),
            include_unreviewed_confirmed=args.include_unreviewed_confirmed,
            force=args.force,
        )
    except ConfigError as exc:
        print(f"torus promote-review: config error: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus promote-review: {exc}", file=sys.stderr)
        return 1
    print(f"promoted case {written.set_id}/{written.case_id} under TORUS_DATA_DIR")
    return 0


def _cmd_doctor(config_dir: str | None) -> int:
    try:
        report = run_doctor(config_dir=config_dir)
    except ConfigError as exc:
        print(f"torus doctor: config error: {exc}", file=sys.stderr)
        return 1
    except TorusError as exc:
        print(f"torus doctor: {exc}", file=sys.stderr)
        return 1
    print(format_report(report), end="")
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    from torus.config import default_config_dir
    from torus.eval.label import (
        LabelCaseRequest,
        load_rows_json,
        open_pdf_locally,
        prompt_rows_interactive,
        resolve_template,
        write_labelled_case,
    )
    from torus.eval.schema import HandwritingLevel, ScanType, Split

    try:
        config_root = (
            Path(args.config_dir).expanduser()
            if args.config_dir
            else default_config_dir()
        )
        template = resolve_template(args.template_id, config_dir=config_root)

        if not args.no_open:
            open_pdf_locally(args.pdf)

        if args.rows_json is not None:
            rows = load_rows_json(Path(args.rows_json))
        else:
            print(
                f"Labelling {args.case_id!r} with template {template.id!r} "
                f"({len(template.scored_fields())} columns). "
                "Enter = omit column, - = explicit null."
            )
            rows = prompt_rows_interactive(template, row_count=args.rows)

        result = write_labelled_case(
            LabelCaseRequest(
                pdf_path=Path(args.pdf),
                set_id=args.set_id,
                case_id=args.case_id,
                template=template,
                split=Split(args.split),
                rows=rows,
                year=args.year,
                scan_type=ScanType(args.scan_type) if args.scan_type else None,
                handwriting_level=(
                    HandwritingLevel(args.handwriting_level)
                    if args.handwriting_level
                    else None
                ),
                page_count_expected=args.page_count,
                notes=args.notes,
                force=args.force,
            )
        )
    except ConfigError as exc:
        print(f"torus label: config error: {exc}", file=sys.stderr)
        return 2
    except EvalError as exc:
        print(f"torus label: {exc}", file=sys.stderr)
        return 1
    except TorusError as exc:
        print(f"torus label: {exc}", file=sys.stderr)
        return 1

    # Paths and ids only — never print field values.
    print(
        f"wrote case {result.set_id}/{result.case_id} "
        f"(template={result.template_id}) under TORUS_DATA_DIR"
    )
    print(f"  {result.path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from torus.config import load_config
    from torus.consensus.calibrate import Calibrator
    from torus.doctor.weights import model_dir
    from torus.output.excel import write_blank_template, write_workbook
    from torus.readers import ReaderRegistry, ReaderRunner
    from torus.service.run import (
        RunContext,
        RunnerReaders,
        collect_inputs,
        default_llm_client,
        run_files,
    )

    try:
        config = load_config(args.config_dir)
        try:
            template = config.template(args.template_id)
        except KeyError as exc:
            raise ConfigError(str(exc.args[0])) from exc

        calibrator_path = args.calibrator
        if calibrator_path is None:
            root = model_dir()
            default = (
                root / "calibration" / "calibrator.json" if root is not None else None
            )
            calibrator_path = default if default and default.is_file() else None
        calibrator = Calibrator.load(calibrator_path) if calibrator_path else None

        inputs = collect_inputs(args.source)
        if not inputs:
            print("torus run: no supported files found", file=sys.stderr)
            return 1

        runner = ReaderRunner(ReaderRegistry(config.readers))
        ctx = RunContext(
            config=config,
            template=template,
            readers=RunnerReaders(runner),
            llm_client=default_llm_client(config),
            calibrator=calibrator,
        )
        report = run_files(
            inputs, ctx, progress=lambda m: print(m, file=sys.stderr, flush=True)
        )

        template_xlsx = args.template_xlsx
        if template_xlsx is None:
            template_xlsx = write_blank_template(
                template, args.out.with_name(f"{args.out.stem}.template.xlsx")
            )
        summary = write_workbook(
            template_xlsx,
            args.out,
            template,
            report.results,
            config.pipeline.output,
        )
    except ConfigError as exc:
        print(f"torus run: config error: {exc}", file=sys.stderr)
        return 2
    except TorusError as exc:
        print(f"torus run: {exc}", file=sys.stderr)
        return 1

    # Counts and paths only — never field values.
    print(
        f"files ok={report.files_ok} failed={report.files_failed}; "
        f"rows={summary.rows} written={summary.written} "
        f"protected={summary.skipped_protected} "
        f"not_written_by_status={summary.skipped_status} "
        f"empty={summary.skipped_empty}"
    )
    print(f"  {summary.path}")
    for line in report.failed_files:
        print(f"  failed: {line}", file=sys.stderr)
    return 1 if report.files_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
