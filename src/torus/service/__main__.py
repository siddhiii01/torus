"""``python -m torus.service {serve,worker}``.

The pipeline runner is supplied as ``package.module:factory`` so the service
never imports pipeline internals itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path

from torus.service.audit import AuditLog
from torus.service.runner import PipelineRunner
from torus.service.settings import load_settings
from torus.service.store import JobStore


def load_runner(spec: str) -> PipelineRunner:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit("--runner must look like package.module:factory")
    factory = getattr(importlib.import_module(module_name), attr)
    runner: PipelineRunner = factory()
    return runner


def _bundle_command(args: argparse.Namespace) -> int:
    from torus.config.loader import default_config_dir
    from torus.service import bundle

    cmd = args.bundle_command
    if cmd == "create":
        src = args.config_dir or default_config_dir()
        print(bundle.create_bundle(src, args.out))
    elif cmd == "install":
        bundle_id = bundle.install_bundle(args.archive, args.root)
        if args.activate:
            bundle.activate(args.root, bundle_id)
        print(bundle_id)
    elif cmd == "activate":
        bundle.activate(args.root, args.bundle_id)
    elif cmd == "rollback":
        print(bundle.rollback(args.root))
    elif cmd == "path":
        print(bundle.current_path(args.root))
    else:
        print(json.dumps(bundle.read_state(args.root), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="torus.service")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the HTTP API and folder watcher.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    worker = sub.add_parser("worker", help="Run a queue worker (and model servers).")
    worker.add_argument("--runner", required=True)
    worker.add_argument("--with-model-servers", action="store_true")
    sub.add_parser("purge", help="Delete job data older than the retention period.")
    sub.add_parser("verify-audit", help="Check the audit log hash chain.")
    load = sub.add_parser("loadtest", help="Run synthetic packs through workers.")
    load.add_argument("--template", dest="template_id", required=True)
    load.add_argument("--packs", type=int, default=20)
    load.add_argument("--pack-pages", default="4,8,12", help="Comma-separated sizes.")
    load.add_argument("--workers", type=int, default=1)
    load.add_argument("--timeout-s", type=float, default=None)
    load.add_argument("--runner", default="torus.service.pipeline_runner:build_runner")
    load.add_argument("--out", type=Path, required=True)
    bundle = sub.add_parser("bundle", help="Create, install, activate, roll back.")
    bsub = bundle.add_subparsers(dest="bundle_command", required=True)
    b_create = bsub.add_parser("create", help="Pack and validate a config dir.")
    b_create.add_argument("--config-dir", type=Path, default=None)
    b_create.add_argument("--out", type=Path, required=True)
    for name in ("install", "activate", "rollback", "path", "status"):
        b = bsub.add_parser(name)
        b.add_argument("--root", type=Path, required=True, help="Bundle root dir.")
        if name == "install":
            b.add_argument("archive", type=Path)
            b.add_argument("--activate", action="store_true")
        if name == "activate":
            b.add_argument("bundle_id")
    args = parser.parse_args(argv)

    if args.command == "loadtest":
        from torus.service import loadtest

        sizes = [int(x) for x in args.pack_pages.split(",") if x.strip()]
        result = loadtest.run_load_test(
            template_id=args.template_id,
            packs=args.packs,
            pack_pages=sizes,
            workers=args.workers,
            runner_spec=args.runner,
            timeout_s=args.timeout_s or loadtest.default_timeout(args.packs),
        )
        report = loadtest.render_report(
            result,
            template_id=args.template_id,
            pack_pages=sizes,
            readers=loadtest.readers_summary(),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(report)
        return 1 if result.failed or result.timed_out else 0

    if args.command == "bundle":
        return _bundle_command(args)

    settings = load_settings()
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)

    if args.command == "purge":
        from torus.service.retention import purge_expired

        print(f"purged={purge_expired(settings, store, audit)}")
        return 0
    if args.command == "verify-audit":
        broken = audit.verify()
        print("audit chain intact" if broken is None else f"broken at seq {broken}")
        return 0 if broken is None else 1

    if args.command == "serve":
        import uvicorn

        from torus.service.api import create_app
        from torus.service.auth import load_token
        from torus.service.watcher import FolderWatcher

        stop = threading.Event()
        cfg = settings.config
        if cfg.watcher.enabled:
            settings.inbox_dir.mkdir(parents=True, exist_ok=True)
            watcher = FolderWatcher(
                settings.inbox_dir,
                store,
                template_id=cfg.default_template_id,
                stable_for_s=cfg.watcher.stable_for_s,
                audit=audit,
            )
            threading.Thread(
                target=watcher.run,
                args=(stop, cfg.watcher.poll_interval_s),
                daemon=True,
            ).start()
        from torus.review.store import CorrectionStore
        from torus.service.audit_review import review_source_from_store
        from torus.service.monitoring import model_health

        review = review_source_from_store(CorrectionStore.from_env())
        servers = cfg.model_servers
        app = create_app(
            settings,
            store,
            load_token(),
            audit=audit,
            review_source=review,
            health=lambda: model_health(servers),
        )
        try:
            uvicorn.run(app, host=args.host, port=args.port)
        finally:
            stop.set()
        return 0

    from torus.service import network_guard
    from torus.service.supervisor import ModelServerSupervisor
    from torus.service.worker import Worker

    # Workers talk only to local model servers; block everything else.
    network_guard.install()
    supervisor = ModelServerSupervisor(settings.config.model_servers)
    if args.with_model_servers:
        supervisor.start_all()
    stop = threading.Event()
    try:
        Worker(
            settings,
            store,
            load_runner(args.runner),
            worker_id=uuid.uuid4().hex[:12],
            audit=audit,
        ).run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        supervisor.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
