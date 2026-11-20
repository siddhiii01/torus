"""Service output must match the CLI path (acceptance for P11) on a synthetic page."""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageDraw
from tests.unit.service_helpers import make_settings

from torus.config import load_config
from torus.output.excel import write_blank_template, write_workbook
from torus.readers import ReaderRegistry, ReaderRunner
from torus.service.pipeline_runner import PipelineStageRunner
from torus.service.run import RunContext, RunnerReaders, process_file
from torus.service.schemas import JobStatus
from torus.service.store import JobStore
from torus.service.worker import Worker

TEMPLATE = "deb_v1"


def _cells(path: Path) -> list[list[object]]:
    ws = load_workbook(path).worksheets[0]
    return [list(r) for r in ws.iter_rows(values_only=True)]


def test_worker_matches_cli_path(tmp_path: Path) -> None:
    img = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(img).rectangle((100, 100, 700, 180), outline="black", width=3)
    src = tmp_path / "page.png"
    img.save(src)

    config = load_config(None)
    # CLI path (same calls `torus run` makes).
    template = config.template(TEMPLATE)
    ctx = RunContext(
        config=config,
        template=template,
        readers=RunnerReaders(ReaderRunner(ReaderRegistry(config.readers))),
    )
    cli_result = process_file(src, ctx)
    blank = write_blank_template(template, tmp_path / "cli_blank.xlsx")
    cli_out = tmp_path / "cli.xlsx"
    write_workbook(blank, cli_out, template, [cli_result], config.pipeline.output)

    # Service path.
    settings = make_settings(tmp_path / "svc_root")
    store = JobStore(settings.db_path)
    jid = store.enqueue(source_path=str(src), content_hash="h", template_id=TEMPLATE).id
    worker = Worker(
        settings,
        store,
        PipelineStageRunner(config),
        worker_id="w",
        sleep=lambda _: None,
    )
    assert worker.run_one()
    rec = store.get_required(jid)
    assert rec.status is JobStatus.SUCCEEDED, rec.error_code
    assert _cells(Path(rec.result_path or "")) == _cells(cli_out)
