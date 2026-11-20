"""Adapter: the real extraction pipeline behind the service ``PipelineRunner``.

Two retryable stages, mirroring ``torus run``:

* ``extract``: file -> ``ExtractionResult`` (saved in the job work dir, which
  lives under ``TORUS_DATA_DIR``).
* ``write``: ``ExtractionResult`` -> Excel, using the same writer as the CLI,
  so the service and the CLI produce the same workbook.

Use it with ``python -m torus.service worker --runner
torus.service.pipeline_runner:build_runner``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from torus.config import load_config
from torus.config.schemas import TorusConfig
from torus.consensus.calibrate import Calibrator
from torus.core.models import ExtractionResult
from torus.doctor.weights import model_dir
from torus.output.excel import write_blank_template, write_workbook
from torus.readers import ReaderRegistry, ReaderRunner
from torus.service.errors import ServiceError
from torus.service.run import (
    RunContext,
    RunnerReaders,
    default_llm_client,
    process_file,
)
from torus.service.runner import JobContext

_RESULT_FILE = "extraction.json"
_EXCEL_FILE = "result.xlsx"


class PipelineStageRunner:
    def __init__(
        self,
        config: TorusConfig,
        *,
        calibrator: Calibrator | None = None,
    ) -> None:
        self._config = config
        self._calibrator = calibrator
        self._runner = ReaderRunner(ReaderRegistry(config.readers))
        self._llm = default_llm_client(config)

    def stages(self) -> list[str]:
        return ["extract", "write"]

    def _template(self, template_id: str):  # type: ignore[no-untyped-def]
        try:
            return self._config.template(template_id)
        except KeyError as exc:
            raise ServiceError("unknown template") from exc

    def run_stage(self, stage: str, ctx: JobContext) -> None:
        template = self._template(ctx.template_id)
        result_json = ctx.work_dir / _RESULT_FILE
        if stage == "extract":
            run_ctx = RunContext(
                config=self._config,
                template=template,
                readers=RunnerReaders(self._runner),
                llm_client=self._llm,
                calibrator=self._calibrator,
            )
            result = process_file(ctx.source_path, run_ctx)
            result_json.write_text(result.model_dump_json(), encoding="utf-8")
            counts = Counter(
                fv.status.value for row in result.rows for fv in row.values()
            )
            ctx.summary.update(counts)
            ctx.summary["rows"] = len(result.rows)
            ctx.meta.update(
                document_id=result.document_id,
                config_hash=result.config_hash,
                model_versions=dict(result.model_versions),
                timings_ms=dict(result.timings_ms),
            )
        elif stage == "write":
            result = ExtractionResult.model_validate_json(
                result_json.read_text(encoding="utf-8")
            )
            out = ctx.work_dir / _EXCEL_FILE
            blank = write_blank_template(template, ctx.work_dir / "template.xlsx")
            write_workbook(blank, out, template, [result], self._config.pipeline.output)
        else:
            raise ServiceError("unknown stage")

    def result_path(self, ctx: JobContext) -> Path:
        return ctx.work_dir / _EXCEL_FILE


def build_runner() -> PipelineStageRunner:
    config = load_config(None)
    root = model_dir()
    path = root / "calibration" / "calibrator.json" if root is not None else None
    calibrator = Calibrator.load(path) if path and path.is_file() else None
    return PipelineStageRunner(config, calibrator=calibrator)
