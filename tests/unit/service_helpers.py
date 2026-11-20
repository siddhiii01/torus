"""Shared synthetic helpers for service tests (no real documents)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torus.service.runner import JobContext
from torus.service.settings import ServiceConfig, ServiceSettings

TOKEN = "t" * 24
PDF_BYTES = b"%PDF-1.4 synthetic"


def make_settings(tmp_path: Path, **overrides: object) -> ServiceSettings:
    raw: dict[str, object] = {"default_template_id": "tpl", **overrides}
    return ServiceSettings(
        config=ServiceConfig.model_validate(raw), data_dir=tmp_path.resolve()
    )


class StubRunner:
    """Two-stage runner; behaviour injected per stage for failure tests."""

    def __init__(
        self, hooks: dict[str, Callable[[JobContext], None]] | None = None
    ) -> None:
        self.hooks = hooks or {}
        self.calls: list[str] = []

    def stages(self) -> list[str]:
        return ["read", "write"]

    def run_stage(self, stage: str, ctx: JobContext) -> None:
        self.calls.append(stage)
        hook = self.hooks.get(stage)
        if hook:
            hook(ctx)
        if stage == "write":
            (ctx.work_dir / "out.xlsx").write_bytes(b"xlsx")
            ctx.summary["rows"] = 1

    def result_path(self, ctx: JobContext) -> Path:
        return ctx.work_dir / "out.xlsx"
