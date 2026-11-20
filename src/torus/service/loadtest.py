"""Load test with synthetic multi-page packs (P11.7).

Generates PDFs of realistic pack sizes from drawn synthetic pages (no real
documents), queues them, runs real worker *processes* against the shared queue,
and writes a markdown report: throughput, service-time percentiles, failures
and the hardware profile. The report holds counts, timings and machine facts
only, never document content.

    python -m torus.service loadtest --template deb_v1 --packs 20 \
        --pack-pages 4,8,12 --workers 2 --out reports/load/latest.md
"""

from __future__ import annotations

import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from torus.service.audit import AuditLog
from torus.service.schemas import JobRecord, JobStatus
from torus.service.settings import load_service_config
from torus.service.store import JobStore

_PAGE_PX = (1240, 1754)  # A4 at 150 dpi


@dataclass(frozen=True)
class LoadResult:
    packs: int
    pages: int
    workers: int
    wall_s: float
    succeeded: int
    failed: int
    timed_out: int
    service_s: list[float]
    error_codes: dict[str, int]


def make_pack(path: Path, pages: int, rng: random.Random) -> None:
    """Write a synthetic ``pages``-page PDF: boxes, rules and text-like marks."""
    from PIL import Image, ImageDraw

    images = []
    for _ in range(pages):
        img = Image.new("RGB", _PAGE_PX, "white")
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(6, 14)):
            x0 = rng.randint(60, 700)
            y0 = rng.randint(60, 1600)
            draw.rectangle(
                (x0, y0, x0 + rng.randint(200, 500), y0 + rng.randint(30, 80)),
                outline="black",
                width=2,
            )
            draw.text(
                (x0 + 8, y0 + 8),
                "".join(rng.choices("ABCDEF0123456789", k=12)),
                fill="black",
            )
        images.append(img)
    images[0].save(
        path, "PDF", save_all=True, append_images=images[1:], resolution=150.0
    )


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return ordered[idx]


def run_load_test(
    *,
    template_id: str,
    packs: int,
    pack_pages: list[int],
    workers: int,
    runner_spec: str,
    timeout_s: float,
    seed: int = 7,
) -> LoadResult:
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="torus-load-") as tmp:
        data = Path(tmp)
        db = data / "service" / "jobs.sqlite3"
        store = JobStore(db)
        audit = AuditLog(db)
        total_pages = 0
        for i in range(packs):
            pages = pack_pages[i % len(pack_pages)]
            total_pages += pages
            pdf = data / "uploads" / f"pack{i}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            make_pack(pdf, pages, rng)
            store.enqueue(
                source_path=str(pdf), content_hash=f"load{i}", template_id=template_id
            )
        env = {**os.environ, "TORUS_DATA_DIR": str(data)}
        started = time.monotonic()
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "torus.service",
                    "worker",
                    "--runner",
                    runner_spec,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(workers)
        ]
        timed_out = False
        try:
            while True:
                counts = store.counts_by_status()
                open_jobs = counts.get("queued", 0) + counts.get("running", 0)
                if open_jobs == 0:
                    break
                if time.monotonic() - started > timeout_s:
                    timed_out = True
                    break
                time.sleep(0.5)
            wall = time.monotonic() - started
        finally:
            for proc in procs:
                proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        records = store.updated_since(0.0)
        return _summarise(records, audit, packs, total_pages, workers, wall, timed_out)


def _summarise(
    records: list[JobRecord],
    audit: AuditLog,
    packs: int,
    pages: int,
    workers: int,
    wall: float,
    timed_out: bool,
) -> LoadResult:
    done = {e.job_id: e for e in audit.events() if e.event == "job_succeeded"}
    service = [
        float(e.data["duration_ms"]) / 1000.0
        for e in done.values()
        if "duration_ms" in e.data
    ]
    codes: dict[str, int] = {}
    for r in records:
        if r.status is JobStatus.FAILED:
            codes[r.error_code or "unknown"] = (
                codes.get(r.error_code or "unknown", 0) + 1
            )
    open_jobs = sum(
        1 for r in records if r.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    )
    return LoadResult(
        packs=packs,
        pages=pages,
        workers=workers,
        wall_s=wall,
        succeeded=sum(1 for r in records if r.status is JobStatus.SUCCEEDED),
        failed=sum(1 for r in records if r.status is JobStatus.FAILED),
        timed_out=open_jobs if timed_out else 0,
        service_s=service,
        error_codes=codes,
    )


def render_report(
    result: LoadResult, *, template_id: str, pack_pages: list[int], readers: str
) -> str:
    from torus.doctor.hardware import probe_hardware

    hw = probe_hardware()
    gpus = ", ".join(f"{g.name} ({g.vram_total_gb:.0f} GB)" for g in hw.gpus) or "none"
    hours = result.wall_s / 3600.0
    svc = result.service_s
    lines = [
        "# Load test report",
        "",
        "Synthetic packs only; no document content. Numbers are for this machine "
        "and this reader set only.",
        "",
        "## Hardware profile",
        f"- CPU: {hw.cpu_name} ({hw.cpu_count} threads)",
        f"- RAM: {hw.ram_gb if hw.ram_gb is not None else 'unknown'} GB",
        f"- GPU: {gpus}",
        f"- OS: {hw.os_name} {hw.os_release}, Python {hw.python_version}",
        f"- Readers enabled: {readers}",
        "",
        "## Run",
        f"- Template: {template_id}; pack sizes cycled: {pack_pages} pages",
        f"- Packs: {result.packs} ({result.pages} pages); workers: {result.workers}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Wall time | {result.wall_s:.1f} s |",
        "| Succeeded / failed / not finished | "
        f"{result.succeeded} / {result.failed} / {result.timed_out} |",
    ]
    if hours > 0:
        packs_h = result.succeeded / hours
        pages_h = packs_h * result.pages / max(result.packs, 1)
        lines += [
            f"| Throughput | {packs_h:.0f} packs/h |",
            f"| Page throughput | {pages_h:.0f} pages/h |",
        ]
    if svc:
        lines += [
            f"| Service time p50 | {statistics.median(svc):.2f} s |",
            f"| Service time p95 | {_percentile(svc, 95):.2f} s |",
            f"| Service time max | {max(svc):.2f} s |",
        ]
    if result.error_codes:
        lines += [
            "",
            "Failures by error code: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(result.error_codes.items())),
        ]
    return "\n".join(lines) + "\n"


def readers_summary() -> str:
    from torus.config import load_config

    config = load_config(None)
    enabled = sorted(spec.name for spec in config.readers.readers if spec.enabled)
    return ", ".join(enabled) if enabled else "none (pipeline overhead only)"


def default_timeout(packs: int) -> float:
    cfg = load_service_config(Path("config") / "service.yaml")
    return cfg.retry.timeout_s * max(packs, 1)
