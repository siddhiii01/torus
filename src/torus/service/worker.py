"""Queue worker: claims jobs, runs pipeline stages with retries and timeouts.

Each stage has its own timeout and retry policy (``config/service.yaml``). A
stage that exhausts its attempts fails the job with a metadata-only error code
(``<stage>:<ExceptionType>``); exception text is never stored or logged because
it could carry document content.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from torus.logging import get_logger
from torus.service.audit import AuditLog
from torus.service.errors import StageTimeoutError
from torus.service.runner import JobContext, PipelineRunner
from torus.service.schemas import JobRecord
from torus.service.settings import RetryPolicy, ServiceSettings
from torus.service.store import JobStore

_log = get_logger(__name__)


class LeaseLostError(Exception):
    """The job was re-queued or failed under us; stop working on it."""


def run_with_deadline(
    fn: Callable[[], None],
    *,
    timeout_s: float,
    tick_s: float,
    on_tick: Callable[[], bool],
) -> None:
    """Run ``fn`` in a daemon thread, calling ``on_tick`` every ``tick_s``.

    Raises :class:`StageTimeoutError` at the deadline and
    :class:`LeaseLostError` when ``on_tick`` returns False. A stuck thread is
    abandoned (daemon); the worker process is the isolation boundary.
    """
    error: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StageTimeoutError("stage exceeded its timeout")
        thread.join(min(tick_s, remaining))
        if thread.is_alive() and not on_tick():
            raise LeaseLostError
    if error:
        raise error[0]


class Worker:
    def __init__(
        self,
        settings: ServiceSettings,
        store: JobStore,
        runner: PipelineRunner,
        *,
        worker_id: str,
        sleep: Callable[[float], None] = time.sleep,
        audit: AuditLog | None = None,
    ) -> None:
        self._audit = audit
        self._settings = settings
        self._store = store
        self._runner = runner
        self._id = worker_id
        self._sleep = sleep

    def run_one(self) -> bool:
        """Process one job if any is runnable. Returns True if a job was taken."""
        cfg = self._settings.config
        self._store.requeue_expired(cfg.retry.max_attempts)
        job = self._store.claim_next(self._id, cfg.queue.lease_s)
        if job is None:
            return False
        self._process(job)
        return True

    def run_forever(self, stop: threading.Event) -> None:
        poll = self._settings.config.queue.poll_interval_s
        while not stop.is_set():
            if not self.run_one():
                stop.wait(poll)

    # -- internals ----------------------------------------------------------

    def _emit(self, job_id: str, event: str, **data: object) -> None:
        if self._audit is not None:
            self._audit.record(job_id, event, actor=f"worker:{self._id}", data=data)

    def _process(self, job: JobRecord) -> None:
        cfg = self._settings.config
        work_dir = self._settings.jobs_dir / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = JobContext(
            job_id=job.id,
            source_path=Path(job.source_path),
            template_id=job.template_id,
            work_dir=work_dir,
        )
        _log.info("job_started", job_id=job.id, attempt=job.attempts)
        self._emit(
            job.id,
            "job_started",
            attempt=job.attempts,
            template_id=job.template_id,
            content_hash=job.content_hash,
        )
        started = time.monotonic()
        try:
            for stage in self._runner.stages():
                self._run_stage(stage, ctx, cfg.policy_for(stage))
            result = self._runner.result_path(ctx)
            if not result.is_file():
                raise FileNotFoundError("runner produced no result file")
        except LeaseLostError:
            _log.warning("job_lease_lost", job_id=job.id)
            self._emit(job.id, "lease_lost")
            return
        except Exception as exc:
            code = getattr(exc, "stage_code", type(exc).__name__)
            self._store.fail(job.id, self._id, error_code=str(code), retry_at=None)
            _log.error("job_failed", job_id=job.id, error_code=str(code))
            self._emit(
                job.id,
                "job_failed",
                error_code=str(code),
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            return
        self._store.complete(
            job.id, self._id, result_path=str(result), summary=dict(ctx.summary)
        )
        _log.info("job_succeeded", job_id=job.id, duration_s=time.monotonic() - started)
        self._emit(
            job.id,
            "job_succeeded",
            duration_ms=(time.monotonic() - started) * 1000.0,
            status_counts=dict(ctx.summary),
            **ctx.meta,
        )

    def _run_stage(self, stage: str, ctx: JobContext, policy: RetryPolicy) -> None:
        q = self._settings.config.queue
        delay = policy.backoff_s
        for attempt in range(1, policy.max_attempts + 1):
            if not self._store.heartbeat(ctx.job_id, self._id, q.lease_s, stage):
                raise LeaseLostError
            attempt_started = time.monotonic()
            try:
                run_with_deadline(
                    lambda: self._runner.run_stage(stage, ctx),
                    timeout_s=policy.timeout_s,
                    tick_s=q.heartbeat_s,
                    on_tick=lambda: self._store.heartbeat(
                        ctx.job_id, self._id, q.lease_s, stage
                    ),
                )
                self._emit(
                    ctx.job_id,
                    "stage_finished",
                    stage=stage,
                    attempt=attempt,
                    duration_ms=(time.monotonic() - attempt_started) * 1000.0,
                )
                return
            except LeaseLostError:
                raise
            except Exception as exc:
                _log.warning(
                    "stage_attempt_failed",
                    job_id=ctx.job_id,
                    stage=stage,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
                self._emit(
                    ctx.job_id,
                    "stage_failed",
                    stage=stage,
                    attempt=attempt,
                    error_code=f"{stage}:{type(exc).__name__}",
                    duration_ms=(time.monotonic() - attempt_started) * 1000.0,
                )
                if attempt == policy.max_attempts:
                    exc.stage_code = f"{stage}:{type(exc).__name__}"  # type: ignore[attr-defined]
                    raise
                self._sleep(delay)
                delay *= policy.backoff_factor
