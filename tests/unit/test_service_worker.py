from __future__ import annotations

import time
from pathlib import Path

import pytest

from torus.service.errors import StageTimeoutError
from torus.service.runner import JobContext
from torus.service.schemas import JobStatus
from torus.service.store import JobStore
from torus.service.worker import LeaseLostError, Worker, run_with_deadline

from .service_helpers import StubRunner, make_settings


def _setup(tmp_path: Path, runner: StubRunner, **cfg: object):  # type: ignore[no-untyped-def]
    settings = make_settings(tmp_path, **cfg)
    store = JobStore(settings.db_path)
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    job = store.enqueue(source_path=str(src), content_hash="h", template_id="tpl")
    worker = Worker(settings, store, runner, worker_id="w1", sleep=lambda _: None)
    return store, worker, job.id


def test_happy_path(tmp_path: Path) -> None:
    runner = StubRunner()
    store, worker, jid = _setup(tmp_path, runner)
    assert worker.run_one() is True
    rec = store.get_required(jid)
    assert rec.status is JobStatus.SUCCEEDED and rec.summary == {"rows": 1}
    assert runner.calls == ["read", "write"]
    assert worker.run_one() is False


def test_stage_retries_then_succeeds(tmp_path: Path) -> None:
    n = {"c": 0}

    def flaky(_: JobContext) -> None:
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("transient")

    runner = StubRunner({"read": flaky})
    store, worker, jid = _setup(tmp_path, runner, retry={"max_attempts": 3})
    worker.run_one()
    assert store.get_required(jid).status is JobStatus.SUCCEEDED
    assert runner.calls.count("read") == 3


def test_exhausted_attempts_error_code_has_no_message(tmp_path: Path) -> None:
    def bad(_: JobContext) -> None:
        raise ValueError("SYNTHETIC-SECRET")

    runner = StubRunner({"write": bad})
    store, worker, jid = _setup(tmp_path, runner, stages={"write": {"max_attempts": 2}})
    worker.run_one()
    rec = store.get_required(jid)
    assert rec.status is JobStatus.FAILED
    assert rec.error_code == "write:ValueError"
    assert "SYNTHETIC-SECRET" not in repr(rec)
    assert runner.calls == ["read", "write", "write"]


def test_stage_timeout_fails_job(tmp_path: Path) -> None:
    runner = StubRunner({"read": lambda _: time.sleep(2)})
    store, worker, jid = _setup(
        tmp_path, runner, retry={"max_attempts": 1, "timeout_s": 0.2}
    )
    started = time.monotonic()
    worker.run_one()
    assert time.monotonic() - started < 1.5
    assert store.get_required(jid).error_code == "read:StageTimeoutError"


def test_missing_result_fails(tmp_path: Path) -> None:
    class NoOutput(StubRunner):
        def run_stage(self, stage: str, ctx: JobContext) -> None:
            self.calls.append(stage)

    store, worker, jid = _setup(tmp_path, NoOutput())
    worker.run_one()
    assert store.get_required(jid).status is JobStatus.FAILED


def test_run_with_deadline_paths() -> None:
    run_with_deadline(lambda: None, timeout_s=1, tick_s=0.05, on_tick=lambda: True)
    with pytest.raises(StageTimeoutError):
        run_with_deadline(
            lambda: time.sleep(1), timeout_s=0.1, tick_s=0.05, on_tick=lambda: True
        )
    with pytest.raises(LeaseLostError):
        run_with_deadline(
            lambda: time.sleep(1), timeout_s=2, tick_s=0.05, on_tick=lambda: False
        )

    def boom() -> None:
        raise KeyError("k")

    with pytest.raises(KeyError):
        run_with_deadline(boom, timeout_s=1, tick_s=0.05, on_tick=lambda: True)


def test_dead_worker_job_is_recovered(tmp_path: Path) -> None:
    clock = [1000.0]
    settings = make_settings(tmp_path, queue={"lease_s": 10.0})
    store = JobStore(settings.db_path, clock=lambda: clock[0])
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF")
    jid = store.enqueue(source_path=str(src), content_hash="h", template_id="t").id
    store.claim_next("dead", 10.0)
    clock[0] += 20
    worker = Worker(settings, store, StubRunner(), worker_id="w2", sleep=lambda _: None)
    assert worker.run_one() is True
    rec = store.get_required(jid)
    assert rec.status is JobStatus.SUCCEEDED and rec.attempts == 2
