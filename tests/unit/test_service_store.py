from __future__ import annotations

from pathlib import Path

from torus.service.schemas import JobStatus
from torus.service.store import JobStore


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _store(tmp_path: Path, clock: Clock) -> JobStore:
    return JobStore(tmp_path / "s" / "j.sqlite3", clock=clock)


def _enq(store: JobStore, n: str = "a") -> str:
    return store.enqueue(source_path=n, content_hash=n, template_id="t").id


def test_claim_is_exclusive_and_fifo(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    first = _enq(store, "a")
    clock.t += 1
    second = _enq(store, "b")
    j1 = store.claim_next("w1", 60)
    j2 = store.claim_next("w2", 60)
    assert j1 and j2 and (j1.id, j2.id) == (first, second)
    assert j1.attempts == 1 and j1.status is JobStatus.RUNNING
    assert store.claim_next("w3", 60) is None


def test_complete_only_by_owner(tmp_path: Path) -> None:
    store = _store(tmp_path, Clock())
    jid = _enq(store)
    store.claim_next("w1", 60)
    assert not store.complete(jid, "other", result_path="r", summary={})
    assert store.complete(jid, "w1", result_path="r", summary={"rows": 2})
    rec = store.get_required(jid)
    assert rec.status is JobStatus.SUCCEEDED and rec.summary == {"rows": 2}


def test_retry_respects_next_run_at(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    jid = _enq(store)
    store.claim_next("w1", 60)
    assert store.fail(jid, "w1", error_code="x", retry_at=clock.t + 10)
    assert store.claim_next("w1", 60) is None
    clock.t += 11
    again = store.claim_next("w1", 60)
    assert again and again.attempts == 2


def test_expired_lease_requeues_then_fails(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    jid = _enq(store)
    store.claim_next("w1", 10)
    clock.t += 11
    assert store.requeue_expired(max_attempts=2) == [jid]
    assert store.get_required(jid).status is JobStatus.QUEUED
    store.claim_next("w2", 10)
    clock.t += 11
    store.requeue_expired(max_attempts=2)
    rec = store.get_required(jid)
    assert rec.status is JobStatus.FAILED and rec.error_code == "lease_expired"
    assert not store.heartbeat(jid, "w1", 10, "read")


def test_cancel_only_queued(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    a = _enq(store, "a")
    clock.t += 1
    b = _enq(store, "b")
    store.claim_next("w", 60)
    assert store.cancel(b) and not store.cancel(a)
    assert store.queue_length() == 0


def test_exists_hash_ignores_failed(tmp_path: Path) -> None:
    store = _store(tmp_path, Clock())
    jid = _enq(store)
    assert store.exists_hash("a", "t")
    store.claim_next("w", 60)
    store.fail(jid, "w", error_code="x", retry_at=None)
    assert not store.exists_hash("a", "t")
