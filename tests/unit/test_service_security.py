from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from torus.service import network_guard
from torus.service.api import create_app
from torus.service.audit import AuditLog
from torus.service.retention import purge_expired
from torus.service.schemas import JobStatus
from torus.service.store import JobStore

from .service_helpers import PDF_BYTES, TOKEN, make_settings

H = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def guard() -> Iterator[None]:
    network_guard.install()
    yield
    network_guard.uninstall()


def test_guard_blocks_public_and_hostnames(guard: None) -> None:
    sock = socket.socket()
    try:
        with pytest.raises(network_guard.OutboundNetworkBlockedError):
            sock.connect(("93.184.216.34", 80))
        with pytest.raises(network_guard.OutboundNetworkBlockedError):
            sock.connect_ex(("example.com", 80))
    finally:
        sock.close()


def test_guard_allows_loopback(guard: None) -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    try:
        client.connect(server.getsockname())
    finally:
        client.close()
        server.close()


def test_guard_uninstall_restores() -> None:
    network_guard.install()
    network_guard.uninstall()
    assert socket.socket.connect is network_guard._original_connect


def _finish_upload(client: TestClient, store: JobStore) -> str:
    jid = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H).json()[
        "job_ids"
    ][0]
    return jid


def test_retention_purges_old_jobs_only(tmp_path: Path) -> None:
    clock = [1_000_000.0]
    settings = make_settings(tmp_path, retention={"job_days": 1})
    store = JobStore(settings.db_path, clock=lambda: clock[0])
    audit = AuditLog(settings.db_path)
    client = TestClient(create_app(settings, store, TOKEN, audit=audit))
    old = _finish_upload(client, store)
    clock[0] += 3600
    new = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES + b"2", headers=H)
    new_id = new.json()["job_ids"][0]
    inbox_file = settings.inbox_dir / "keep.pdf"
    inbox_file.parent.mkdir(parents=True, exist_ok=True)
    inbox_file.write_bytes(PDF_BYTES)
    inbox_job = store.enqueue(
        source_path=str(inbox_file), content_hash="in", template_id="t"
    ).id
    for jid in (old, new_id, inbox_job):
        work = settings.jobs_dir / jid
        work.mkdir(parents=True)
        (work / "result.xlsx").write_bytes(b"x")
    # finish all three at their own times
    while (job := store.claim_next("w", 60)) is not None:
        store.complete(
            job.id,
            "w",
            result_path=str(settings.jobs_dir / job.id / "result.xlsx"),
            summary={},
        )
    clock[0] += 86400 * 0.6
    assert purge_expired(settings, store, audit, now=clock[0]) == 0
    clock[0] += 86400 * 0.6  # now the first-created jobs are > 1 day old
    purged = purge_expired(settings, store, audit, now=clock[0])
    assert purged == 3
    assert not (settings.jobs_dir / old).exists()
    assert not list(settings.uploads_dir.iterdir())
    assert inbox_file.exists()  # operator-owned input is never deleted
    assert store.get_required(old).result_path is None
    assert store.get_required(old).status is JobStatus.SUCCEEDED
    assert any(e.event == "data_purged" for e in audit.events(old))
    assert purge_expired(settings, store, audit, now=clock[0]) == 0


def test_delete_endpoint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)
    client = TestClient(create_app(settings, store, TOKEN, audit=audit))
    jid = _finish_upload(client, store)
    assert client.delete(f"/v1/jobs/{jid}", headers=H).status_code == 409
    job = store.claim_next("w", 60)
    assert job
    work = settings.jobs_dir / jid
    work.mkdir(parents=True)
    (work / "result.xlsx").write_bytes(b"x")
    store.complete(jid, "w", result_path=str(work / "result.xlsx"), summary={})
    assert client.delete(f"/v1/jobs/{jid}", headers=H).json() == {"deleted": True}
    assert not work.exists()
    assert client.get(f"/v1/jobs/{jid}/result", headers=H).status_code == 409
    assert client.delete("/v1/jobs/none", headers=H).status_code == 404
    assert client.delete(f"/v1/jobs/{jid}").status_code == 401
