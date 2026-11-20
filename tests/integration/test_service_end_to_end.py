"""API -> queue -> worker -> download, with a stub pipeline runner."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.unit.service_helpers import PDF_BYTES, TOKEN, StubRunner, make_settings

from torus.service.api import create_app
from torus.service.store import JobStore
from torus.service.worker import Worker


def test_submit_process_download(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    client = TestClient(create_app(settings, store, TOKEN))
    h = {"Authorization": f"Bearer {TOKEN}"}

    ids = []
    for i in range(3):
        r = client.put(
            "/v1/jobs/upload?suffix=pdf", content=PDF_BYTES + bytes([i]), headers=h
        )
        ids.append(r.json()["job_ids"][0])
    worker = Worker(settings, store, StubRunner(), worker_id="w", sleep=lambda _: None)
    while worker.run_one():
        pass
    for jid in ids:
        view = client.get(f"/v1/jobs/{jid}", headers=h).json()
        assert view["status"] == "succeeded" and view["result_ready"]
        assert client.get(f"/v1/jobs/{jid}/result", headers=h).content == b"xlsx"
