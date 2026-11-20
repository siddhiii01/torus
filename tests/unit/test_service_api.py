from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from torus.service.api import create_app
from torus.service.auth import load_token
from torus.service.errors import AuthNotConfiguredError
from torus.service.schemas import JobStatus
from torus.service.settings import ServiceSettings
from torus.service.store import JobStore

from .service_helpers import PDF_BYTES, TOKEN, make_settings

H = {"Authorization": f"Bearer {TOKEN}"}
Env = tuple[ServiceSettings, JobStore, TestClient]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    return settings, store, TestClient(create_app(settings, store, TOKEN))


def test_requires_token(env: Env) -> None:
    _, _, client = env
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/jobs/x").status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.get("/v1/jobs/x", headers=bad).status_code == 401
    assert client.get("/v1/jobs/x", headers=H).status_code == 404


def test_token_fails_closed() -> None:
    with pytest.raises(AuthNotConfiguredError):
        load_token({})
    with pytest.raises(AuthNotConfiguredError):
        load_token({"TORUS_API_TOKEN": "short"})


def test_upload_and_status(env: Env) -> None:
    _, _, client = env
    r = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H)
    assert r.status_code == 201
    job_id = r.json()["job_ids"][0]
    view = client.get(f"/v1/jobs/{job_id}", headers=H).json()
    assert view["status"] == "queued" and view["result_ready"] is False
    assert "source_path" not in view and "result_path" not in view
    assert client.get(f"/v1/jobs/{job_id}/result", headers=H).status_code == 409
    batch = client.get(f"/v1/batches/{r.json()['batch_id']}", headers=H).json()
    assert batch["total"] == 1 and batch["counts"] == {"queued": 1}


def test_upload_rejections(env: Env) -> None:
    settings, _, client = env
    r = client.put("/v1/jobs/upload?suffix=exe", content=b"x", headers=H)
    assert r.status_code == 415
    r = client.put("/v1/jobs/upload?suffix=pdf", content=b"", headers=H)
    assert r.status_code == 422
    assert list(settings.uploads_dir.iterdir()) == []


def test_upload_size_limit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api={"max_upload_bytes": 10})
    client = TestClient(create_app(settings, JobStore(settings.db_path), TOKEN))
    r = client.put("/v1/jobs/upload?suffix=pdf", content=b"x" * 50, headers=H)
    assert r.status_code == 413
    assert list(settings.uploads_dir.iterdir()) == []


def test_submit_folder_and_traversal(env: Env) -> None:
    settings, _, client = env
    folder = settings.data_dir / "batch1"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(PDF_BYTES)
    (folder / "b.PNG").write_bytes(b"png")
    (folder / "notes.txt").write_bytes(b"skip")
    r = client.post("/v1/jobs", json={"path": "batch1"}, headers=H)
    assert r.status_code == 200 and len(r.json()["job_ids"]) == 2
    escape = client.post("/v1/jobs", json={"path": "../outside"}, headers=H)
    assert escape.status_code == 400
    (settings.data_dir / "empty").mkdir()
    assert client.post("/v1/jobs", json={"path": "empty"}, headers=H).status_code == 422
    body = {"path": "batch1", "template_id": "a b"}
    assert client.post("/v1/jobs", json=body, headers=H).status_code == 422


def test_result_download_and_cancel(env: Env) -> None:
    settings, store, client = env
    r = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H)
    job_id = r.json()["job_ids"][0]
    out = settings.jobs_dir / job_id / "out.xlsx"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"xlsx-bytes")
    store.claim_next("w", 60)
    store.complete(job_id, "w", result_path=str(out), summary={"rows": 1})
    got = client.get(f"/v1/jobs/{job_id}/result", headers=H)
    assert got.status_code == 200 and got.content == b"xlsx-bytes"
    assert store.get_required(job_id).status is JobStatus.SUCCEEDED
    cancel = client.post(f"/v1/jobs/{job_id}/cancel", headers=H)
    assert cancel.json() == {"cancelled": False}


def test_result_outside_jobs_dir_refused(env: Env, tmp_path: Path) -> None:
    _, store, client = env
    r = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H)
    job_id = r.json()["job_ids"][0]
    stray = tmp_path.parent / "stray.xlsx"
    stray.write_bytes(b"x")
    store.claim_next("w", 60)
    store.complete(job_id, "w", result_path=str(stray), summary={})
    assert client.get(f"/v1/jobs/{job_id}/result", headers=H).status_code == 404
