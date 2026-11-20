from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from torus.service.api import create_app
from torus.service.audit import AuditLog
from torus.service.monitoring import (
    collect_metrics,
    is_spot_check_sample,
    model_health,
    pending_spot_checks,
)
from torus.service.settings import ModelServerConfig, MonitoringConfig
from torus.service.store import JobStore

from .service_helpers import PDF_BYTES, TOKEN, make_settings

H = {"Authorization": f"Bearer {TOKEN}"}


def _finished(store: JobStore, name: str, summary: dict[str, int]) -> str:
    jid = store.enqueue(source_path=name, content_hash=name, template_id="t").id
    store.claim_next("w", 60)
    store.complete(jid, "w", result_path="r", summary=summary)
    return jid


def test_metrics_counts_rates_and_health(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "d.sqlite3")
    audit = AuditLog(tmp_path / "d.sqlite3")
    _finished(store, "a", {"confirmed": 6, "needs_review": 2, "not_found": 2})
    _finished(store, "b", {"confirmed": 8, "needs_review": 2})
    store.enqueue(source_path="q", content_hash="q", template_id="t")
    cfg = MonitoringConfig(window_s=3600)
    m = collect_metrics(store, audit, cfg, {"vlm": True})
    assert m.queue_length == 1
    assert m.succeeded_in_window == 2 and m.throughput_per_hour == 2.0
    assert m.fields_in_window["needs_review"] == 4
    assert m.review_rate == 4 / 20
    assert m.spot_checks.silent_error_rate is None
    assert m.model_health == {"vlm": True}


def test_model_health_uses_probe() -> None:
    servers = [
        ModelServerConfig(name="a", command=["x"], health_url="http://127.0.0.1:1/h"),
        ModelServerConfig(name="b", command=["x"], health_url="http://127.0.0.1:2/h"),
    ]
    assert model_health(servers, lambda url: url.endswith(":1/h")) == {
        "a": True,
        "b": False,
    }


def test_sampling_is_deterministic_and_respects_rate() -> None:
    ids = [f"job{i}" for i in range(2000)]
    assert [is_spot_check_sample(i, 0.0) for i in ids[:50]] == [False] * 50
    assert all(is_spot_check_sample(i, 1.0) for i in ids[:50])
    share = sum(is_spot_check_sample(i, 0.1) for i in ids) / len(ids)
    assert 0.06 < share < 0.14
    assert is_spot_check_sample("x", 0.5) == is_spot_check_sample("x", 0.5)


def test_spot_check_flow_over_api(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, monitoring={"spot_check_rate": 1.0})
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)
    client = TestClient(
        create_app(settings, store, TOKEN, audit=audit, health=lambda: {"m": False})
    )
    with_confirmed = _finished(store, "a", {"confirmed": 3})
    _finished(store, "b", {"needs_review": 3})  # nothing confirmed: not sampled
    pending = client.get("/v1/spot-checks/pending", headers=H).json()["job_ids"]
    assert pending == [with_confirmed]
    body = {"field_key": "col", "correct": False, "reviewer": "rev"}
    ok = client.post(f"/v1/jobs/{with_confirmed}/spot-check", json=body, headers=H)
    assert ok.status_code == 200
    assert client.get("/v1/spot-checks/pending", headers=H).json()["job_ids"] == []
    m = client.get("/v1/metrics", headers=H).json()
    assert m["spot_checks"] == {
        "checked": 1,
        "incorrect": 1,
        "silent_error_rate": 1.0,
    }
    assert m["model_health"] == {"m": False}
    assert client.get("/v1/metrics").status_code == 401
    queued = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H)
    qid = queued.json()["job_ids"][0]
    assert (
        client.post(f"/v1/jobs/{qid}/spot-check", json=body, headers=H).status_code
        == 404
    )
    bad = {"field_key": "bad key!", "correct": True, "reviewer": "r"}
    assert (
        client.post(
            f"/v1/jobs/{with_confirmed}/spot-check", json=bad, headers=H
        ).status_code
        == 422
    )
    assert pending_spot_checks(store, audit, settings.config.monitoring) == []
