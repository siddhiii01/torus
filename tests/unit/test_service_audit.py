from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from torus.service.api import create_app
from torus.service.audit import AuditLog
from torus.service.audit_review import ReviewAction
from torus.service.errors import ServiceError
from torus.service.schemas import JobStatus
from torus.service.store import JobStore
from torus.service.worker import Worker

from .service_helpers import PDF_BYTES, TOKEN, StubRunner, make_settings

H = {"Authorization": f"Bearer {TOKEN}"}


def test_chain_detects_tampering(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.sqlite3")
    for i in range(3):
        log.record("j", "e", actor="t", data={"attempt": i})
    assert log.verify() is None
    with closing(sqlite3.connect(tmp_path / "a.sqlite3", isolation_level=None)) as conn:
        conn.execute("UPDATE audit_events SET data='{\"attempt\": 9}' WHERE seq=2")
    assert log.verify() == 2


def test_chain_detects_deleted_row(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.sqlite3")
    for _ in range(3):
        log.record("j", "e", actor="t")
    with closing(sqlite3.connect(tmp_path / "a.sqlite3", isolation_level=None)) as conn:
        conn.execute("DELETE FROM audit_events WHERE seq=2")
    assert log.verify() == 3


@pytest.mark.parametrize(
    "data",
    [
        {"text": "x"},  # not allow-listed
        {"field_key": {"a": [1]}},  # nested value
        {"field_key": ["a"]},  # list
        {"field_key": "x" * 300},  # too long to be metadata
    ],
)
def test_rejects_content_like_data(tmp_path: Path, data: dict[str, object]) -> None:
    log = AuditLog(tmp_path / "a.sqlite3")
    with pytest.raises(ServiceError):
        log.record("j", "e", actor="t", data=data)
    assert log.events() == []


def test_worker_audits_models_config_timings_and_failures(tmp_path: Path) -> None:
    class Meta(StubRunner):
        def run_stage(self, stage, ctx) -> None:  # type: ignore[no-untyped-def]
            super().run_stage(stage, ctx)
            if stage == "write":
                ctx.meta.update(
                    document_id="doc1",
                    config_hash="abc",
                    model_versions={"r1": "1.0"},
                    timings_ms={"total": 5.0},
                )

    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)
    src = tmp_path / "a.pdf"
    src.write_bytes(PDF_BYTES)
    jid = store.enqueue(source_path=str(src), content_hash="h", template_id="t").id
    Worker(
        settings, store, Meta(), worker_id="w", sleep=lambda _: None, audit=audit
    ).run_one()
    names = [e.event for e in audit.events(jid)]
    assert names == [
        "job_started",
        "stage_finished",
        "stage_finished",
        "job_succeeded",
    ]
    done = audit.events(jid)[-1]
    assert done.data["model_versions"] == {"r1": "1.0"}
    assert done.data["config_hash"] == "abc"
    assert done.data["document_id"] == "doc1"
    assert audit.verify() is None


def test_failed_stage_is_audited_without_message(tmp_path: Path) -> None:
    def bad(_: object) -> None:
        raise ValueError("SYNTHETIC-SECRET")

    settings = make_settings(tmp_path, retry={"max_attempts": 1})
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)
    src = tmp_path / "a.pdf"
    src.write_bytes(PDF_BYTES)
    jid = store.enqueue(source_path=str(src), content_hash="h", template_id="t").id
    runner = StubRunner({"read": bad})
    Worker(
        settings, store, runner, worker_id="w", sleep=lambda _: None, audit=audit
    ).run_one()
    events = audit.events(jid)
    assert [e.event for e in events] == ["job_started", "stage_failed", "job_failed"]
    assert "SYNTHETIC-SECRET" not in repr(events)
    assert store.get_required(jid).status is JobStatus.FAILED


def test_api_audit_report_includes_review_actions_without_values(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    audit = AuditLog(settings.db_path)
    from datetime import UTC, datetime

    def source(doc_id: str) -> list[ReviewAction]:
        assert doc_id == "doc1"
        return [
            ReviewAction(
                reviewer="rev",
                action="correct",
                column_key="col",
                status_before="needs_review",  # type: ignore[arg-type]
                decided_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]

    client = TestClient(
        create_app(settings, store, TOKEN, audit=audit, review_source=source)
    )
    jid = client.put("/v1/jobs/upload?suffix=pdf", content=PDF_BYTES, headers=H).json()[
        "job_ids"
    ][0]
    audit.record(jid, "job_succeeded", actor="worker:w", data={"document_id": "doc1"})
    report = client.get(f"/v1/jobs/{jid}/audit", headers=H).json()
    assert [e["event"] for e in report["events"]] == ["job_submitted", "job_succeeded"]
    assert report["review_actions"][0]["reviewer"] == "rev"
    assert "value" not in report["review_actions"][0]
    assert client.get("/v1/jobs/nope/audit", headers=H).status_code == 404
    assert client.get(f"/v1/jobs/{jid}/audit").status_code == 401
