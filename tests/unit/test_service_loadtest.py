from __future__ import annotations

import random
from pathlib import Path

import fitz

from torus.service.audit import AuditLog
from torus.service.loadtest import (
    LoadResult,
    _summarise,
    make_pack,
    render_report,
)
from torus.service.store import JobStore


def test_make_pack_has_requested_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    make_pack(pdf, 3, random.Random(1))
    with fitz.open(pdf) as doc:
        assert doc.page_count == 3


def test_make_pack_is_deterministic_per_seed(tmp_path: Path) -> None:
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    make_pack(a, 2, random.Random(5))
    make_pack(b, 2, random.Random(5))
    with fitz.open(a) as da, fitz.open(b) as db:
        assert [p.get_pixmap(dpi=40).samples for p in da] == [
            p.get_pixmap(dpi=40).samples for p in db
        ]


def test_summarise_counts_and_percentiles(tmp_path: Path) -> None:
    db = tmp_path / "d.sqlite3"
    store = JobStore(db)
    audit = AuditLog(db)
    for i, seconds in enumerate([1.0, 2.0, 3.0]):
        jid = store.enqueue(source_path="x", content_hash=str(i), template_id="t").id
        store.claim_next("w", 60)
        store.complete(jid, "w", result_path="r", summary={})
        audit.record(
            jid, "job_succeeded", actor="w", data={"duration_ms": seconds * 1000}
        )
    bad = store.enqueue(source_path="x", content_hash="bad", template_id="t").id
    store.claim_next("w", 60)
    store.fail(bad, "w", error_code="extract:ValueError", retry_at=None)
    store.enqueue(source_path="x", content_hash="open", template_id="t")
    result = _summarise(
        store.updated_since(0.0),
        audit,
        packs=5,
        pages=20,
        workers=2,
        wall=10.0,
        timed_out=True,
    )
    assert (result.succeeded, result.failed, result.timed_out) == (3, 1, 1)
    assert sorted(result.service_s) == [1.0, 2.0, 3.0]
    assert result.error_codes == {"extract:ValueError": 1}


def test_report_has_hardware_and_no_content() -> None:
    result = LoadResult(
        packs=4,
        pages=32,
        workers=2,
        wall_s=36.0,
        succeeded=4,
        failed=0,
        timed_out=0,
        service_s=[10.0, 12.0, 14.0, 16.0],
        error_codes={},
    )
    text = render_report(
        result, template_id="t", pack_pages=[8], readers="none (overhead only)"
    )
    assert "Hardware profile" in text and "CPU:" in text
    assert "| Throughput | 400 packs/h |" in text
    assert "| Page throughput | 3200 pages/h |" in text
    assert "Service time p95" in text
