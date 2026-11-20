from __future__ import annotations

from pathlib import Path

from torus.service.store import JobStore
from torus.service.watcher import FolderWatcher

from .service_helpers import PDF_BYTES


def test_waits_for_stable_file_then_queues_once(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    store = JobStore(tmp_path / "db.sqlite3")
    now = [0.0]
    w = FolderWatcher(
        inbox, store, template_id="t", stable_for_s=2.0, clock=lambda: now[0]
    )
    f = inbox / "a.pdf"
    f.write_bytes(PDF_BYTES)
    assert w.scan_once() == 0  # first sighting
    now[0] = 1.0
    assert w.scan_once() == 0  # not stable long enough
    f.write_bytes(PDF_BYTES + b"more")  # still being written
    now[0] = 3.5
    assert w.scan_once() == 0
    now[0] = 6.0
    assert w.scan_once() == 1
    now[0] = 9.0
    assert w.scan_once() == 0
    assert store.queue_length() == 1


def test_restart_does_not_requeue_known_content(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.pdf").write_bytes(PDF_BYTES)
    store = JobStore(tmp_path / "db.sqlite3")
    now = [0.0]
    for _ in range(2):
        now[0] = 0.0
        w = FolderWatcher(
            inbox, store, template_id="t", stable_for_s=0.0, clock=lambda: now[0]
        )
        w.scan_once()
        now[0] = 1.0
        w.scan_once()
    assert store.queue_length() == 1


def test_ignores_empty_and_unsupported(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "e.pdf").write_bytes(b"")
    (inbox / "x.txt").write_bytes(b"x")
    store = JobStore(tmp_path / "db.sqlite3")
    now = [0.0]
    w = FolderWatcher(
        inbox, store, template_id="t", stable_for_s=0, clock=lambda: now[0]
    )
    w.scan_once()
    now[0] = 5.0
    assert w.scan_once() == 0
