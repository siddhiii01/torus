"""Batch folder watcher: queue new files that appear in the inbox.

Polling (no extra dependency). A file is queued only after its size and mtime
have been stable for ``stable_for_s``, so half-copied files are never read.
Content hashes dedupe, so a restart does not re-queue finished work.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from torus.logging import get_logger
from torus.service.audit import AuditLog
from torus.service.intake import enqueue_path, list_supported
from torus.service.store import JobStore

_log = get_logger(__name__)


class FolderWatcher:
    def __init__(
        self,
        inbox: Path,
        store: JobStore,
        *,
        template_id: str,
        stable_for_s: float,
        audit: AuditLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inbox = inbox
        self._store = store
        self._template_id = template_id
        self._stable_for_s = stable_for_s
        self._clock = clock
        self._audit = audit
        # path -> (size, mtime_ns, first time this signature was seen)
        self._seen: dict[Path, tuple[int, int, float]] = {}
        self._queued: set[Path] = set()

    def scan_once(self) -> int:
        """Queue stable new files; return how many jobs were created."""
        now = self._clock()
        created = 0
        current = set(list_supported(self._inbox)) if self._inbox.is_dir() else set()
        for gone in set(self._seen) - current:
            self._seen.pop(gone, None)
            self._queued.discard(gone)
        for path in sorted(current - self._queued):
            try:
                stat = path.stat()
            except OSError:
                continue
            sig = (stat.st_size, stat.st_mtime_ns)
            prev = self._seen.get(path)
            if prev is None or prev[:2] != sig:
                self._seen[path] = (*sig, now)
                continue
            if now - prev[2] < self._stable_for_s or stat.st_size == 0:
                continue
            _, jobs = enqueue_path(
                self._store,
                path,
                template_id=self._template_id,
                skip_known=True,
                audit=self._audit,
                actor="watcher",
            )
            self._queued.add(path)
            created += len(jobs)
        if created:
            _log.info("watcher_queued", job_count=created)
        return created

    def run(self, stop: threading.Event, poll_interval_s: float) -> None:
        while not stop.is_set():
            try:
                self.scan_once()
            except Exception:
                _log.exception("watcher_scan_failed", error_type="scan")
            stop.wait(poll_interval_s)
