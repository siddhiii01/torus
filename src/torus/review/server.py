"""Local review web app (P9.1): stdlib HTTP server, loopback only.

No new dependency and no network exposure: the server binds ``127.0.0.1``,
rejects foreign ``Host`` headers (DNS rebinding) and requires a per-session
token on writes (CSRF). Document content is served to the local browser and is
never logged. Persisting decisions is P9.2: the app calls a ``DecisionSink``,
and the default sink only keeps them in memory.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol
from urllib.parse import unquote, urlsplit

from torus.core.models import BBox
from torus.logging import get_logger
from torus.review.images import crop_png, page_png
from torus.review.queue import ReviewError, ReviewItem, parse_item_id
from torus.review.ui import INDEX_HTML

_log = get_logger(__name__)

DEFAULT_PORT = 8765
_MAX_BODY = 64 * 1024
_MAX_VALUE_LEN = 10_000

Action = Literal["accept", "correct", "skip"]
_ACTIONS: frozenset[str] = frozenset({"accept", "correct", "skip"})

PageProvider = Callable[[int], Any]


@dataclass(frozen=True)
class Decision:
    item_id: str
    action: Action
    value: str | None


class DecisionSink(Protocol):
    """Where decisions go. P9.2 supplies a persistent implementation."""

    def record(self, item: ReviewItem, decision: Decision) -> None: ...


class MemorySink:
    """Keeps decisions for the session only (nothing is written to disk)."""

    def __init__(self) -> None:
        self.decisions: dict[str, Decision] = {}

    def record(self, item: ReviewItem, decision: Decision) -> None:
        self.decisions[item.item_id] = decision


class ReviewApp:
    """Request-independent state and logic, so it is testable without sockets."""

    def __init__(
        self,
        items: Sequence[ReviewItem],
        page_provider: PageProvider,
        sink: DecisionSink | None = None,
    ) -> None:
        self._items = {i.item_id: i for i in items}
        self._order = [i.item_id for i in items]
        self._provider = page_provider
        self._sink: DecisionSink = sink if sink is not None else MemorySink()
        self._decided: dict[str, Decision] = {}
        self._pages: dict[int, Any] = {}
        self._lock = Lock()
        self.token = secrets.token_urlsafe(24)

    def queue(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item_id in self._order:
            item = self._items[item_id]
            decision = self._decided.get(item_id)
            out.append(
                {
                    "item_id": item.item_id,
                    "column_key": item.column_key,
                    "row": item.row,
                    "status": item.status.value,
                    "confidence": item.confidence,
                    "decision": decision.action if decision else None,
                }
            )
        return out

    def item(self, item_id: str) -> dict[str, Any]:
        item = self._get(item_id)
        data: dict[str, Any] = json.loads(item.model_dump_json())
        decision = self._decided.get(item_id)
        data["decision"] = (
            {"action": decision.action, "value": decision.value} if decision else None
        )
        return data

    def crop(self, item_id: str, evidence_index: int) -> bytes:
        item = self._get(item_id)
        ev = self._evidence(item, evidence_index)
        return crop_png(self._page(ev.page_index), ev.bbox)

    def page(self, item_id: str, evidence_index: int) -> bytes:
        item = self._get(item_id)
        ev = self._evidence(item, evidence_index)
        others: list[BBox] = [
            e.bbox
            for k, e in enumerate(item.evidence)
            if k != evidence_index and e.page_index == ev.page_index
        ]
        return page_png(self._page(ev.page_index), ev.bbox, others=others)

    def decide(self, item_id: str, action: str, value: object) -> dict[str, Any]:
        item = self._get(item_id)
        if action not in _ACTIONS:
            raise ReviewError("unknown action")
        final: str | None
        if action == "accept":
            if item.suggestion is None or item.suggestion == "":
                raise ReviewError("nothing to accept; correct the value instead")
            final = item.suggestion
        elif action == "correct":
            if not isinstance(value, str) or not value.strip():
                raise ReviewError("a corrected value is required")
            if len(value) > _MAX_VALUE_LEN:
                raise ReviewError("value too long")
            final = value.strip()
        else:
            final = None
        decision = Decision(item_id=item_id, action=action, value=final)  # type: ignore[arg-type]
        with self._lock:
            self._sink.record(item, decision)
            self._decided[item_id] = decision
        # Metadata only: never the value.
        _log.info("review_decision", column_key=item.column_key, action=action)
        return {"item_id": item_id, "decision": action}

    def _get(self, item_id: str) -> ReviewItem:
        parse_item_id(item_id)
        try:
            return self._items[item_id]
        except KeyError:
            raise ReviewError("unknown item") from None

    @staticmethod
    def _evidence(item: ReviewItem, index: int) -> Any:
        if not 0 <= index < len(item.evidence):
            raise ReviewError("no such evidence region")
        return item.evidence[index]

    def _page(self, page_index: int) -> Any:
        with self._lock:
            if page_index not in self._pages:
                self._pages[page_index] = self._provider(page_index)
            return self._pages[page_index]


def _make_handler(app: ReviewApp, allowed_hosts: frozenset[str]) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "torus-review"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Request paths carry item ids (column keys), not content, but stay
            # silent anyway; the app logs decisions as metadata.
            return

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
                return
            parts = [unquote(p) for p in urlsplit(self.path).path.split("/") if p]
            try:
                if not parts:
                    html = INDEX_HTML.replace("__TOKEN__", app.token)
                    self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
                elif parts == ["api", "queue"]:
                    self._json(HTTPStatus.OK, app.queue())
                elif len(parts) == 3 and parts[:2] == ["api", "item"]:
                    self._json(HTTPStatus.OK, app.item(parts[2]))
                elif (
                    len(parts) == 4
                    and parts[0] == "img"
                    and parts[1]
                    in {
                        "crop",
                        "page",
                    }
                ):
                    stem, _, ext = parts[3].partition(".")
                    if ext != "png" or not stem.isdigit():
                        raise ReviewError("bad image path")
                    render = app.crop if parts[1] == "crop" else app.page
                    self._send(HTTPStatus.OK, render(parts[2], int(stem)), "image/png")
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ReviewError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
                return
            if self.headers.get("X-Torus-Token") != app.token:
                self._json(HTTPStatus.FORBIDDEN, {"error": "bad token"})
                return
            if urlsplit(self.path).path != "/api/decision":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= _MAX_BODY:
                    raise ReviewError("bad body size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ReviewError("body must be an object")
                item_id = body.get("item_id")
                action = body.get("action")
                if not isinstance(item_id, str) or not isinstance(action, str):
                    raise ReviewError("item_id and action are required")
                self._json(
                    HTTPStatus.OK, app.decide(item_id, action, body.get("value"))
                )
            except (ReviewError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def make_server(app: ReviewApp, *, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Server bound to loopback. ``port=0`` picks a free port (tests)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    bound = server.server_address[1]
    allowed = frozenset({f"127.0.0.1:{bound}", f"localhost:{bound}"})
    server.RequestHandlerClass = _make_handler(app, allowed)
    return server


def serve(
    bundle_path: Path,
    *,
    port: int = DEFAULT_PORT,
    sink: DecisionSink | None = None,
) -> int:
    """Open a bundle in the review UI and block until interrupted."""
    from torus.ingest import ingest
    from torus.review.bundle import load_bundle
    from torus.review.queue import build_queue

    bundle = load_bundle(bundle_path)
    items = build_queue(bundle.result, bundle.readings)
    source = Path(bundle.source_path)
    document = ingest(source)

    def provider(page_index: int) -> Any:
        if not 0 <= page_index < len(document.document.pages):
            raise ReviewError("page out of range")
        return document.document.pages[page_index].image

    app = ReviewApp(items, provider, sink)
    server = make_server(app, port=port)
    print(
        f"Review queue: {len(items)} item(s) at "
        f"http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
