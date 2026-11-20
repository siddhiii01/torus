"""P9.1 review queue, images and local server (synthetic data only)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from torus.core.models import (
    BBox,
    Evidence,
    ExtractionResult,
    FieldValue,
    Reading,
    ValidationResult,
)
from torus.core.types import FieldStatus
from torus.review.bundle import ReviewBundle, load_bundle, save_bundle
from torus.review.images import crop_png, page_png
from torus.review.queue import ReviewError, build_queue, parse_item_id
from torus.review.server import MemorySink, ReviewApp, make_server


def _ev(region: str = "r1", page: int = 0) -> Evidence:
    return Evidence(
        region_id=region,
        page_index=page,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.2),
    )


def _fv(
    key: str, status: FieldStatus, conf: float, value: str | None = "v"
) -> FieldValue:
    return FieldValue(
        column_key=key,
        value=value,
        status=status,
        confidence=conf,
        evidence=[_ev(f"reg_{key}")] if value else [],
        validator_results=[
            ValidationResult(validator_name="fmt", passed=False, message="bad shape")
        ],
        reason="why" if status != FieldStatus.CONFIRMED else None,
    )


def _result() -> ExtractionResult:
    return ExtractionResult(
        document_id="doc",
        template_id="t",
        config_hash="h",
        rows=[
            {
                "a": _fv("a", FieldStatus.CONFIRMED, 0.99),
                "b": _fv("b", FieldStatus.LIKELY, 0.7),
                "c": _fv("c", FieldStatus.NEEDS_REVIEW, 0.6),
                "d": _fv("d", FieldStatus.NOT_FOUND, 0.0, value=None),
            },
            {"c": _fv("c", FieldStatus.NEEDS_REVIEW, 0.2)},
        ],
    )


def _readings() -> list[Reading]:
    return [
        Reading(region_id="reg_c", reader_name="r_a", reader_version="1", text="x1"),
        Reading(region_id="reg_c", reader_name="r_b", reader_version="1", text="x2"),
        Reading(region_id="other", reader_name="r_a", reader_version="1", text="no"),
    ]


def test_queue_filters_and_orders() -> None:
    items = build_queue(_result(), _readings())
    assert [i.item_id for i in items] == ["1:c", "0:c", "0:b"]
    assert all(i.status != FieldStatus.CONFIRMED for i in items)


def test_queue_attaches_only_matching_readings() -> None:
    first = next(i for i in build_queue(_result(), _readings()) if i.item_id == "0:c")
    assert [r.reader_name for r in first.readings] == ["r_a", "r_b"]
    assert first.validators[0].message == "bad shape"


def test_parse_item_id_strict() -> None:
    assert parse_item_id("3:col_x") == (3, "col_x")
    for bad in ("", "x:y", "3:", "3", ":a"):
        with pytest.raises(ReviewError):
            parse_item_id(bad)


def test_images_are_png() -> None:
    img = Image.new("RGB", (400, 300), "white")
    box = BBox(x0=0.1, y0=0.1, x1=0.5, y1=0.2)
    assert crop_png(img, box).startswith(b"\x89PNG")
    assert page_png(img, box, others=[box]).startswith(b"\x89PNG")


def test_bundle_roundtrip(tmp_path: Path) -> None:
    bundle = ReviewBundle(result=_result(), readings=_readings(), source_path="s.pdf")
    path = tmp_path / "b.json"
    save_bundle(bundle, path)
    assert load_bundle(path) == bundle
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewError):
        load_bundle(path)


def _app() -> tuple[ReviewApp, MemorySink]:
    sink = MemorySink()
    items = build_queue(_result(), _readings())
    img = Image.new("RGB", (200, 100), "white")
    return ReviewApp(items, lambda _i: img, sink), sink


def test_decide_accept_correct_skip() -> None:
    app, sink = _app()
    app.decide("0:b", "accept", None)
    app.decide("0:c", "correct", "  fixed ")
    app.decide("1:c", "skip", None)
    assert sink.decisions["0:b"].value == "v"
    assert sink.decisions["0:c"].value == "fixed"
    assert sink.decisions["1:c"].value is None
    assert all(q["decision"] for q in app.queue())


def test_decide_rejects_bad_input() -> None:
    app, _ = _app()
    for args in (("0:b", "nope", None), ("0:c", "correct", ""), ("9:z", "skip", None)):
        with pytest.raises(ReviewError):
            app.decide(*args)


@pytest.fixture
def live() -> Iterator[tuple[ReviewApp, MemorySink, str]]:
    app, sink = _app()
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, sink, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read()


def test_server_serves_queue_item_and_images(
    live: tuple[ReviewApp, MemorySink, str],
) -> None:
    app, _, base = live
    status, body = _get(base + "/api/queue")
    assert status == 200 and len(json.loads(body)) == 3
    status, body = _get(base + "/api/item/0%3Ac")
    assert status == 200 and len(json.loads(body)["readings"]) == 2
    for kind in ("crop", "page"):
        status, body = _get(f"{base}/img/{kind}/0%3Ac/0.png")
        assert status == 200 and body.startswith(b"\x89PNG")
    assert _get(base + "/img/crop/0%3Ac/7.png")[0] == 404
    assert _get(base + "/api/item/..%2Fx")[0] == 404
    status, body = _get(base + "/")
    assert status == 200 and app.token.encode() in body


def test_server_rejects_foreign_host_and_missing_token(
    live: tuple[ReviewApp, MemorySink, str],
) -> None:
    _, sink, base = live
    assert _get(base + "/api/queue", {"Host": "evil.example"})[0] == 403
    req = urllib.request.Request(
        base + "/api/decision",
        data=json.dumps({"item_id": "0:b", "action": "accept"}).encode(),
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    with err.value:
        assert err.value.code == 403
    assert not sink.decisions


def test_server_decision_with_token(
    live: tuple[ReviewApp, MemorySink, str],
) -> None:
    app, sink, base = live
    req = urllib.request.Request(
        base + "/api/decision",
        data=json.dumps({"item_id": "0:b", "action": "accept"}).encode(),
        headers={"X-Torus-Token": app.token, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
    assert sink.decisions["0:b"].action == "accept"
