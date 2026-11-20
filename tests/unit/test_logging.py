"""Unit tests for torus.logging metadata-only redaction (P0.5)."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from torus.logging import (
    ALLOWED_METADATA_KEYS,
    REDACTED,
    SENSITIVE_KEYS,
    MetadataLogger,
    format_log_line,
    get_logger,
    sanitize_fields,
)

# Unique synthetic markers — must never appear in captured log output.
_LEAK_TEXT = "SYNTH_DOC_TEXT_ALPHA_NEVER_LOG"
_LEAK_VALUE = "SYNTH_FIELD_VALUE_BETA_NEVER_LOG"
_LEAK_OCR = "SYNTH_OCR_LINE_GAMMA_NEVER_LOG"


@pytest.fixture
def captured_logger() -> tuple[MetadataLogger, StringIO]:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    log = get_logger("torus.test.logging.meta")
    std = log.std_logger
    std.handlers.clear()
    std.addHandler(handler)
    std.setLevel(logging.DEBUG)
    std.propagate = False
    yield log, stream
    std.handlers.clear()


def test_get_logger_redacts_by_default() -> None:
    log = get_logger("torus.test.logging.default")
    assert isinstance(log, MetadataLogger)
    assert log.redact is True


def test_sanitize_fields_redacts_sensitive_and_unknown_keys() -> None:
    raw = {
        "file_hash": "abc123",
        "page_count": 4,
        "text": _LEAK_TEXT,
        "value": _LEAK_VALUE,
        "ocr_text": _LEAK_OCR,
        "mystery_blob": _LEAK_TEXT,
    }
    safe = sanitize_fields(raw, redact=True)
    assert safe["file_hash"] == "abc123"
    assert safe["page_count"] == 4
    assert safe["text"] == REDACTED
    assert safe["value"] == REDACTED
    assert safe["ocr_text"] == REDACTED
    assert safe["mystery_blob"] == REDACTED
    dumped = str(safe)
    assert _LEAK_TEXT not in dumped
    assert _LEAK_VALUE not in dumped
    assert _LEAK_OCR not in dumped


def test_sanitize_fields_redact_false_passthrough() -> None:
    raw = {"text": _LEAK_TEXT, "file_hash": "h"}
    assert sanitize_fields(raw, redact=False) == raw


def test_sensitive_keys_disjoint_from_allowlist() -> None:
    overlap = SENSITIVE_KEYS & ALLOWED_METADATA_KEYS
    assert overlap == frozenset()


def test_metadata_logger_keeps_allowed_fields(
    captured_logger: tuple[MetadataLogger, StringIO],
) -> None:
    log, stream = captured_logger
    log.info(
        "reader_finished",
        document_id="doc-1",
        file_hash="deadbeef",
        page_count=2,
        reader_name="stub",
        duration_ms=3.5,
        status="ok",
        confidence=0.91,
    )
    out = stream.getvalue()
    assert "reader_finished" in out
    assert "document_id='doc-1'" in out
    assert "file_hash='deadbeef'" in out
    assert "page_count=2" in out
    assert "reader_name='stub'" in out
    assert "duration_ms=3.5" in out
    assert "status='ok'" in out
    assert "confidence=0.91" in out


def test_metadata_logger_redacts_document_text_kwargs(
    captured_logger: tuple[MetadataLogger, StringIO],
) -> None:
    log, stream = captured_logger
    log.info(
        "field_read",
        file_hash="abc",
        text=_LEAK_TEXT,
        value=_LEAK_VALUE,
        ocr_text=_LEAK_OCR,
        content=_LEAK_TEXT,
    )
    out = stream.getvalue()
    assert _LEAK_TEXT not in out
    assert _LEAK_VALUE not in out
    assert _LEAK_OCR not in out
    assert REDACTED in out
    assert "file_hash='abc'" in out


def test_stdlib_positional_args_cannot_leak_document_text(
    captured_logger: tuple[MetadataLogger, StringIO],
) -> None:
    log, stream = captured_logger
    # Bypass the MetadataLogger API on purpose.
    log.std_logger.info("raw_read %s %s", _LEAK_TEXT, _LEAK_VALUE)
    out = stream.getvalue()
    assert _LEAK_TEXT not in out
    assert _LEAK_VALUE not in out
    assert out.count(REDACTED) >= 2


def test_stdlib_extra_sensitive_keys_redacted(
    captured_logger: tuple[MetadataLogger, StringIO],
) -> None:
    log, stream = captured_logger
    handler = log.std_logger.handlers[0]
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(message)s text=%(text)s value=%(value)s")
    )
    log.std_logger.info(
        "with_extra",
        extra={"text": _LEAK_TEXT, "value": _LEAK_VALUE},
    )
    out = stream.getvalue()
    assert _LEAK_TEXT not in out
    assert _LEAK_VALUE not in out
    assert f"text={REDACTED}" in out
    assert f"value={REDACTED}" in out


def test_format_log_line_stable_key_order() -> None:
    line = format_log_line("evt", {"b": 1, "a": 2})
    assert line == "evt a=2 b=1"


def test_non_scalar_allowed_value_redacted() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return _LEAK_TEXT

    safe = sanitize_fields({"file_hash": Opaque()}, redact=True)
    assert safe["file_hash"] == REDACTED
    assert _LEAK_TEXT not in str(safe)
