"""Metadata-only logger with redaction.

Logs carry pipeline metadata only: file hash, page count, reader name,
timings, confidence summaries, status codes, and similar keys. Document
text, OCR output, field values and other content are redacted by default.

Use::

    from torus.logging import get_logger

    log = get_logger(__name__)
    log.info(
        "reader_finished",
        document_id=doc_id,
        reader_name="paddle",
        duration_ms=12.5,
        page_count=3,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

REDACTED: Final[str] = "[REDACTED]"

# Fail-closed: only these keys may appear with their real values when
# redaction is enabled. Everything else becomes REDACTED.
ALLOWED_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "agreeing_readers",
        "classifier_name",
        "config_hash",
        "confidence",
        "confidence_mean",
        "confidence_min",
        "document_id",
        "duration_ms",
        "duration_s",
        "error_code",
        "error_type",
        "field_key",
        "file_hash",
        "model_version",
        "page_count",
        "page_index",
        "page_type",
        "reader_count",
        "reader_name",
        "reader_version",
        "region_id",
        "region_kind",
        "stage",
        "status",
        "status_code",
        "strategy",
        "template_id",
        "validator_name",
        # Reader framework (P5.1) — metadata only, never crop/OCR text.
        "batch_size",
        "cache_entries",
        "device",
        "timeout_s",
        # Service (P11) — identifiers and counters only.
        "attempt",
        "batch_id",
        "job_count",
        "job_id",
        "worker_id",
    }
)

# Always redacted, even if later added to the allowlist by mistake.
SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "body",
        "candidate",
        "content",
        "crop",
        "excerpt",
        "field_value",
        "image",
        "label_text",
        "message",
        "ocr",
        "ocr_text",
        "payload",
        "pixels",
        "raw",
        "raw_text",
        "reading",
        "snippet",
        "text",
        "value",
    }
)

_SAFE_VALUE_TYPES: Final[tuple[type, ...]] = (bool, int, float, str, type(None))


def sanitize_fields(
    fields: Mapping[str, Any],
    *,
    redact: bool = True,
) -> dict[str, Any]:
    """Return a copy of ``fields`` safe for logging.

    When ``redact`` is True (default), values for keys outside
    :data:`ALLOWED_METADATA_KEYS` or in :data:`SENSITIVE_KEYS` are replaced
    with :data:`REDACTED`. Nested mappings are sanitized recursively. Non-
    scalar values are never emitted as-is under redaction.
    """
    out: dict[str, Any] = {}
    for key, value in fields.items():
        out[key] = _sanitize_value(key, value, redact=redact)
    return out


def _sanitize_value(key: str, value: Any, *, redact: bool) -> Any:
    if not redact:
        return value
    if key in SENSITIVE_KEYS or key not in ALLOWED_METADATA_KEYS:
        return REDACTED
    if isinstance(value, Mapping):
        return sanitize_fields(value, redact=True)
    if isinstance(value, (list, tuple)):
        # Allowed metadata may be short lists of scalars (e.g. reader names).
        return [_sanitize_scalar_item(item) for item in value]
    if isinstance(value, _SAFE_VALUE_TYPES):
        return value
    return REDACTED


def _sanitize_scalar_item(item: Any) -> Any:
    if isinstance(item, _SAFE_VALUE_TYPES):
        return item
    return REDACTED


def format_log_line(event: str, fields: Mapping[str, Any]) -> str:
    """Format ``event`` plus sanitized fields as a single log line."""
    if not fields:
        return event
    parts = [event]
    for key in sorted(fields):
        parts.append(f"{key}={fields[key]!r}")
    return " ".join(parts)


class RedactingFilter(logging.Filter):
    """Stdlib filter that strips positional args and sensitive record attrs.

    Defence in depth for callers that reach the underlying ``logging.Logger``
    and pass document text via ``%s`` args or ``extra=``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = sanitize_fields(record.args, redact=True)
            else:
                # Positional args are opaque; treat all as potentially content.
                record.args = tuple(REDACTED for _ in record.args)

        for key in list(record.__dict__):
            if key in _LOGRECORD_BUILTIN_ATTRS or key.startswith("_"):
                continue
            # Leave standard LogRecord fields alone; redact extras.
            if key in SENSITIVE_KEYS or key not in ALLOWED_METADATA_KEYS:
                setattr(record, key, REDACTED)
        return True


# Attributes set by logging.LogRecord / Logger — do not touch.
_LOGRECORD_BUILTIN_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class MetadataLogger:
    """Logger that emits structured metadata and redacts document content."""

    def __init__(self, name: str, *, redact: bool = True) -> None:
        self.name = name
        self.redact = redact
        self._logger = logging.getLogger(name)
        self._ensure_filter()

    def _ensure_filter(self) -> None:
        if not self.redact:
            return
        if not any(isinstance(f, RedactingFilter) for f in self._logger.filters):
            self._logger.addFilter(RedactingFilter())

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        """Log at ERROR with exception info; fields are still sanitized."""
        safe = sanitize_fields(fields, redact=self.redact)
        self._logger.error(format_log_line(event, safe), exc_info=True)

    def _log(self, level: int, event: str, fields: Mapping[str, Any]) -> None:
        safe = sanitize_fields(fields, redact=self.redact)
        self._logger.log(level, format_log_line(event, safe))

    @property
    def std_logger(self) -> logging.Logger:
        """Underlying stdlib logger (redacting filter attached when enabled)."""
        return self._logger


def get_logger(name: str | None = None, *, redact: bool = True) -> MetadataLogger:
    """Return a metadata-only logger; redaction is on by default."""
    return MetadataLogger(name or "torus", redact=redact)


__all__ = [
    "ALLOWED_METADATA_KEYS",
    "REDACTED",
    "SENSITIVE_KEYS",
    "MetadataLogger",
    "RedactingFilter",
    "format_log_line",
    "get_logger",
    "sanitize_fields",
]
