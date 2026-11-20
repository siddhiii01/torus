"""Normalise printed label text for matching (no document-specific fixes)."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_label(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    Used for exact and synonym compares. Embedding uses the same form so
    "Mob. No." and "mobile no" land near each other.
    """
    folded = unicodedata.normalize("NFKD", text)
    ascii_ish = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = _PUNCT_RE.sub(" ", ascii_ish.casefold())
    return _SPACE_RE.sub(" ", cleaned).strip()
