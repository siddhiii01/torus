"""Text-cue matching for page-type scoring (config-driven, no hard-coded labels)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from torus.config.schemas import PageTypesConfig, PageTypeSpec

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace for cue matching."""
    lowered = text.casefold()
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return _WS.sub(" ", cleaned).strip()


def cue_match_score(page_text: str, cues: Sequence[str]) -> float:
    """Fraction of cue phrases found as substrings in normalised page text.

    Empty cue lists score 0. Matching is substring-based on normalised text so
    minor OCR noise still hits multi-word cues.
    """
    if not cues:
        return 0.0
    haystack = normalize_text(page_text)
    if not haystack:
        return 0.0
    hits = 0
    for cue in cues:
        needle = normalize_text(cue)
        if needle and needle in haystack:
            hits += 1
    return hits / len(cues)


def text_cue_scores(
    page_text: str,
    page_types: PageTypesConfig | Sequence[PageTypeSpec],
) -> dict[str, float]:
    """Per page-type text-cue score in ``[0, 1]``."""
    specs: Sequence[PageTypeSpec]
    if isinstance(page_types, PageTypesConfig):
        specs = page_types.page_types
    else:
        specs = page_types
    return {spec.id: cue_match_score(page_text, spec.text_cues) for spec in specs}
