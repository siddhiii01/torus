"""Match printed label text to field specs (synonyms + embeddings)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import FieldSpec
from torus.layout.embed import CharNgramEmbedder, LabelEmbedder
from torus.layout.textnorm import normalize_label


class LabelMatchKind(StrEnum):
    EXACT = "exact"
    SYNONYM = "synonym"
    EMBEDDING = "embedding"


class LabelFieldMatch(BaseModel):
    """One label string scored against one field spec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_key: str = Field(min_length=1)
    label_text: str
    score: float = Field(ge=0.0, le=1.0)
    kind: LabelMatchKind
    matched_phrase: str = Field(
        description="excel_header or synonym phrase that scored highest."
    )


def field_label_phrases(spec: FieldSpec) -> list[str]:
    """Phrases that may appear as printed labels for this column."""
    phrases = [spec.excel_header, *spec.synonyms]
    # Deduplicate while preserving order (after normalisation identity).
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        key = normalize_label(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def match_label_to_fields(
    label_text: str,
    fields: Sequence[FieldSpec],
    *,
    embedder: LabelEmbedder | None = None,
    embedding_min: float = 0.72,
) -> LabelFieldMatch | None:
    """Best field for one printed label, or None if nothing clears the bar.

    Preference: exact normalised match → synonym exact → embedding cosine.
    """
    norm = normalize_label(label_text)
    if not norm:
        return None

    # Exact / synonym pass (score 1.0).
    for spec in fields:
        for phrase in field_label_phrases(spec):
            if normalize_label(phrase) == norm:
                kind = (
                    LabelMatchKind.EXACT
                    if normalize_label(spec.excel_header) == norm
                    else LabelMatchKind.SYNONYM
                )
                return LabelFieldMatch(
                    field_key=spec.key,
                    label_text=label_text,
                    score=1.0,
                    kind=kind,
                    matched_phrase=phrase,
                )

    # Embedding near-miss.
    backend = embedder if embedder is not None else CharNgramEmbedder()
    best: LabelFieldMatch | None = None
    for spec in fields:
        for phrase in field_label_phrases(spec):
            score = backend.similarity(label_text, phrase)
            if score < embedding_min:
                continue
            if best is None or score > best.score:
                best = LabelFieldMatch(
                    field_key=spec.key,
                    label_text=label_text,
                    score=score,
                    kind=LabelMatchKind.EMBEDDING,
                    matched_phrase=phrase,
                )
    return best
