"""Zero-shot VLM prompt built from ``page_types.yaml`` descriptions."""

from __future__ import annotations

import json
from collections.abc import Sequence

from torus.config.schemas import PageTypesConfig, PageTypeSpec

_SYSTEM = (
    "You classify document page images into exactly one page type. "
    "Reply with a single JSON object only: "
    '{"page_type": "<id>", "confidence": <0..1>}. '
    "Use only an id from the catalogue. Prefer a specific type over the "
    "fallback when cues clearly match. Never invent new ids."
)


def build_type_catalogue(
    page_types: PageTypesConfig | Sequence[PageTypeSpec],
) -> list[dict[str, object]]:
    """Serialisable catalogue fragment for the VLM prompt (no document text)."""
    specs: Sequence[PageTypeSpec]
    if isinstance(page_types, PageTypesConfig):
        specs = page_types.page_types
    else:
        specs = page_types
    catalogue: list[dict[str, object]] = []
    for spec in specs:
        catalogue.append(
            {
                "id": spec.id,
                "description": spec.description.strip(),
                "visual_cues": list(spec.visual_cues),
                "text_cues": list(spec.text_cues),
                "examples": list(spec.examples),
            }
        )
    return catalogue


def build_zero_shot_messages(
    page_types: PageTypesConfig | Sequence[PageTypeSpec],
    *,
    fallback_page_type: str = "other",
    page_text_excerpt: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages for a local vision / text VLM.

    ``page_text_excerpt`` is optional OCR/embedded text used as a hint only.
    Callers must not log these messages (may contain document text).
    """
    catalogue = build_type_catalogue(page_types)
    user_parts = [
        "Classify this page. Catalogue (JSON):",
        json.dumps(catalogue, ensure_ascii=True),
        f"Fallback id if uncertain: {fallback_page_type}",
        "Respond with JSON only.",
    ]
    if page_text_excerpt:
        # Truncate so prompts stay bounded; never log this field.
        excerpt = page_text_excerpt[:2000]
        user_parts.insert(-1, f"Optional text hint:\n{excerpt}")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def parse_vlm_response(
    raw: str,
    allowed_ids: frozenset[str],
    *,
    fallback_page_type: str = "other",
) -> tuple[str, float]:
    """Extract ``(page_type, confidence)`` from a VLM reply string."""
    payload = _extract_json_object(raw)
    if payload is None:
        return fallback_page_type, 0.0

    page_type = str(payload.get("page_type", "")).strip()
    if page_type not in allowed_ids:
        return fallback_page_type, 0.0

    conf_raw = payload.get("confidence", 0.0)
    confidence = 0.0
    if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool):
        confidence = float(conf_raw)
    elif isinstance(conf_raw, str):
        try:
            confidence = float(conf_raw)
        except ValueError:
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return page_type, confidence


def _extract_json_object(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if not text:
        return None
    # Prefer fenced or bare JSON; fall back to first {...} span.
    if text.startswith("```"):
        lines = text.splitlines()
        inner = "\n".join(line for line in lines if not line.strip().startswith("```"))
        text = inner.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
