"""Shared prompt building and response normalisation for document VLMs.

Layout blocks are optional extras exposed via ``read_layout``; ``Reading`` still
carries only text and optional real scores (never invented).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from torus.core.models import BBox, ReadHint
from torus.core.types import Charset

_FENCE_RE = re.compile(
    r"^```(?:json|text|markdown)?\s*\n?(.*?)\n?```\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class LayoutBlock:
    """Optional structured span from a document VLM that emits layout JSON."""

    text: str
    bbox: BBox | None = None
    kind: str | None = None


def build_ocr_messages(
    hint: ReadHint, *, page_level: bool = False
) -> list[dict[str, str]]:
    """Constrained OCR prompt from ``ReadHint``. Temperature stays 0 at the client."""
    scope = (
        "Read all text visible on this page image."
        if page_level
        else "Read the text in this cropped field image."
    )
    constraints = _hint_constraints(hint)
    system = (
        "You are a document OCR engine running offline. "
        "Return only the transcribed text. "
        "Do not invent characters that are not visible. "
        "Do not explain. "
        "Do not wrap the answer in quotes unless they appear in the image."
    )
    user = scope
    if constraints:
        user = f"{scope}\nConstraints:\n{constraints}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _hint_constraints(hint: ReadHint) -> str:
    lines: list[str] = []
    if hint.charset is Charset.DIGITS:
        lines.append("- Characters: digits 0-9 only.")
    elif hint.charset is Charset.LETTERS:
        lines.append("- Characters: letters only.")
    elif hint.charset is Charset.ALPHANUMERIC:
        lines.append("- Characters: letters and digits only.")
    if hint.script:
        lines.append(f"- Script: {hint.script}.")
    if hint.language:
        lines.append(f"- Language: {hint.language}.")
    if hint.expected_length is not None:
        lines.append(f"- Expected length: exactly {hint.expected_length} characters.")
    else:
        if hint.min_length is not None:
            lines.append(f"- Minimum length: {hint.min_length}.")
        if hint.max_length is not None:
            lines.append(f"- Maximum length: {hint.max_length}.")
    return "\n".join(lines)


def normalise_ocr_text(raw: str) -> str:
    """Strip fences and take plain text; if JSON, join text fields in order."""
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        extracted = _text_from_json_payload(text)
        if extracted is not None:
            return extracted
    return text


def parse_layout_blocks(raw: str) -> list[LayoutBlock] | None:
    """Parse layout JSON when present. Returns ``None`` when not structured layout."""
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return None

    blocks: list[LayoutBlock] = []
    if isinstance(data, dict):
        items = data.get("blocks") or data.get("regions") or data.get("lines")
        if items is None and "text" in data:
            bbox = _bbox_from_obj(data)
            return [LayoutBlock(text=str(data.get("text") or ""), bbox=bbox)]
        if not isinstance(items, list):
            return None
        data = items

    if not isinstance(data, list):
        return None

    for item in data:
        if isinstance(item, str):
            blocks.append(LayoutBlock(text=item))
            continue
        if not isinstance(item, dict):
            continue
        value = item.get("text")
        if value is None:
            value = item.get("content")
        if value is None:
            continue
        blocks.append(
            LayoutBlock(
                text=str(value),
                bbox=_bbox_from_obj(item),
                kind=str(item["kind"]) if item.get("kind") is not None else None,
            )
        )
    return blocks if blocks else None


def _text_from_json_payload(text: str) -> str | None:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        if isinstance(data.get("text"), str):
            return str(data["text"])
        blocks = parse_layout_blocks(text)
        if blocks is not None:
            return "\n".join(b.text for b in blocks if b.text)
        return None

    if isinstance(data, list):
        blocks = parse_layout_blocks(text)
        if blocks is not None:
            return "\n".join(b.text for b in blocks if b.text)
    return None


def _bbox_from_obj(obj: dict[str, Any]) -> BBox | None:
    """Accept normalised [x0,y0,x1,y1] or {x0,y0,x1,y1}. No invented boxes."""
    box = obj.get("bbox") or obj.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            x0, y0, x1, y1 = (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            )
        except (TypeError, ValueError):
            return None
        return _valid_bbox(x0, y0, x1, y1)
    if all(k in obj for k in ("x0", "y0", "x1", "y1")):
        try:
            return _valid_bbox(
                float(obj["x0"]),
                float(obj["y0"]),
                float(obj["x1"]),
                float(obj["y1"]),
            )
        except (TypeError, ValueError):
            return None
    return None


def _valid_bbox(x0: float, y0: float, x1: float, y1: float) -> BBox | None:
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
