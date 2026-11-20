"""Field-crop transcription prompt built from a ``ReadHint``.

The prompt is generic: it carries only the decoding constraints from the field
spec (charset, script, language, length), never document or client knowledge.
"""

from __future__ import annotations

from torus.core.models import ReadHint
from torus.core.types import Charset

_SYSTEM = (
    "You are an offline OCR engine. You are shown one cropped field image. "
    "Transcribe exactly the text written in it, character by character. "
    "Output only the transcription: no explanation, no labels, no quotes unless "
    "they are in the image. Never complete, correct or guess characters you "
    "cannot see; if the crop is empty or illegible, output nothing."
)

_CHARSET_RULES = {
    Charset.DIGITS: "digits 0-9 only",
    Charset.LETTERS: "letters only (no digits)",
    Charset.ALPHANUMERIC: "letters and digits only",
}


def constraint_lines(hint: ReadHint) -> list[str]:
    """One line per constraint present in the hint (empty for an open hint)."""
    lines: list[str] = []
    rule = _CHARSET_RULES.get(hint.charset)
    if rule is not None:
        lines.append(f"- Characters: {rule}.")
    if hint.script:
        lines.append(f"- Script: {hint.script}.")
    if hint.language:
        lines.append(f"- Language: {hint.language}.")
    if hint.expected_length is not None:
        lines.append(f"- Length: exactly {hint.expected_length} characters.")
    else:
        if hint.min_length is not None:
            lines.append(f"- Minimum length: {hint.min_length} characters.")
        if hint.max_length is not None:
            lines.append(f"- Maximum length: {hint.max_length} characters.")
    return lines


def build_messages(hint: ReadHint) -> list[dict[str, str]]:
    """System + user messages; the image is attached to the user message later."""
    user = "Transcribe the text in this image."
    lines = constraint_lines(hint)
    if lines:
        user += "\nThe text satisfies these constraints:\n" + "\n".join(lines)
        user += "\nIf what you see does not fit them, still transcribe what you see."
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
