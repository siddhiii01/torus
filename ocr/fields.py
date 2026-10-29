"""Turn reading-order words into label/value pairs and table rows.

Pairs come from the line geometry: a colon, or one wide gap with a short
label on the left. A run of lines with the same column positions becomes a
table. Nothing here is tied to a vendor layout.
"""

from __future__ import annotations

from ocr.engine import Word


def reading_order(words: list[Word]) -> list[Word]:
    """Group words into visual lines, top to bottom, left to right on each line."""
    if not words:
        return []
    heights = sorted(w.bbox[3] - w.bbox[1] for w in words)
    height = heights[len(heights) // 2] or 0.01
    thresh = max(height * 0.7, 0.004)
    pending = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    lines: list[list[Word]] = []
    for word in pending:
        cy = (word.bbox[1] + word.bbox[3]) / 2
        if lines:
            last = lines[-1]
            last_cy = sum((w.bbox[1] + w.bbox[3]) / 2 for w in last) / len(last)
            if abs(cy - last_cy) <= thresh:
                last.append(word)
                continue
        lines.append([word])
    ordered: list[Word] = []
    for i, line in enumerate(lines):
        for word in sorted(line, key=lambda w: w.bbox[0]):
            ordered.append(Word(word.text, word.bbox, word.conf, i))
    return ordered


def extract(words: list[Word]) -> tuple[list[tuple[str, str]], list[list[dict[str, str]]]]:
    lines = _lines(words)
    fields: list[tuple[str, str]] = []
    tables: list[list[dict[str, str]]] = []
    i = 0
    while i < len(lines):
        header = lines[i]
        if i + 1 < len(lines) and _is_header(header) and _aligns(lines[i + 1], header):
            rows: list[dict[str, str]] = []
            j = i + 1
            while j < len(lines) and _aligns(lines[j], header):
                rows.append(_zip_row(header, lines[j]))
                j += 1
            if rows:
                tables.append(rows)
                i = j
                continue
        pair = _pair(header)
        if pair is not None:
            fields.append(pair)
        i += 1
    return fields, tables


def _lines(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for word in words:
        if word.line >= len(lines):
            lines.append([])
        lines[word.line].append(word)
    return lines


def _pair(line: list[Word]) -> tuple[str, str] | None:
    if len(line) < 2:
        return None
    for n, word in enumerate(line):
        if word.text.endswith(":") and n < len(line) - 1:
            return _text(line[: n + 1]), _text(line[n + 1 :])
    gaps = [(line[n + 1].bbox[0] - line[n].bbox[2], n) for n in range(len(line) - 1)]
    gap, at = max(gaps)
    if gap < 0.06 or at > 5 or at < 0:
        return None
    if sum(1 for g, _ in gaps if g >= 0.06) != 1:
        return None
    return _text(line[: at + 1]), _text(line[at + 1 :])


def _is_header(line: list[Word]) -> bool:
    return len(line) >= 4


def _aligns(row: list[Word], header: list[Word]) -> bool:
    if not row or abs(len(row) - len(header)) > 3:
        return False
    hits = 0
    for word in row:
        cx = (word.bbox[0] + word.bbox[2]) / 2
        nearest = min(abs(cx - (h.bbox[0] + h.bbox[2]) / 2) for h in header)
        if nearest <= 0.06:
            hits += 1
    return hits >= max(2, len(row) // 2)


def _zip_row(header: list[Word], row: list[Word]) -> dict[str, str]:
    cells: dict[str, list[str]] = {h.text: [] for h in header}
    for word in row:
        cx = (word.bbox[0] + word.bbox[2]) / 2
        nearest = min(header, key=lambda h: abs(cx - (h.bbox[0] + h.bbox[2]) / 2))
        cells[nearest.text].append(word.text)
    return {name: " ".join(parts) for name, parts in cells.items()}


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)
