"""Turn reading-order words into label/value pairs and table rows.

Pairs come from the line geometry: a colon, or one wide gap with a short
label on the left. A run of lines with column-like spacing becomes a table.
Nothing here is tied to a vendor layout.
"""

from __future__ import annotations

from ocr.engine import Word

MAX_TABLE_COLS = 14
MAX_EMPTY_RATIO = 0.35
MIN_COL_GAP = 0.025


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
    used: set[int] = set()
    i = 0
    while i < len(lines):
        cols = _columns(lines[i])
        if cols is not None and i + 1 < len(lines):
            names = [c[0] for c in cols]
            centers = [c[1] for c in cols]
            rows: list[dict[str, str]] = []
            j = i + 1
            while j < len(lines) and _row_fits(lines[j], centers):
                rows.append(_zip_row(names, centers, lines[j]))
                j += 1
            if len(rows) >= 1 and _table_ok(rows):
                tables.append(rows)
                used.update(range(i, j))
                i = j
                continue
        i += 1
    for n, line in enumerate(lines):
        if n in used:
            continue
        pair = _pair(line)
        if pair is not None:
            fields.append(pair)
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
            left = _text(line[: n + 1])
            if len(left.split()) <= 6:
                return left, _text(line[n + 1 :])
    gaps = [(line[n + 1].bbox[0] - line[n].bbox[2], n) for n in range(len(line) - 1)]
    gap, at = max(gaps)
    if gap < 0.08 or at > 4:
        return None
    if sum(1 for g, _ in gaps if g >= 0.08) != 1:
        return None
    left = _text(line[: at + 1])
    if len(left.split()) > 5:
        return None
    return left, _text(line[at + 1 :])


def _columns(line: list[Word]) -> list[tuple[str, float]] | None:
    """Split a line into columns by large gaps. Multi-word headers stay together."""
    if len(line) < 4:
        return None
    groups: list[list[Word]] = [[line[0]]]
    for word in line[1:]:
        gap = word.bbox[0] - groups[-1][-1].bbox[2]
        if gap >= MIN_COL_GAP:
            groups.append([word])
        else:
            groups[-1].append(word)
    if not (4 <= len(groups) <= MAX_TABLE_COLS):
        return None
    cols = [(_text(g), (g[0].bbox[0] + g[-1].bbox[2]) / 2) for g in groups]
    # Reject a sentence: column labels should stay short.
    if sum(1 for name, _ in cols if len(name.split()) > 4) > len(cols) // 2:
        return None
    return cols


def _row_fits(row: list[Word], cols: list[float]) -> bool:
    if not row:
        return False
    hits = 0
    for word in row:
        cx = (word.bbox[0] + word.bbox[2]) / 2
        if min(abs(cx - c) for c in cols) <= 0.055:
            hits += 1
    return hits >= max(2, min(len(cols) // 2, len(row)))


def _zip_row(names: list[str], cols: list[float], row: list[Word]) -> dict[str, str]:
    cells: dict[str, list[str]] = {name: [] for name in names}
    for word in row:
        cx = (word.bbox[0] + word.bbox[2]) / 2
        i = min(range(len(cols)), key=lambda n: abs(cx - cols[n]))
        cells[names[i]].append(word.text)
    return {name: " ".join(parts) for name, parts in cells.items()}


def _table_ok(rows: list[dict[str, str]]) -> bool:
    total = sum(len(r) for r in rows)
    if total == 0:
        return False
    empty = sum(1 for r in rows for v in r.values() if not str(v).strip())
    return (empty / total) <= MAX_EMPTY_RATIO


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)
