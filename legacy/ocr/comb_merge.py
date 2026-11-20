"""Merge comb-/character-box OCR fragments into words.

General page-reading helper (not DEB Excel rules). Adjacent single-character
(or very short) tokens on the same line with a small horizontal gap are joined.
"""
from __future__ import annotations

from ocr.engine import Word


def merge_comb_words(words: list[Word], max_gap_ratio: float = 2.5, max_frag_len: int = 1) -> list[Word]:
    if not words:
        return []
    by_line: dict[int, list[Word]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)

    out: list[Word] = []
    for line_idx in sorted(by_line):
        line = sorted(by_line[line_idx], key=lambda w: w.bbox[0])
        out.extend(_merge_line(line, max_gap_ratio, max_frag_len))
    return out


def merge_comb_page_dict(page: dict, max_gap_ratio: float = 2.5, max_frag_len: int = 1) -> dict:
    """Apply merge to an engine page dict (words as dicts)."""
    raw = page.get("words") or []
    if not raw:
        return page
    if not isinstance(raw[0], dict):
        return page
    words = [
        Word(
            str(w.get("text", "")),
            tuple(w.get("bbox") or (0, 0, 0, 0)),
            w.get("conf"),
            int(w.get("line") or 0),
        )
        for w in raw
    ]
    merged = merge_comb_words(words, max_gap_ratio=max_gap_ratio, max_frag_len=max_frag_len)
    new_page = dict(page)
    new_page["words"] = [
        {"text": w.text, "bbox": list(w.bbox), "conf": w.conf, "line": w.line} for w in merged
    ]
    new_page["comb_merge"] = {
        "before": len(words),
        "after": len(merged),
        "max_frag_len": max_frag_len,
        "max_gap_ratio": max_gap_ratio,
    }
    return new_page


def merge_comb_document(ocr: dict, max_gap_ratio: float = 2.5, max_frag_len: int = 1) -> dict:
    pages = [merge_comb_page_dict(p, max_gap_ratio, max_frag_len) for p in ocr.get("pages", [])]
    return {**ocr, "pages": pages, "variant": "comb_merge"}


def _is_frag(text: str, max_frag_len: int) -> bool:
    t = text.strip()
    if not t or len(t) > max_frag_len:
        return False
    return t.isalnum() or t in {".", "/", "-", "'"}


def _merge_line(line: list[Word], max_gap_ratio: float, max_frag_len: int) -> list[Word]:
    merged: list[Word] = []
    buf: Word | None = None

    def flush() -> None:
        nonlocal buf
        if buf is not None:
            merged.append(buf)
            buf = None

    for w in line:
        text = (w.text or "").strip()
        if not text:
            continue
        if _is_frag(text, max_frag_len):
            if buf is None:
                buf = Word(text, w.bbox, w.conf, w.line)
                continue
            gap = w.bbox[0] - buf.bbox[2]
            char_w = max(buf.bbox[2] - buf.bbox[0], w.bbox[2] - w.bbox[0], 0.004)
            if gap <= char_w * max_gap_ratio and gap >= -char_w * 0.5:
                confs = [c for c in (buf.conf, w.conf) if c is not None]
                conf = sum(confs) / len(confs) if confs else None
                buf = Word(
                    buf.text + text,
                    (
                        buf.bbox[0],
                        min(buf.bbox[1], w.bbox[1]),
                        w.bbox[2],
                        max(buf.bbox[3], w.bbox[3]),
                    ),
                    conf,
                    w.line,
                )
                continue
            flush()
            buf = Word(text, w.bbox, w.conf, w.line)
            continue
        flush()
        merged.append(w)
    flush()
    return merged
