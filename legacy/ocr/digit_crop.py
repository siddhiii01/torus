"""Crop-level digit sequence recovery (page OCR → tight field crop → re-read).

General OCR helper — no DEB Excel row rules. Callers pass label geometry /
expected schema and decide whether to override page text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np

# Optional heavy deps loaded lazily so import stays cheap for unit-style use.


@dataclass(frozen=True)
class DigitRead:
    field: str
    value: str
    conf: float
    reader: str
    bbox: tuple[float, float, float, float]  # normalized page crop used


_CONFUSABLE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
        "U": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "T": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "B": "8",
        "G": "6",
        "g": "6",
    }
)


def digits_only(text: str, *, confusable: bool = True) -> str:
    t = text.translate(_CONFUSABLE) if confusable else text
    return re.sub(r"\D", "", t)


def sliding(digits: str, n: int) -> list[str]:
    if n <= 0:
        return []
    if len(digits) < n:
        return [digits] if digits else []
    return [digits[i : i + n] for i in range(len(digits) - n + 1)]


def validate_dob_ddmmyyyy(digits: str) -> str | None:
    """Return ISO date if digits look like a plausible Indian form DOB."""
    if len(digits) != 8 or not digits.isdigit():
        return None
    dd, mm, yyyy = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    try:
        date(yyyy, mm, dd)
    except ValueError:
        return None
    if not (1920 <= yyyy <= 2015):
        return None
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"


def validate_pin(digits: str) -> str | None:
    if len(digits) == 6 and digits[0] != "0" and digits.isdigit():
        return digits
    return None


def validate_mobile(digits: str) -> str | None:
    if len(digits) == 10 and digits[0] in "6789" and digits.isdigit():
        return digits
    return None


def validate_std(digits: str) -> str | None:
    if 2 <= len(digits) <= 5 and digits[0] == "0" and digits.isdigit():
        return digits
    return None


def validate_landline(digits: str) -> str | None:
    if 6 <= len(digits) <= 8 and digits.isdigit():
        return digits
    return None


def crop_norm(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Crop RGB page image with normalized 0–1 bbox (x0,y0,x1,y1)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    xi0, xi1 = int(max(0.0, x0) * w), int(min(1.0, x1) * w)
    yi0, yi1 = int(max(0.0, y0) * h), int(min(1.0, y1) * h)
    if xi1 <= xi0 or yi1 <= yi0:
        return img[0:0, 0:0]
    return img[yi0:yi1, xi0:xi1]


def upscale(img: np.ndarray, scale: float = 3.0) -> np.ndarray:
    import cv2

    if scale == 1.0 or img.size == 0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def black_ink_mask(img: np.ndarray) -> np.ndarray:
    """Keep dark neutral ink; suppress bluish printed comb lines when present."""
    import cv2

    if img.size == 0:
        return img
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = gray < 90
    blueish = (b.astype(np.int16) - r.astype(np.int16)) > 15
    ink = (dark & ~blueish) | (gray < 55)
    out = np.full_like(gray, 255)
    out[ink] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def trim_to_ink_row(img: np.ndarray, *, pad: int = 3) -> np.ndarray:
    """Keep the densest horizontal ink band — drops Landmark / Owned labels."""
    import cv2

    if img.size == 0 or img.shape[0] < 8:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    ink = (gray < 100).astype(np.float32)
    # suppress long horizontal rules so the comb baseline does not dominate
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, img.shape[1] // 6), 1))
    hlines = cv2.morphologyEx((ink > 0).astype(np.uint8), cv2.MORPH_OPEN, hker)
    ink = np.clip(ink - hlines.astype(np.float32), 0, 1)
    row = ink.mean(axis=1)
    if row.max() < 1e-6:
        return img
    # prefer a band near the vertical center of the crop (comb sits mid-label)
    weights = row * (0.6 + 0.4 * np.exp(-0.5 * ((np.arange(len(row)) - len(row) / 2) / (len(row) / 4)) ** 2))
    peak = int(np.argmax(weights))
    thr = max(row.max() * 0.25, 0.02)
    ys = np.where(row >= thr)[0]
    if len(ys) == 0:
        return img
    # take contiguous run containing peak
    run = [peak]
    i = peak - 1
    while i >= 0 and row[i] >= thr * 0.7:
        run.append(i)
        i -= 1
    i = peak + 1
    while i < len(row) and row[i] >= thr * 0.7:
        run.append(i)
        i += 1
    y0, y1 = max(0, min(run) - pad), min(img.shape[0], max(run) + pad + 1)
    if y1 - y0 < 6:
        return img
    return img[y0:y1]


def prep_digit_crop(img: np.ndarray, scale: float = 3.0, *, trim: bool = True) -> np.ndarray:
    base = trim_to_ink_row(img) if trim else img
    # If trim collapsed the crop, fall back.
    if base.size == 0 or base.shape[0] < 6 or base.shape[1] < 12:
        base = img
    return upscale(black_ink_mask(base), scale=scale)


_readers: dict[str, object] = {}


def _doctr_ocr(reco: str = "parseq"):
    key = f"doctr_{reco}"
    if key not in _readers:
        from doctr.models import ocr_predictor

        kw = {"pretrained": True}
        if reco and reco != "default":
            kw["reco_arch"] = reco
        _readers[key] = ocr_predictor(**kw)
    return _readers[key]


def recognize_doctr(img: np.ndarray, reco: str = "parseq") -> tuple[str, float]:
    if img.size == 0:
        return "", 0.0
    model = _doctr_ocr(reco)
    result = model([img])
    words = []
    confs = []
    for block in result.pages[0].blocks:
        for ln in block.lines:
            for w in ln.words:
                words.append(w.value)
                confs.append(float(w.confidence))
    text = " ".join(words)
    conf = sum(confs) / len(confs) if confs else 0.0
    return text, conf


def recognize_tesseract(img: np.ndarray) -> tuple[str, float]:
    if img.size == 0:
        return "", 0.0
    try:
        import cv2
        import pytesseract
    except ImportError:
        return "", 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    best_txt, best_n = "", 0
    for psm in (8, 7, 13, 6):
        txt = pytesseract.image_to_string(
            gray, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        )
        digs = re.sub(r"\D", "", txt)
        if len(digs) > best_n:
            best_txt, best_n = digs, len(digs)
    conf = min(0.95, 0.55 + 0.05 * best_n) if best_n else 0.0
    return best_txt, conf


def _jitter_crops(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Small vertical / horizontal trims — comb OCR is sensitive to padding."""
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    out: list[tuple[str, np.ndarray]] = [("base", crop)]
    for name, y0, y1, x0, x1 in [
        ("y_in", max(0, int(0.08 * h)), max(int(0.08 * h) + 1, int(0.92 * h)), 0, w),
        ("y_tight", max(0, int(0.15 * h)), max(int(0.15 * h) + 1, int(0.85 * h)), 0, w),
        ("x_in", 0, h, max(0, int(0.03 * w)), max(int(0.03 * w) + 1, int(0.97 * w))),
    ]:
        piece = crop[y0:y1, x0:x1]
        if piece.size and piece.shape[0] >= 6 and piece.shape[1] >= 12:
            out.append((name, piece))
    return out


def majority_digit_string(candidates: list[str], n: int) -> str | None:
    """Per-position majority over exact-length digit strings."""
    cands = [c for c in candidates if len(c) == n and c.isdigit()]
    if len(cands) < 2:
        return None
    chars = []
    for i in range(n):
        counts: dict[str, int] = {}
        for c in cands:
            counts[c[i]] = counts.get(c[i], 0) + 1
        chars.append(max(counts, key=counts.get))  # type: ignore[arg-type]
    return "".join(chars)


def collect_digit_votes(
    crop: np.ndarray,
    *,
    readers: tuple[str, ...] = ("tesseract", "parseq"),
    scale: float = 3.0,
) -> list[tuple[str, str, float]]:
    """Return (reader, digit_string, conf) votes for a raw RGB crop.

    Runs each reader on ink-separated crop both with and without ink-row trim
    (label clutter vs over-trim). Default order prefers Tesseract, then parseq.
    Also emits a tesseract jitter majority vote when enough 6–8 digit strings agree.
    """
    if crop.size == 0:
        return []
    votes: list[tuple[str, str, float]] = []
    tess_exact: list[str] = []
    for jname, jcrop in _jitter_crops(crop):
        variants = [
            (f"{jname}", prep_digit_crop(jcrop, scale=scale, trim=True)),
            (f"{jname}_full", prep_digit_crop(jcrop, scale=scale, trim=False)),
            (f"{jname}_x2", prep_digit_crop(jcrop, scale=2.0, trim=False)),
            (f"{jname}_x4", prep_digit_crop(jcrop, scale=4.0, trim=False)),
        ]
        for suffix, prepped in variants:
            if prepped.size == 0:
                continue
            for reader in readers:
                tag = f"{reader}_{suffix}"
                if reader == "tesseract":
                    text, conf = recognize_tesseract(prepped)
                    digs = digits_only(text, confusable=False)
                    votes.append((tag, digs, conf))
                    if len(digs) in {6, 8, 10}:
                        tess_exact.append(digs)
                elif reader == "parseq":
                    # parseq is slower — only on base crop variants
                    if jname != "base":
                        continue
                    text, conf = recognize_doctr(prepped, reco="parseq")
                    votes.append((tag, digits_only(text), conf))
                elif reader == "crnn":
                    if jname != "base":
                        continue
                    text, conf = recognize_doctr(prepped, reco="crnn_vgg16_bn")
                    votes.append((tag, digits_only(text), conf))
                elif reader == "default":
                    if jname != "base":
                        continue
                    text, conf = recognize_doctr(prepped, reco="default")
                    votes.append((tag, digits_only(text), conf))
    for n in (6, 8, 10):
        maj = majority_digit_string(tess_exact, n)
        if maj:
            votes.insert(0, ("tesseract_majority", maj, 0.9))
    return votes


def pick_validated(
    votes: list[tuple[str, str, float]],
    lengths: list[int],
    validator: Callable[[str], str | None],
    *,
    prefer_len: int | None = None,
    require_agreement: bool = False,
    min_support: int = 1,
) -> tuple[str | None, str | None, float]:
    """Pick schema-valid candidate.

    Ranking: prefer_len match → longer digit length → earlier vote order → conf.
    When require_agreement=True, only return a value that at least two engine
    families (tesseract vs parseq) produce. min_support counts distinct vote
    tags that produced the same validated value (jitter ensemble).
    """
    by_reader: dict[str, list[tuple[str, float, int]]] = {}
    support: dict[str, set[str]] = {}
    confs: dict[str, float] = {}
    order_bonus = {name: i for i, (name, _, _) in enumerate(votes)}
    ranked: list[tuple[tuple, str, str, float]] = []

    for reader, digs, conf in votes:
        for n in lengths:
            for cand in sliding(digs, n):
                val = validator(cand)
                if not val:
                    continue
                prefer = 1 if (prefer_len is not None and len(cand) == prefer_len) else 0
                score = (prefer, len(cand), -order_bonus.get(reader, 99), conf)
                ranked.append((score, reader, val, conf))
                by_reader.setdefault(reader, []).append((val, conf, len(cand)))
                support.setdefault(val, set()).add(reader)
                confs[val] = max(confs.get(val, 0.0), conf)

    if not ranked:
        return None, None, 0.0

    if require_agreement:
        seen: dict[str, set[str]] = {}
        for reader, items in by_reader.items():
            family = reader.split("_", 1)[0]
            for val, conf, _n in items:
                seen.setdefault(val, set()).add(family)
        agreed = [v for v, rs in seen.items() if len(rs) >= 2]
        if not agreed:
            return None, None, 0.0

        def agr_score(v: str) -> tuple:
            n = 8 if re.match(r"^\d{4}-\d{2}-\d{2}$", v) else len(re.sub(r"\D", "", v))
            pref = 1 if (prefer_len is not None and n == prefer_len) else 0
            return (pref, n, confs[v])

        best_v = max(agreed, key=agr_score)
        return best_v, ",".join(sorted(seen[best_v])), confs[best_v]

    if min_support > 1:
        eligible = [v for v, tags in support.items() if len(tags) >= min_support]
        if not eligible:
            return None, None, 0.0
        ranked = [r for r in ranked if r[2] in eligible]
        if not ranked:
            return None, None, 0.0

    ranked.sort(key=lambda x: x[0], reverse=True)
    _score, reader, val, conf = ranked[0]
    return val, reader, conf


def read_digit_field(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    field: str,
    lengths: list[int],
    validator: Callable[[str], str | None],
    readers: tuple[str, ...] = ("tesseract", "parseq"),
    scale: float = 3.0,
    prefer_len: int | None = None,
    require_agreement: bool = False,
    min_support: int = 1,
    return_votes: bool = False,
) -> DigitRead | None | tuple[DigitRead | None, list[tuple[str, str, float]]]:
    """Read a located digit crop. When return_votes=True, also return raw votes for escalate."""
    crop = crop_norm(page_img, bbox)
    if crop.size == 0:
        empty: list[tuple[str, str, float]] = []
        return (None, empty) if return_votes else None
    votes = collect_digit_votes(crop, readers=readers, scale=scale)
    value, reader, conf = pick_validated(
        votes,
        lengths,
        validator,
        prefer_len=prefer_len,
        require_agreement=require_agreement,
        min_support=min_support,
    )
    hit = None
    if value is not None and reader is not None:
        hit = DigitRead(field=field, value=value, conf=conf, reader=reader, bbox=bbox)
    return (hit, votes) if return_votes else hit
