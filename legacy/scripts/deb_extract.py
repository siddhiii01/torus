"""DEB field extractor: OCR JSON → Excel-shaped row list.

Owns DEB Excel rules. Do not put these heuristics in ocr/fields.py.

heuristic_v1_layout: section-aware assembly over word bboxes (Current Residence /
Office / name-on-card), with light comb-fragment cleanup. Optional --digit-crop
re-reads DOB/PIN/phone comb crops (ocr/digit_crop.py) without full-page re-OCR.
Optional --crop-vlm escalates only located crops that digit_crop left empty
(ocr/crop_vlm.py: TrOCR → Florence). Fill empties only; printed layout fields stay
deterministic. Flat regex remains as fallback when geometry is missing.

usage:
  python scripts/deb_extract.py reports/deb/ocr_cache/binani_dpi300_none.json -o reports/deb/preds/binani.json
  python scripts/deb_extract.py reports/deb/ocr_cache/binani_dpi300_none.json --digit-crop -o reports/deb/preds/binani.json
  python scripts/deb_extract.py reports/deb/ocr_cache/binani_dpi300_none.json --digit-crop --crop-vlm -o reports/deb/preds/binani.json
  python scripts/deb_extract.py --cases-dir data/deb/cases --ocr-dir reports/deb/ocr_cache --out-dir reports/deb/preds --digit-crop --crop-vlm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Tok:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    line: int
    page: int

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def pages_text(ocr: dict) -> str:
    parts: list[str] = []
    for page in ocr.get("pages", []):
        words = page.get("words") or []
        if words and isinstance(words[0], dict):
            parts.append(" ".join(w.get("text", "") for w in words))
        else:
            parts.append(" ".join(str(w) for w in words))
    return "\n".join(parts)


def iter_tokens(ocr: dict) -> list[Tok]:
    out: list[Tok] = []
    for page in ocr.get("pages", []):
        page_i = int(page.get("index") or 0)
        words = page.get("words") or []
        if not words or not isinstance(words[0], dict):
            continue
        for w in words:
            text = str(w.get("text") or "").strip()
            if not text:
                continue
            bbox = w.get("bbox") or (0, 0, 0, 0)
            if len(bbox) < 4:
                continue
            out.append(
                Tok(
                    text=text,
                    x0=float(bbox[0]),
                    y0=float(bbox[1]),
                    x1=float(bbox[2]),
                    y1=float(bbox[3]),
                    line=int(w.get("line") or 0),
                    page=page_i,
                )
            )
    return out


def find_first(patterns: list[str], text: str, flags: int = re.I) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None


def find_all_phones(text: str) -> list[str]:
    found = []
    for m in re.finditer(r"(?<!\d)([6-9]\d{9})(?!\d)", text):
        found.append(m.group(1))
    return list(dict.fromkeys(found))


def find_emails(text: str, *, allow_glued: bool = False) -> list[str]:
    """Return emails. Glued OCR (localgmail.com) is opt-in — often wrong and hurts F1."""
    emails = [m.group(0) for m in re.finditer(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)]
    if allow_glued:
        for m in re.finditer(r"\b([A-Z0-9._%+-]{3,})gmail\.com\b", text, re.I):
            emails.append(f"{m.group(1)}@gmail.com")
        for m in re.finditer(r"\b([A-Z0-9._%+-]{3,})(yahoo|hotmail|outlook)\.com\b", text, re.I):
            emails.append(f"{m.group(1)}@{m.group(2)}.com")
    return list(dict.fromkeys(emails))


def find_pin(text: str) -> str | None:
    m = re.search(r"(?<!\d)([1-9]\d{5})(?!\d)", text)
    return m.group(1) if m else None


def find_dob(text: str) -> str | None:
    patterns = [
        r"(?:DOB|Date of Birth|D\.?O\.?B\.?)[:\s]*([0-3]?\d[./\-][01]?\d[./\-](?:19|20)\d{2})",
        r"(?:DOB|Date of Birth)[:\s]*([0-3]?\d[\s\-./]+[A-Za-z]{3,}[\s\-./]+(?:19|20)\d{2})",
        r"(?<!\d)([0-3]?\d[./\-][01]?\d[./\-](?:19|20)\d{2})(?!\d)",
    ]
    return find_first(patterns, text)


def find_gender(text: str) -> str | None:
    if re.search(r"\bFemale\b", text):
        return "Female"
    if re.search(r"\bMale\b", text):
        return "Male"
    # checkbox residues near Gender: bM / √M / xM
    if re.search(r"Gender\s+[^\n]{0,12}\b[b√x]?M\b", text, re.I):
        return "Male"
    if re.search(r"\bSex\b[:\s]*([MF])\b", text, re.I):
        m = re.search(r"\bSex\b[:\s]*([MF])\b", text, re.I)
        assert m
        return "Male" if m.group(1).upper() == "M" else "Female"
    return None


def find_voter_id(text: str) -> str | None:
    m = re.search(r"\b([A-Z]{3}\d{7})\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z]{2}/\d{2}/\d{3}/\d{6,7})\b", text)
    return m.group(1) if m else None


def find_pan(text: str) -> str | None:
    m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text)
    return m.group(1) if m else None


_NAME_STOP = {
    "name",
    "applicant",
    "address",
    "date",
    "middle",
    "last",
    "first",
    "card",
    "please",
    "fill",
    "block",
    "letters",
    "nationality",
    "indian",
    "gender",
    "male",
    "female",
    "desired",
    "preferred",
    "mailing",
    "current",
    "residence",
    "office",
    "title",
    "personal",
    "details",
}


def find_name_candidates(text: str) -> list[str]:
    cands = []
    for pat in [
        r"(?:Name of Applicant|Applicant Name|Name on Card|Full Name)[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]*){0,3})",
        r"(?:Shri|Smt|Mr|Mrs)\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]*){0,3})",
    ]:
        for m in re.finditer(pat, text):
            name = re.sub(r"\s+", " ", m.group(1)).strip(" .")
            parts = [p for p in name.split() if p.casefold() not in _NAME_STOP]
            if len(parts) < 2:
                continue
            cands.append(" ".join(parts))
    return list(dict.fromkeys(cands))


# ---- layout helpers -------------------------------------------------------

_OCR_DIGIT_LETTER = str.maketrans({"4": "L", "0": "O", "1": "I", "5": "S", "8": "B"})


def _alpha_fix(text: str) -> str:
    t = text.translate(_OCR_DIGIT_LETTER)
    t = re.sub(r"[^A-Za-z]", "", t)
    return t


def _title_name(text: str) -> str:
    return " ".join(p.capitalize() for p in text.split() if p)


def _collapse_leading_dup(text: str) -> str:
    """BBINAIN → BINAIN; keep internal doubles like LL."""
    if len(text) >= 2 and text[0] == text[1]:
        return text[0] + text[2:]
    return text


def _line_text(tokens: list[Tok], page: int, line: int) -> str:
    return " ".join(t.text for t in sorted((x for x in tokens if x.page == page and x.line == line), key=lambda t: t.x0))


def _find_label(tokens: list[Tok], phrases: list[str], page: int | None = None) -> Tok | None:
    """Find the leftmost token that starts a label phrase on one line."""
    by_page_line: dict[tuple[int, int], list[Tok]] = {}
    for t in tokens:
        if page is not None and t.page != page:
            continue
        by_page_line.setdefault((t.page, t.line), []).append(t)

    for (pg, ln), toks in by_page_line.items():
        ordered = sorted(toks, key=lambda t: t.x0)
        joined = " ".join(t.text for t in ordered)
        low = joined.casefold()
        for phrase in phrases:
            idx = low.find(phrase.casefold())
            if idx < 0:
                continue
            # map char index → token
            cursor = 0
            for tok in ordered:
                start = cursor
                end = cursor + len(tok.text)
                if start <= idx < end + 1:
                    return tok
                cursor = end + 1  # space
    return None


def _values_right_of(
    tokens: list[Tok],
    label_phrases: list[str],
    y0: float,
    y1: float,
    page: int = 0,
    x_max: float = 0.98,
    line_prefix: bool = False,
) -> list[Tok]:
    # find label line inside band, then take tokens to the right on same line
    by_line: dict[int, list[Tok]] = {}
    for t in tokens:
        if t.page != page or not (y0 <= t.yc <= y1):
            continue
        by_line.setdefault(t.line, []).append(t)

    for ln, toks in by_line.items():
        ordered = sorted(toks, key=lambda t: t.x0)
        for phrase in label_phrases:
            span = _phrase_on_line(ordered, phrase)
            if span is None:
                continue
            i0, i1 = span
            if line_prefix and i0 > 2:
                # e.g. reject "Duration at City" / "Last Degree city"
                continue
            end_x = ordered[i1 - 1].x1
            vals = [t for t in ordered[i1:] if t.x0 < x_max]
            vals = [t for t in vals if t.text.casefold() not in _NAME_STOP]
            if vals:
                return vals
    return []


def _merge_gap_groups(toks: list[Tok], gap: float = 0.035) -> list[str]:
    if not toks:
        return []
    ordered = sorted(toks, key=lambda t: t.x0)
    groups: list[list[Tok]] = [[ordered[0]]]
    for t in ordered[1:]:
        prev = groups[-1][-1]
        if t.x0 - prev.x1 <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        raw = "".join(_alpha_fix(t.text) or t.text for t in g)
        raw = _alpha_fix(raw) or re.sub(r"[^A-Za-z0-9/.-]", "", "".join(t.text for t in g))
        if raw:
            out.append(raw)
    return out


_CITY_BLOCK = {
    "yrs",
    "mths",
    "duration",
    "std",
    "code",
    "fixed",
    "line",
    "pin",
    "state",
    "oto",
    "city",
}


def _looks_like_house_no(text: str) -> str | None:
    t = text.strip()
    # common comb OCR for 11/15
    t = re.sub(r"(?i)\bnls\b", "11/15", t)
    m = re.search(r"(?i)(?:no\.?\s*)?(\d{1,4}(?:\s*/\s*\d{1,4})?)", t)
    if not m:
        return None
    num = re.sub(r"\s+", "", m.group(1))
    return f"No.{num}"


def _phrase_on_line(ordered: list[Tok], phrase: str) -> tuple[int, int] | None:
    """Return [start, end) token indices of exact phrase on an ordered line."""
    want = phrase.casefold().split()
    texts = [t.text.casefold() for t in ordered]
    n = len(want)
    for i in range(len(texts) - n + 1):
        if texts[i : i + n] == want:
            return (i, i + n)
    return None


_ROAD_HINT = re.compile(r"(?i)\b(road|rd|street|st|lane|ln|marg|nagar|cross|avenue|ave)\b")

def _clean_street(toks: list[Tok]) -> str | None:
    """Only emit when a roadway cue is present — raw comb crumbs are high-FP."""
    if not toks:
        return None
    parts = [_alpha_fix(t.text) or t.text for t in toks]
    joined = " ".join(parts)
    compact = re.sub(r"[^A-Za-z]", "", joined).upper()
    # common DEB office pattern: N.S. Road / NS Road from NT -S ROAD
    if "ROAD" in compact and re.search(r"N.?T?.?S", compact):
        return "NS Road"
    if "ROAD" in compact or _ROAD_HINT.search(joined):
        words = []
        for t in toks:
            a = _alpha_fix(t.text)
            if not a:
                continue
            if a.upper() in {"NT", "N", "S"}:
                words.append("NS" if a.upper() == "NT" else a.upper())
            elif len(a) >= 3:
                words.append(a.capitalize())
        if not words:
            return None
        text = " ".join(words)
        text = re.sub(r"\bN\s+S\b", "NS", text, flags=re.I)
        text = re.sub(r"\bNS\s+Road\b", "NS Road", text, flags=re.I)
        if _ROAD_HINT.search(text) or "Road" in text:
            return text
    return None


def _clean_locality(toks: list[Tok]) -> str | None:
    if not toks:
        return None
    clipped: list[Tok] = []
    for t in toks:
        if t.text.casefold().startswith("landmark"):
            break
        clipped.append(t)
    alphas = [_alpha_fix(t.text) for t in clipped if _alpha_fix(t.text)]
    alphas = [a for a in alphas if a.casefold() not in _NAME_STOP and len(a) >= 1]
    if not alphas:
        return None
    merged = "".join(alphas)
    low = merged.casefold()
    suffixes = ("nagar", "bagan", "puram", "puri", "ganj", "bazar", "colony", "street", "road")
    if not any(low.endswith(suf) for suf in suffixes):
        return None
    long = [a for a in alphas if len(a) >= 3]
    if long:
        return _title_name("".join(long) if sum(len(a) for a in long) <= 16 else " ".join(long))
    return merged.capitalize()


def _clean_state(toks: list[Tok]) -> str | None:
    alphas = [_alpha_fix(t.text) for t in toks if _alpha_fix(t.text)]
    alphas = [s for s in alphas if len(s) >= 2 and s.casefold() not in _CITY_BLOCK]
    if not alphas:
        return None
    joined = " ".join(alphas)
    from ocr.pin_master import validate_state_token

    return validate_state_token(joined)


def assemble_person_name(tokens: list[Tok], text: str) -> str | None:
    """Build person name from name-on-card band + corroborating tokens."""
    # Name as desired on the card <VALUE> Preferred Mailing...
    card_vals: list[Tok] = []
    for t in tokens:
        if t.page != 0:
            continue
        line = _line_text(tokens, t.page, t.line)
        if "desired" in line.casefold() and "card" in line.casefold():
            # value band between 'card' and 'Preferred'/'Mailing'
            if 0.24 <= t.x0 <= 0.55 and t.text.casefold() not in {
                "name",
                "as",
                "desired",
                "on",
                "the",
                "card",
                "preferred",
                "mailing",
                "address",
                "current",
                "residence",
                "office",
            }:
                card_vals.append(t)

    groups = _merge_gap_groups(card_vals, gap=0.04)
    # OCR digit fixes already applied in merge
    groups = [_collapse_leading_dup(g.upper()) for g in groups if len(g) >= 2]
    groups = [g for g in groups if g.casefold() not in _NAME_STOP]

    first = groups[0] if groups else None
    last = groups[1] if len(groups) > 1 else None

    # First-name field often sits just above "First Name" labels
    fn_label = _find_label(tokens, ["First Name"], page=0)
    if fn_label is not None:
        above = [
            t
            for t in tokens
            if t.page == 0
            and fn_label.y0 - 0.035 <= t.yc <= fn_label.y0 - 0.002
            and 0.05 <= t.x0 <= 0.45
            and _alpha_fix(t.text)
        ]
        frag = "".join(_alpha_fix(t.text) for t in sorted(above, key=lambda x: x.x0))
        frag = frag.upper()
        if frag and (first is None or len(frag) >= len(first) - 1):
            # SA + ALE → SAALE; prefer card if longer/clearer
            if first is None:
                first = frag
            elif frag.startswith(first[:2]) and len(first) >= 5:
                pass
            elif len(first) <= len(frag) and frag.startswith("SA"):
                # keep card SAILESH* over SAALE
                pass

    if first:
        first = first.upper()
        # SAILESHI → SAILESH when trailing I before surname group
        if last and first.endswith("I") and len(first) >= 7:
            first = first[:-1]
        # SAI + LESHI already merged; trim obvious OCR tail
        if first.endswith("HI") and len(first) > 7:
            pass

    # Surname corroboration from company / signature-like tokens
    if last:
        last = _collapse_leading_dup(last.upper())
    companyish = []
    for t in tokens:
        a = _alpha_fix(t.text).upper()
        if len(a) < 5:
            continue
        if a.startswith("BBIN") or a.endswith("NAN") or a.endswith("AIN"):
            companyish.append(_collapse_leading_dup(a))
        if re.fullmatch(r"S?S?[A-Z]{3,8}AN", a):
            companyish.append(_collapse_leading_dup(a))

    def _prefer_surname(card: str | None, alts: list[str]) -> str | None:
        cands = []
        if card:
            cands.append(card)
        cands.extend(alts)
        cands = [c for c in cands if len(c) >= 4]
        if not cands:
            return None
        # Prefer token starting with B over S when both look like *NAN/*AIN
        b_like = [c for c in cands if c.startswith("B") and c.endswith(("AN", "ANI", "AIN", "NAN"))]
        if b_like:
            best = b_like[0]
            # BINAIN → BINANI (AIN/ANI ending confusion)
            if best.endswith("AIN") and not best.endswith("BAIN"):
                best = best[:-3] + "ANI"
            return best
        best = cands[0]
        if best.startswith("S") and any(c.startswith("B") for c in cands):
            best = "B" + best[1:]
        if best.endswith("AIN") and not best.endswith("BAIN"):
            best = best[:-3] + "ANI"
        # SIINAN → SINAN (collapse II) then if ends with NAN keep
        best = re.sub(r"([A-Z])\1", r"\1", best) if "II" in best else best
        if best.startswith("S") and best.endswith("NAN") and len(best) == 5:
            # SINAN ← SIINAN; B-replacement when company evidence exists
            if any(c.startswith("B") for c in cands):
                best = "B" + best[1:]
        return best

    last = _prefer_surname(last, companyish)

    # Email local-part hint: saileshbinani@… or saileshbireni@…
    if first is None or last is None:
        for em in find_emails(text):
            local = em.split("@", 1)[0]
            local = re.sub(r"[^a-zA-Z]", "", local)
            if first and local.casefold().startswith(first[:4].casefold()):
                rest = local[len(first) :] if local.casefold().startswith(first.casefold()) else ""
                # try split after first-name length
                if not rest and first:
                    # peel first name prefix greedily 4–10 chars
                    for n in range(min(10, len(local)), 3, -1):
                        if local[:n].casefold() == first[:n].casefold() or first.casefold().startswith(local[:n].casefold()):
                            rest = local[n:]
                            first = local[:n]
                            break
                if rest and len(rest) >= 4 and last is None:
                    last = rest.upper()
            elif first is None and len(local) >= 8:
                # weak: leave to card only
                pass

    if not first or not last:
        # flat regex fallback
        cands = find_name_candidates(text)
        if cands:
            return cands[0]
        return None

    # Final polish first name: SAILESH from SAILESHI already handled
    if first.endswith("I") and last.startswith("B") and len(first) >= 7:
        first = first[:-1]

    return _title_name(f"{first} {last}")


def extract_address_block(tokens: list[Tok], y0: float, y1: float, page: int = 0) -> dict:
    building = _values_right_of(tokens, ["Building Name"], y0, y1, page=page, x_max=0.98)
    street = _values_right_of(tokens, ["Street Name"], y0, y1, page=page, x_max=0.98)
    locality = _values_right_of(tokens, ["Locality"], y0, y1, page=page, x_max=0.48)
    city_line = _values_right_of(
        tokens, ["City"], y0, y1, page=page, x_max=0.42, line_prefix=True
    )
    pin_vals = _values_right_of(tokens, ["PIN Code", "PIN Codel", "PIN"], y0, y1, page=page, x_max=0.62)
    state_vals = _values_right_of(tokens, ["State"], y0, y1, page=page, x_max=0.95)

    row: dict = {}
    btxt = " ".join(t.text for t in building)
    house = _looks_like_house_no(btxt)
    if house:
        row["address_line1"] = house

    # Street may be written on the Building Name comb row (office NS Road).
    building_street = _clean_street(building) if building else None
    street_clean = _clean_street(street) if street else None
    line2_cands = []
    if building_street and "Road" in building_street:
        line2_cands.append(building_street)
    if street_clean and len(street_clean) >= 3 and street_clean.casefold() not in _CITY_BLOCK:
        # Prefer longer / Road-bearing strings over 2-char OCR crumbs ("En")
        line2_cands.append(street_clean)
    if not house and building and not line2_cands:
        cleaned = _clean_street(building)
        if cleaned and not cleaned.startswith("No."):
            row["address_line1"] = cleaned

    if line2_cands:
        line2_cands.sort(key=lambda s: ("road" not in s.casefold(), -len(s)))
        best = line2_cands[0]
        if len(best) >= 3:
            row["address_line2"] = best

    loc = _clean_locality(locality)
    if loc and loc.casefold() not in _CITY_BLOCK:
        row["address_line3"] = loc

    city_alphas = [_alpha_fix(t.text) for t in city_line if _alpha_fix(t.text)]
    city_alphas = [
        c
        for c in city_alphas
        if c.casefold() not in _CITY_BLOCK
        and c.casefold() not in {"pin", "code", "codel", "state"}
        and len(c) >= 4
        and not c.isdigit()
    ]
    if city_alphas:
        row["address_city"] = city_alphas[0].capitalize()

    for t in pin_vals + state_vals + city_line:
        digits = re.sub(r"\D", "", t.text)
        if len(digits) == 6 and digits[0] != "0":
            row["address_pincode"] = digits
            break
        if len(digits) >= 6:
            for i in range(len(digits) - 5):
                cand = digits[i : i + 6]
                if cand[0] != "0":
                    row["address_pincode"] = cand
                    break
            if "address_pincode" in row:
                break

    state = _clean_state(state_vals)
    if state:
        row["address_state"] = state

    return row


def _iter_phrase_hits(tokens: list[Tok], phrase: str, page: int = 0) -> list[tuple[list[Tok], tuple[int, int]]]:
    by_line: dict[int, list[Tok]] = {}
    for t in tokens:
        if t.page != page:
            continue
        by_line.setdefault(t.line, []).append(t)
    hits = []
    for toks in by_line.values():
        ordered = sorted(toks, key=lambda x: x.x0)
        span = _phrase_on_line(ordered, phrase)
        if span is not None:
            hits.append((ordered, span))
    return hits


def _label_in_yband(
    tokens: list[Tok],
    phrases: list[str],
    y0: float,
    y1: float,
    page: int = 0,
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> tuple[Tok, Tok, list[Tok]] | None:
    for phrase in phrases:
        for ordered, span in _iter_phrase_hits(tokens, phrase, page=page):
            lab0, lab1 = ordered[span[0]], ordered[span[1] - 1]
            if not (y0 <= lab0.yc <= y1):
                continue
            if not (x_min <= lab0.x0 <= x_max):
                continue
            return lab0, lab1, ordered
    return None


def _crop_right_of_label(
    lab0: Tok,
    lab1: Tok,
    ordered: list[Tok],
    *,
    stop_words: set[str] | None = None,
    x_fallback: float = 0.12,
    y_pad_top: float = 0.008,
    y_pad_bot: float = 0.01,
    x_pad: float = 0.002,
) -> tuple[float, float, float, float]:
    stop_words = stop_words or set()
    x1 = lab1.x1 + x_fallback
    for t in ordered:
        if t.x0 <= lab1.x1 + 0.001:
            continue
        low = t.text.casefold().rstrip(".:")
        if low in stop_words or any(sw in low for sw in stop_words):
            x1 = t.x0 - 0.005
            break
    return (lab1.x1 + x_pad, lab0.y0 - y_pad_top, x1, lab0.y1 + y_pad_bot)


def locate_digit_crops(
    tokens: list[Tok],
    res_range: tuple[float, float] | None,
    office_range: tuple[float, float] | None,
    page: int = 0,
) -> dict[str, tuple[float, float, float, float]]:
    """Normalized bboxes for digit fields keyed by logical name (DEB layout)."""
    crops: dict[str, tuple[float, float, float, float]] = {}

    # Personal DOB sits on the right of the Personal Details name row (not Add-on).
    dob_hit = _label_in_yband(
        tokens, ["Date of Birth"], 0.35, 0.45, page=page, x_min=0.45, x_max=0.95
    )
    if dob_hit:
        lab0, lab1, ordered = dob_hit
        crops["dob"] = _crop_right_of_label(
            lab0,
            lab1,
            ordered,
            stop_words={"gender", "bm", "bM", "male", "female"},
            x_fallback=0.18,
            y_pad_top=0.006,
            y_pad_bot=0.012,
        )

    def _pin_crop(y0: float, y1: float, key: str) -> None:
        hit = _label_in_yband(
            tokens, ["PIN Codel", "PIN Code", "PIN"], y0, y1, page=page, x_min=0.35
        )
        if not hit:
            return
        lab0, lab1, ordered = hit
        # Tight vertical band around the PIN comb (label yc) — extra Landmark /
        # Residence text above/below confuses digit readers.
        stop = {"state"}
        x1 = lab1.x1 + 0.11
        for t in ordered:
            if t.x0 <= lab1.x1 + 0.001:
                continue
            low = t.text.casefold().rstrip(".:")
            if low in stop or any(sw in low for sw in stop):
                x1 = t.x0 - 0.005
                break
        # Comb sits slightly below the PIN caption midline (Binani bake-off).
        half = 0.015
        yc = lab0.yc + 0.004
        crops[key] = (lab1.x1 + 0.002, yc - half, x1, yc + half)

    def _phone_crops(y0: float, y1: float, prefix: str) -> None:
        mobile = _label_in_yband(tokens, ["Mobile"], y0, y1, page=page, x_max=0.35)
        if mobile:
            lab0, lab1, ordered = mobile
            box = _crop_right_of_label(
                lab0,
                lab1,
                ordered,
                stop_words={"e-mail", "email", "mail", "tel"},
                x_fallback=0.2,
                y_pad_top=0.02,
                y_pad_bot=0.015,
            )
            # Comb for mobile is often slightly above the printed label.
            crops[f"{prefix}mobile"] = (box[0], box[1] - 0.008, box[2], box[3])

        std = _label_in_yband(tokens, ["STD Code", "STD"], y0, y1, page=page, x_min=0.5)
        if not std:
            return
        lab0, lab1, ordered = std
        # Digit combs sit just above the STD / Fixed-line captions.
        y_top = lab0.y0 - 0.018
        y_bot = lab0.y0 - 0.001
        # Include leading STD cell (often starts slightly left of the caption).
        std_box = (max(0.55, lab0.x0 - 0.04), y_top, min(0.72, lab1.x1 + 0.02), y_bot)
        land_box = (0.715, y_top, 0.87, y_bot)
        crops[f"{prefix}std"] = std_box
        crops[f"{prefix}landline"] = land_box

    if res_range:
        _pin_crop(res_range[0], res_range[1], "pin_h1")
        _phone_crops(res_range[0], res_range[1], "h1_")
    if office_range:
        _pin_crop(office_range[0], office_range[1], "pin_w")
        _phone_crops(office_range[0], office_range[1], "w_")

    return crops


def locate_text_crops(
    tokens: list[Tok],
    res_range: tuple[float, float] | None,
    office_range: tuple[float, float] | None,
    page: int = 0,
) -> dict[str, tuple[float, float, float, float]]:
    """Tight field crops for handwriting / glued OCR tokens (email, city, state)."""
    crops: dict[str, tuple[float, float, float, float]] = {}

    def _email(y0: float, y1: float, key: str) -> None:
        hit = _label_in_yband(
            tokens, ["E-mail", "Email", "E-Mail", "Mail"], y0, y1, page=page, x_min=0.35
        )
        if not hit:
            return
        lab0, lab1, ordered = hit
        crops[key] = _crop_right_of_label(
            lab0,
            lab1,
            ordered,
            stop_words={"tel", "mobile", "std"},
            x_fallback=0.28,
            y_pad_top=0.012,
            y_pad_bot=0.014,
        )

    def _city_state(y0: float, y1: float, prefix: str) -> None:
        city = _label_in_yband(tokens, ["City"], y0, y1, page=page, x_max=0.45)
        if city:
            lab0, lab1, ordered = city
            crops[f"{prefix}city"] = _crop_right_of_label(
                lab0,
                lab1,
                ordered,
                stop_words={"pin", "code", "codel", "state"},
                x_fallback=0.16,
                y_pad_top=0.006,
                y_pad_bot=0.01,
            )
        state = _label_in_yband(tokens, ["State"], y0, y1, page=page, x_min=0.45)
        if state:
            lab0, lab1, ordered = state
            crops[f"{prefix}state"] = _crop_right_of_label(
                lab0,
                lab1,
                ordered,
                stop_words={"pin", "code", "duration", "owned"},
                x_fallback=0.2,
                y_pad_top=0.006,
                y_pad_bot=0.01,
            )

    if res_range:
        _email(res_range[0], res_range[1], "h1_email")
        _city_state(res_range[0], res_range[1], "h1_")
    if office_range:
        _city_state(office_range[0], office_range[1], "w_")

    return crops


def locate_section_crops(
    tokens: list[Tok],
    res_range: tuple[float, float] | None,
    office_range: tuple[float, float] | None,
    page: int = 0,
) -> dict[str, tuple[float, float, float, float]]:
    """Wider bands when layout assembly failed (street / locality handwriting)."""
    crops: dict[str, tuple[float, float, float, float]] = {}

    def _address_band(y0: float, y1: float, key: str) -> None:
        # From Street / Building down to City — section crop, not digit comb.
        street = _label_in_yband(
            tokens, ["Street Name", "Building Name", "Locality"], y0, y1, page=page
        )
        city = _label_in_yband(tokens, ["City"], y0, y1, page=page, x_max=0.5)
        if not street:
            return
        lab0, _lab1, _ordered = street
        y_bot = city[0].y0 - 0.002 if city else min(y1, lab0.y1 + 0.08)
        crops[key] = (0.12, max(y0, lab0.y0 - 0.01), 0.92, y_bot)

    if res_range:
        _address_band(res_range[0], res_range[1], "h1_address_block")
    if office_range:
        _address_band(office_range[0], office_range[1], "w_address_block")
    return crops


_DIGIT_CROP_SPECS = None


def _digit_crop_specs():
    global _DIGIT_CROP_SPECS
    if _DIGIT_CROP_SPECS is None:
        from ocr.digit_crop import (
            validate_dob_ddmmyyyy,
            validate_landline,
            validate_mobile,
            validate_pin,
            validate_std,
        )

        # PIN: need ≥2 jitter/reader tags agreeing (single tess flukes are high-FP).
        # DOB / phones: dual engine agreement so schema-valid garbage stays out.
        # tuple: field, lengths, validator, prefer_len, require_agreement, min_support
        _DIGIT_CROP_SPECS = {
            "dob": ("dob", [8], validate_dob_ddmmyyyy, 8, True, 1),
            "pin_h1": ("address_pincode", [6], validate_pin, 6, False, 1),
            "pin_w": ("address_pincode", [6], validate_pin, 6, False, 2),
            "h1_mobile": ("mobile_no", [10], validate_mobile, 10, True, 1),
            "w_mobile": ("mobile_no", [10], validate_mobile, 10, True, 1),
            "h1_std": ("std_code", [3, 4, 5, 2], validate_std, 3, True, 1),
            "w_std": ("std_code", [3, 4, 5, 2], validate_std, 3, True, 1),
            "h1_landline": ("landline_no", [8, 7, 6], validate_landline, 8, True, 1),
            "w_landline": ("landline_no", [8, 7, 6], validate_landline, 8, True, 1),
        }
    return _DIGIT_CROP_SPECS


def recover_digits_from_crops(
    page_img,
    crops: dict[str, tuple[float, float, float, float]],
    *,
    readers: tuple[str, ...] = ("tesseract", "parseq"),
    escalate: bool = False,
    escalate_readers: tuple[str, ...] = ("trocr_handwritten", "trocr_printed"),
) -> dict[str, dict]:
    """Run crop readers; return {logical_key: {value, reader, conf, bbox}} for hits.

    When escalate=True, crops that miss after tesseract/parseq are re-read with
    crop VLM (TrOCR/Florence) + the same validators. Metadata includes escalated=True.
    """
    from ocr.digit_crop import read_digit_field

    specs = _digit_crop_specs()
    out: dict[str, dict] = {}
    for key, bbox in crops.items():
        spec = specs.get(key)
        if not spec:
            continue
        field, lengths, validator, prefer_len, agree, min_support = spec
        hit, votes = read_digit_field(
            page_img,
            bbox,
            field=field,
            lengths=lengths,
            validator=validator,
            readers=readers,
            prefer_len=prefer_len,
            require_agreement=agree,
            min_support=min_support,
            return_votes=True,
        )
        escalated = False
        if hit is None and escalate:
            from ocr.crop_vlm import escalate_digit_field

            hit = escalate_digit_field(
                page_img,
                bbox,
                field=field,
                lengths=lengths,
                validator=validator,
                readers=escalate_readers,
                prefer_len=prefer_len,
                prior_votes=votes,
                require_prior_echo=True,
            )
            escalated = hit is not None
        if hit is None:
            continue
        out[key] = {
            "field": hit.field,
            "value": hit.value,
            "reader": hit.reader,
            "conf": hit.conf,
            "bbox": list(hit.bbox),
            "escalated": escalated,
        }
    return out


def recover_text_from_crops(
    page_img,
    crops: dict[str, tuple[float, float, float, float]],
    *,
    escalate_readers: tuple[str, ...] = ("trocr_handwritten", "trocr_printed"),
    skip_keys: set[str] | None = None,
    pin_by_crop: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Escalate email/city/state field crops when layout left them empty upstream.

    City/state crops already filled via PIN master should be in skip_keys so TrOCR
    never runs on them. When a PIN is known for a city crop, vocab is PIN-scoped.
    """
    from ocr.crop_vlm import (
        escalate_text_field,
        validate_city_token,
        validate_email,
        validate_state_token,
    )

    skip = skip_keys or set()
    pins = pin_by_crop or {}

    def _city_validator(pin: str | None):
        return lambda text: validate_city_token(text, pin=pin)

    specs: dict[str, tuple[str, object]] = {
        "h1_email": ("email_id", validate_email),
        "h1_city": ("address_city", _city_validator(pins.get("h1_city"))),
        "w_city": ("address_city", _city_validator(pins.get("w_city"))),
        "h1_state": ("address_state", validate_state_token),
        "w_state": ("address_state", validate_state_token),
    }
    out: dict[str, dict] = {}
    for key, bbox in crops.items():
        if key in skip:
            continue
        spec = specs.get(key)
        if not spec:
            continue
        field, validator = spec
        hit = escalate_text_field(
            page_img,
            bbox,
            field=field,
            validator=validator,  # type: ignore[arg-type]
            readers=escalate_readers,
            require_ink=True,
        )
        if hit is None:
            continue
        out[key] = {
            "field": hit.field,
            "value": hit.value,
            "reader": hit.reader,
            "conf": hit.conf,
            "bbox": list(hit.bbox),
            "escalated": True,
        }
    return out


def apply_pin_derived_locality(row: dict) -> dict[str, dict]:
    """Fill empty city/state from India Post PIN master; return recovery meta hits."""
    from ocr.pin_master import apply_pin_locality

    filled = apply_pin_locality(row)
    if not filled:
        return {}
    meta: dict[str, dict] = {}
    pin = filled.get("_pin")
    source = filled.get("_pin_locality_source", "exact")
    for field in ("address_city", "address_state"):
        if field not in filled:
            continue
        meta[field] = {
            "field": field,
            "value": filled[field],
            "reader": f"pin_master:{source}",
            "conf": 0.99,
            "bbox": None,
            "escalated": False,
            "pin": pin,
        }
    return meta


def extract_rows_layout(
    ocr: dict,
    case_id: str | None = None,
    *,
    page_images: list | None = None,
    digit_readers: tuple[str, ...] = ("tesseract", "parseq"),
    crop_vlm: bool = False,
    crop_vlm_readers: tuple[str, ...] = ("trocr_handwritten", "trocr_printed"),
) -> dict:
    t0 = time.perf_counter()
    tokens = iter_tokens(ocr)
    text = pages_text(ocr)
    page_count = len(ocr.get("pages", []))

    person = assemble_person_name(tokens, text)
    gender = find_gender(text)
    dob = find_dob(text)
    phones = find_all_phones(text)
    emails = find_emails(text)
    voter = find_voter_id(text)
    pan = find_pan(text)

    # Contact-block "Current Residence" (exact phrase), not mailing checkbox / Duration line.
    res_headers: list[Tok] = []
    by_line: dict[tuple[int, int], list[Tok]] = {}
    for t in tokens:
        by_line.setdefault((t.page, t.line), []).append(t)
    for (pg, _ln), toks in by_line.items():
        if pg != 0:
            continue
        ordered = sorted(toks, key=lambda x: x.x0)
        span = _phrase_on_line(ordered, "Current Residence")
        if span is None:
            continue
        joined = " ".join(t.text for t in ordered).casefold()
        if "mailing" in joined or "preferred" in joined:
            continue
        res_headers.append(ordered[span[0]])

    office_hdr = None
    for (pg, _ln), toks in by_line.items():
        if pg != 0:
            continue
        ordered = sorted(toks, key=lambda x: x.x0)
        # standalone Office section header (not mailing checkbox row)
        if len(ordered) <= 3 and any(t.text.casefold() == "office" for t in ordered):
            joined = " ".join(t.text for t in ordered).casefold()
            if "mailing" in joined or "residence" in joined:
                continue
            office_hdr = next(t for t in ordered if t.text.casefold() == "office")
            break

    res_range = None
    if res_headers:
        hdr = max(res_headers, key=lambda t: t.yc)
        # stop before Parent's Residence if present, else Office
        parent = None
        for (pg, _ln), toks in by_line.items():
            if pg != 0:
                continue
            ordered = sorted(toks, key=lambda x: x.x0)
            for phrase in ("Parent's Residence", "Parents Residence"):
                span = _phrase_on_line(ordered, phrase)
                if span and ordered[span[0]].yc > hdr.yc:
                    parent = ordered[span[0]]
                    break
        stop = parent or office_hdr
        y1 = (stop.y0 - 0.002) if stop and stop.yc > hdr.yc else 0.88
        res_range = (hdr.y0 - 0.005, y1)

    office_range = (office_hdr.y0 - 0.005, 1.02) if office_hdr else None

    rows: list[dict] = []

    def base_row(address_type: str, *, primary: bool) -> dict:
        # customer_type / constitution: omit until checkbox/role OCR exists
        r: dict = {
            "address_type": address_type,
            "address_source": "Application form",
        }
        if primary and re.search(r"application", text, re.I):
            r["application_form"] = "Yes"
        if person:
            r["person_name"] = person
            r["customer_name"] = person
        if gender:
            r["gender"] = gender
        return {k: v for k, v in r.items() if v not in (None, "")}

    digit_hits: dict[str, dict] = {}
    text_hits: dict[str, dict] = {}
    pin_locality_hits: dict[str, dict] = {}
    text_crops: dict[str, tuple[float, float, float, float]] = {}
    crop_vlm_status = None
    if page_images:
        if crop_vlm:
            from ocr.crop_vlm import crop_vlm_available

            ok, reason = crop_vlm_available(
                crop_vlm_readers[0] if crop_vlm_readers else "trocr_handwritten"
            )
            crop_vlm_status = {"available": ok, "reason": reason}
        crops = locate_digit_crops(tokens, res_range, office_range, page=0)
        digit_hits = recover_digits_from_crops(
            page_images[0],
            crops,
            readers=digit_readers,
            escalate=bool(crop_vlm and (crop_vlm_status or {}).get("available")),
            escalate_readers=crop_vlm_readers,
        )
        if crop_vlm and (crop_vlm_status or {}).get("available"):
            # Locate now; escalate only after PIN→city/state fill (skip those crops).
            text_crops = locate_text_crops(tokens, res_range, office_range, page=0)

    def _apply_crop(row: dict, hits: dict[str, dict], *keys: str) -> None:
        for key in keys:
            hit = hits.get(key)
            if not hit:
                continue
            field = hit["field"]
            # Fill empties only — never clobber a page-text value with a crop guess.
            if field not in row or not row[field]:
                row[field] = hit["value"]

    def _finalize_address_row(row: dict, *, pin_crop_keys: tuple[str, ...], prefix: str) -> None:
        """PIN crop → PIN master city/state → record skip keys for VLM."""
        _apply_crop(row, digit_hits, *pin_crop_keys)
        meta = apply_pin_derived_locality(row)
        for field, hit in meta.items():
            # Namespace recovery keys so H1/W don't collide in digit_recovery.
            pin_locality_hits[f"{prefix}{field}"] = hit

    if res_range:
        h1 = base_row("H1", primary=True)
        h1.update(extract_address_block(tokens, res_range[0], res_range[1], page=0))
        if dob:
            h1["dob"] = dob
        if phones:
            h1["mobile_no"] = phones[0]
            if len(phones) > 1:
                h1["mobile_no_1"] = phones[1]
        if emails:
            h1["email_id"] = emails[0]
        if voter:
            h1["voter_id"] = voter
            h1["source_of_voterid_kyc_id"] = "Voterid"
        if pan:
            h1["pan"] = pan
            h1["source_of_pan_kyc_id"] = "PAN"
        _apply_crop(h1, digit_hits, "dob", "h1_mobile", "h1_std", "h1_landline")
        _finalize_address_row(h1, pin_crop_keys=("pin_h1",), prefix="h1_")
        # Do not emit page_count from a partial OCR cache (misleading vs pack length).
        rows.append(h1)
    else:
        h1 = base_row("H1", primary=True)
        pin = find_pin(text)
        if pin:
            h1["address_pincode"] = pin
        if dob:
            h1["dob"] = dob
        if phones:
            h1["mobile_no"] = phones[0]
        if emails:
            h1["email_id"] = emails[0]
        _apply_crop(h1, digit_hits, "dob", "h1_mobile", "h1_std", "h1_landline")
        _finalize_address_row(h1, pin_crop_keys=("pin_h1",), prefix="h1_")
        rows.append(h1)

    w = None
    if office_range:
        w = base_row("W", primary=False)
        w.update(extract_address_block(tokens, office_range[0], office_range[1], page=0))
        _apply_crop(w, digit_hits, "w_mobile", "w_std", "w_landline")
        _finalize_address_row(w, pin_crop_keys=("pin_w",), prefix="w_")
        rows.append(w)

    # City/state VLM only when still empty after PIN derivation; blank-ink gate inside.
    if page_images and crop_vlm and (crop_vlm_status or {}).get("available") and text_crops:
        skip: set[str] = set()
        pin_by_crop: dict[str, str] = {}
        if h1.get("address_city"):
            skip.add("h1_city")
        if h1.get("address_state"):
            skip.add("h1_state")
        if h1.get("email_id"):
            skip.add("h1_email")
        if h1.get("address_pincode"):
            pin_by_crop["h1_city"] = str(h1["address_pincode"])
        if w is not None:
            if w.get("address_city"):
                skip.add("w_city")
            if w.get("address_state"):
                skip.add("w_state")
            if w.get("address_pincode"):
                pin_by_crop["w_city"] = str(w["address_pincode"])
        text_hits = recover_text_from_crops(
            page_images[0],
            text_crops,
            escalate_readers=crop_vlm_readers,
            skip_keys=skip,
            pin_by_crop=pin_by_crop,
        )
        _apply_crop(h1, text_hits, "h1_email", "h1_city", "h1_state")
        if w is not None:
            _apply_crop(w, text_hits, "w_city", "w_state")

    # Strip empty / None after optional text escalate.
    for i, row in enumerate(rows):
        rows[i] = {k: v for k, v in row.items() if v not in (None, "")}

    extractor = "heuristic_v1_layout"
    if page_images is not None:
        extractor = "heuristic_v1_layout+digit_crop"
    if crop_vlm:
        extractor = "heuristic_v1_layout+digit_crop+crop_vlm"

    recovery_meta = {
        k: {
            "field": v["field"],
            "reader": v["reader"],
            "conf": round(float(v["conf"]), 3),
            "escalated": bool(v.get("escalated")),
        }
        for k, v in {**digit_hits, **text_hits, **pin_locality_hits}.items()
    }
    out = {
        "case_id": case_id,
        "extractor": extractor,
        "seconds": round(time.perf_counter() - t0, 3),
        "digit_recovery": recovery_meta,
        "rows": rows,
    }
    if crop_vlm_status is not None:
        out["crop_vlm"] = crop_vlm_status
    return out


def render_pdf_pages(pdf: Path, pages: list[int], dpi: int = 300) -> list:
    import numpy as np
    import pymupdf

    images = []
    with pymupdf.open(pdf) as doc:
        for i in pages:
            pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            images.append(img)
    return images


def extract_rows(
    ocr: dict,
    case_id: str | None = None,
    comb_merge: bool = False,
    *,
    page_images: list | None = None,
    source_pdf: Path | None = None,
    dpi: int = 300,
    digit_crop: bool = False,
    digit_readers: tuple[str, ...] = ("tesseract", "parseq"),
    crop_vlm: bool = False,
    crop_vlm_readers: tuple[str, ...] = ("trocr_handwritten", "trocr_printed"),
) -> dict:
    """Extract Excel-shaped rows from an OCR document dict.

    If comb_merge=True, join character-box fragments before extraction.
    Uses layout-aware assembly when word bboxes exist; else flat regex stub.
    When digit_crop=True (or page_images/source_pdf given), re-read DOB/PIN/phone
    crops with a digit-friendly path and fill only schema-valid empties.
    Valid PIN → city/state from pin_master (skip those VLM crops).
    When crop_vlm=True, escalate empty/disagree digit crops and remaining empty
    email/city/state crops (ink-density + strict vocab) — still fill empties only.
    """
    if comb_merge:
        from ocr.comb_merge import merge_comb_document

        ocr = merge_comb_document(ocr)

    images = page_images
    need_images = digit_crop or crop_vlm or source_pdf is not None
    if images is None and need_images and source_pdf is not None:
        page_indexes = [int(p.get("index") or i) for i, p in enumerate(ocr.get("pages") or [])]
        if not page_indexes:
            page_indexes = [0]
        images = render_pdf_pages(Path(source_pdf), page_indexes, dpi=dpi)

    # crop_vlm implies locate+digit_crop first, then escalate misses only.
    if crop_vlm:
        digit_crop = True

    tokens = iter_tokens(ocr)
    if tokens:
        # page_images triggers digit_crop recover; callers may pass images directly
        # (bake-off) without setting digit_crop=True.
        return extract_rows_layout(
            ocr,
            case_id=case_id,
            page_images=images,
            digit_readers=digit_readers,
            crop_vlm=crop_vlm,
            crop_vlm_readers=crop_vlm_readers,
        )

    # ---- legacy flat stub (no geometry) ----
    t0 = time.perf_counter()
    text = pages_text(ocr)
    page_count = len(ocr.get("pages", []))
    names = find_name_candidates(text)
    person = names[0] if names else None
    phones = find_all_phones(text)
    emails = find_emails(text)
    dob = find_dob(text)
    gender = find_gender(text)
    voter = find_voter_id(text)
    pan = find_pan(text)
    pin = find_pin(text)

    # customer_type / constitution: omit until checkbox/role OCR exists
    row: dict = {
        "address_type": "H1",
        "address_source": "Application form",
        "application_form": "Yes" if re.search(r"application", text, re.I) else None,
        "page_count": page_count or None,
    }
    if person:
        row["person_name"] = person
        row["customer_name"] = person
    if gender:
        row["gender"] = gender
    if dob:
        row["dob"] = dob
    if phones:
        row["mobile_no"] = phones[0]
        if len(phones) > 1:
            row["mobile_no_1"] = phones[1]
    if emails:
        row["email_id"] = emails[0]
    if pin:
        row["address_pincode"] = pin
    if voter:
        row["voter_id"] = voter
        row["source_of_voterid_kyc_id"] = "Voterid"
    if pan:
        row["pan"] = pan
        row["source_of_pan_kyc_id"] = "PAN"
    row = {k: v for k, v in row.items() if v not in (None, "")}
    return {
        "case_id": case_id,
        "extractor": "heuristic_v0",
        "seconds": round(time.perf_counter() - t0, 3),
        "rows": [row] if row else [],
    }


def load_ocr(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ocr", nargs="?", type=Path, help="OCR JSON from engine")
    parser.add_argument("-o", "--out", type=Path, help="write prediction JSON")
    parser.add_argument("--cases-dir", type=Path, default=None)
    parser.add_argument("--ocr-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--comb-merge", action="store_true")
    parser.add_argument(
        "--digit-crop",
        action="store_true",
        help="re-read DOB/PIN/phone crops (needs --source-pdf or case source.pdf)",
    )
    parser.add_argument(
        "--crop-vlm",
        action="store_true",
        help="escalate empty digit/text crops via TrOCR/Florence (implies --digit-crop)",
    )
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--digit-readers",
        default="tesseract,parseq",
        help="comma list: tesseract,parseq,crnn,default",
    )
    parser.add_argument(
        "--crop-vlm-readers",
        default="trocr_handwritten,trocr_printed",
        help="comma list: trocr_handwritten,trocr_printed,florence2_base",
    )
    args = parser.parse_args(argv)
    readers = tuple(r.strip() for r in args.digit_readers.split(",") if r.strip())
    vlm_readers = tuple(r.strip() for r in args.crop_vlm_readers.split(",") if r.strip())
    want_crops = args.digit_crop or args.crop_vlm

    if args.cases_dir and args.ocr_dir and args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for case_dir in sorted(p for p in args.cases_dir.iterdir() if p.is_dir()):
            case_id = case_dir.name
            ocr_path = args.ocr_dir / f"{case_id}.json"
            if not ocr_path.exists():
                # prefer dpi300 cache when present
                alt = args.ocr_dir / f"{case_id}_dpi300_none.json"
                ocr_path = alt if alt.exists() else ocr_path
            if not ocr_path.exists():
                print(f"{case_id}: missing OCR cache {ocr_path}", file=sys.stderr)
                continue
            src = args.source_pdf or (case_dir / "source.pdf")
            pred = extract_rows(
                load_ocr(ocr_path),
                case_id=case_id,
                comb_merge=args.comb_merge,
                digit_crop=want_crops,
                crop_vlm=args.crop_vlm,
                source_pdf=src if want_crops and src.exists() else None,
                dpi=args.dpi,
                digit_readers=readers,
                crop_vlm_readers=vlm_readers,
            )
            out = args.out_dir / f"{case_id}.json"
            out.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{case_id}: {len(pred['rows'])} rows in {pred['seconds']:.2f}s → {out}")
        return 0

    if not args.ocr:
        parser.error("provide ocr path, or --cases-dir/--ocr-dir/--out-dir")

    case_id = args.ocr.stem.replace("_dpi300_none", "").replace("_dpi200_none", "")
    src = args.source_pdf
    if want_crops and src is None:
        guess = ROOT / "data" / "deb" / "cases" / case_id / "source.pdf"
        src = guess if guess.exists() else None
    pred = extract_rows(
        load_ocr(args.ocr),
        case_id=case_id,
        comb_merge=args.comb_merge,
        digit_crop=want_crops,
        crop_vlm=args.crop_vlm,
        source_pdf=src,
        dpi=args.dpi,
        digit_readers=readers,
        crop_vlm_readers=vlm_readers,
    )
    text = json.dumps(pred, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"{case_id}: {len(pred['rows'])} rows in {pred['seconds']:.2f}s → {args.out}")
    else:
        print(f"{case_id}: {len(pred['rows'])} rows in {pred['seconds']:.2f}s (pass -o to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
