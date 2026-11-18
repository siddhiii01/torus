"""India Post PIN → city/state lookup + closed vocab validators.

PIN-derived locality is the precision-safe path for address_city / address_state:
once digit_crop recovers a valid 6-digit PIN, fill city/state from this master and
skip free-text VLM on those crops. Official PO locality names can differ from the
handwritten city string (suburb vs city) — store the master city as canonical.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MASTER = _ROOT / "schemas" / "deb" / "pin_master.json"

# Closed Indian states / UTs — keys are alphanumeric casefold; values are DEB emit form.
IN_STATES: dict[str, str] = {
    "andamanandnicobarislands": "Andaman And Nicobar Islands",
    "andhrapradesh": "AndhraPradesh",
    "arunachalpradesh": "ArunachalPradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "dadraandnagarhavelianddamananddiu": "Dadra And Nagar Haveli And Daman And Diu",
    "damananddiu": "Daman And Diu",
    "delhi": "Delhi",
    "nctofdelhi": "Delhi",
    "newdelhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachalpradesh": "HimachalPradesh",
    "jammuandkashmir": "JammuAndKashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhyapradesh": "MadhyaPradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamilnadu": "TamilNadu",
    "tn": "TamilNadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttarpradesh": "UttarPradesh",
    "up": "UttarPradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "westbengal": "Westbengal",
    "wb": "Westbengal",
}

_FORM_LABELS = {
    "city",
    "state",
    "pin",
    "code",
    "india",
    "duration",
    "owned",
    "rented",
    "landmark",
    "yrs",
    "mths",
}


@dataclass(frozen=True)
class PinLocality:
    pin: str
    city: str
    state: str
    source: str  # exact | override | prefix3


def _fold_alpha(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").casefold())


def normalize_place(text: str) -> str:
    """Collapse spaces/punct; keep letters for near-exact compare."""
    return _fold_alpha(text)


@lru_cache(maxsize=1)
def _load_master(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_MASTER
    if not p.is_file():
        return {"pins": {}, "prefix3": {}, "overrides": {}, "cities": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "pins": {str(k): v for k, v in (data.get("pins") or {}).items()},
        "prefix3": {str(k): v for k, v in (data.get("prefix3") or {}).items()},
        "overrides": {str(k): v for k, v in (data.get("overrides") or {}).items()},
        "cities": list(data.get("cities") or []),
    }


def reload_master(path: str | Path | None = None) -> None:
    _load_master.cache_clear()
    if path is not None:
        _load_master(str(path))


def lookup_pin(pin: str, *, master_path: str | Path | None = None) -> PinLocality | None:
    """Return canonical city/state for a validated 6-digit PIN."""
    digits = re.sub(r"\D", "", str(pin or ""))
    if len(digits) != 6 or digits[0] == "0":
        return None
    data = _load_master(str(master_path) if master_path else None)
    for source, table, key in (
        ("override", data["overrides"], digits),
        ("exact", data["pins"], digits),
        ("prefix3", data["prefix3"], digits[:3]),
    ):
        hit = table.get(key)
        if not hit:
            continue
        if isinstance(hit, (list, tuple)) and len(hit) >= 2:
            city, state = str(hit[0]).strip(), str(hit[1]).strip()
        elif isinstance(hit, dict):
            city, state = str(hit.get("city") or "").strip(), str(hit.get("state") or "").strip()
        else:
            continue
        if city and state:
            return PinLocality(pin=digits, city=city, state=state, source=source)
    return None


def city_candidates_for_pin(pin: str | None) -> set[str]:
    """PIN-scoped city names for vocab gating (canonical city + national list fallback)."""
    data = _load_master()
    national = {c for c in data["cities"] if c}
    if not pin:
        return national
    loc = lookup_pin(pin)
    if not loc:
        return national
    out = {loc.city}
    # Peer cities sharing the sorting-district prefix — still tight vs national.
    for other_pin, pair in data["pins"].items():
        if other_pin.startswith(loc.pin[:3]) and isinstance(pair, (list, tuple)) and pair:
            out.add(str(pair[0]))
    return out or national


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def near_exact_place(text: str, candidates: set[str] | list[str], *, max_edits: int = 1) -> str | None:
    """Accept only exact / near-exact after normalize — not wide fuzzy mop."""
    needle = normalize_place(text)
    if len(needle) < 4 or needle in _FORM_LABELS:
        return None
    best: tuple[int, str] | None = None
    for cand in candidates:
        key = normalize_place(cand)
        if not key or len(key) < 3:
            continue
        if needle == key:
            return cand
        # Allow 1 edit only when both sides are long enough (avoids I/L → city).
        limit = max_edits if min(len(needle), len(key)) >= 5 else 0
        if limit <= 0:
            continue
        d = _levenshtein(needle, key)
        if d <= limit and (best is None or d < best[0]):
            best = (d, cand)
    return best[1] if best else None


def validate_state_token(text: str) -> str | None:
    """Closed state list — exact / near-exact after normalize."""
    cleaned = re.sub(r"[^A-Za-z\s]", " ", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) < 2:
        return None
    key = _fold_alpha(cleaned)
    if key in _FORM_LABELS or "duration" in cleaned.casefold():
        return None
    if key in IN_STATES:
        return IN_STATES[key]
    # near-exact against canonical fold keys (West Begnal → Westbengal)
    best: tuple[int, str] | None = None
    for fold, canon in IN_STATES.items():
        if len(fold) < 4:
            continue
        limit = 1 if min(len(key), len(fold)) >= 6 else 0
        if limit == 0:
            continue
        d = _levenshtein(key, fold)
        if d <= limit and (best is None or d < best[0]):
            best = (d, canon)
    return best[1] if best else None


def validate_city_token(text: str, *, pin: str | None = None) -> str | None:
    """Strict city vocab: PIN-scoped when pin known, else national district list."""
    alphas = re.sub(r"[^A-Za-z\s]", " ", text or "")
    alphas = re.sub(r"\s+", " ", alphas).strip()
    if len(normalize_place(alphas)) < 4:
        return None
    if normalize_place(alphas) in _FORM_LABELS:
        return None
    cands = city_candidates_for_pin(pin)
    hit = near_exact_place(alphas, cands, max_edits=1)
    if hit:
        return hit
    return None


def apply_pin_locality(row: dict, *, pin_key: str = "address_pincode") -> dict:
    """Fill empty city/state from PIN master. Never clobbers existing values."""
    pin = row.get(pin_key)
    if not pin:
        return {}
    loc = lookup_pin(str(pin))
    if not loc:
        return {}
    filled: dict = {}
    if not row.get("address_city"):
        row["address_city"] = loc.city
        filled["address_city"] = loc.city
    if not row.get("address_state"):
        row["address_state"] = loc.state
        filled["address_state"] = loc.state
    if filled:
        filled["_pin_locality_source"] = loc.source
        filled["_pin"] = loc.pin
    return filled
