"""
Regression gate for the Flowgenix v1 extraction profile (envelope {fields, lineItems}).

    export CELERIS_BASE_URL=http://localhost:4100
    export CELERIS_KEY_FLOWGENIX=clr_flowgenix_...
    deepeval test run eval/deepeval/test_flowgenix.py

LLM-as-judge must stay local and opt-in. This file uses deterministic compares only.
"""

from __future__ import annotations

import json
import os
import re

import pytest
import requests

BASE_URL = os.environ.get("CELERIS_BASE_URL", "http://localhost:4100")
API_KEY = os.environ.get("CELERIS_KEY_FLOWGENIX", "")

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "flowgenix", "golden.json")
try:
    with open(_GOLDEN_PATH, encoding="utf-8") as _f:
        GOLDEN: list[dict] = json.load(_f)
except FileNotFoundError:
    GOLDEN = []

PLACEHOLDERS = {"", "-", "--", "n/a", "na", "null", "none", "nil"}
AMOUNT_RE = re.compile(r"(amount|total|subtotal|tax|price|rate|qty|quantity)", re.I)
DATE_RE = re.compile(r"date", re.I)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _ymd(y, m, d) -> str | None:
    y, m, d = int(y), int(m), int(d)
    if not y or m < 1 or m > 12 or d < 1 or d > 31:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def _to_ymd(raw) -> str | None:
    s = re.sub(r"\s+", " ", str(raw).strip())
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return _ymd(*m.groups())
    m = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$", s)
    if m:
        return _ymd(*m.groups())
    m = re.match(r"^(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-](\d{2,4})$", s)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if not month:
            return None
        year = int(m.group(3))
        if year < 100:
            year += 2000 if year <= 68 else 1900
        return _ymd(year, month, int(m.group(1)))
    m = re.match(r"^(\d{1,2})([/\-.])(\d{1,2})\2(\d{4})$", s)
    if m:
        a, b, year = int(m.group(1)), int(m.group(3)), int(m.group(4))
        if a > 12:
            return _ymd(year, b, a)
        if b > 12:
            return _ymd(year, a, b)
        # Ambiguous: prefer day-first (INR printed dates)
        return _ymd(year, b, a)
    return None


def _date_candidates(raw) -> list[str]:
    s = re.sub(r"\s+", " ", str(raw).strip())
    out: set[str] = set()
    primary = _to_ymd(raw)
    if primary:
        out.add(primary)
    m = re.match(r"^(\d{1,2})([/\-.])(\d{1,2})\2(\d{4})$", s)
    if m:
        a, b, year = int(m.group(1)), int(m.group(3)), int(m.group(4))
        if a <= 12 and b <= 12:
            df, mf = _ymd(year, b, a), _ymd(year, a, b)
            if df:
                out.add(df)
            if mf:
                out.add(mf)
    return list(out)


def _dates_equal(a, b) -> bool:
    if str(a).strip().lower() == str(b).strip().lower():
        return True
    ca, cb = _date_candidates(a), _date_candidates(b)
    return bool(ca) and bool(cb) and bool(set(ca) & set(cb))


def _canon(k: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(k)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _nullish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in PLACEHOLDERS:
        return True
    return False


def _remap(obj):
    if isinstance(obj, list):
        return [_remap(x) for x in obj]
    if isinstance(obj, dict):
        return {_canon(k): _remap(v) for k, v in obj.items()}
    return obj


def _header(obj: dict) -> dict:
    fields = obj.get("fields")
    if isinstance(fields, dict):
        return fields
    return obj


def _lines(obj: dict) -> list:
    for k in ("line_items", "lineitems"):
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return []


def _num(v):
    if _nullish(v):
        return None
    cleaned = re.sub(r"[$€£₹,\s]", "", str(v))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _eq(key: str, a, b) -> bool:
    if _nullish(a) and _nullish(b):
        return True
    if _nullish(a) or _nullish(b):
        return False
    if AMOUNT_RE.search(key):
        na, nb = _num(a), _num(b)
        if na is not None and nb is not None:
            return abs(na - nb) <= 0.01
    if DATE_RE.search(key):
        return _dates_equal(a, b)
    if key == "currency":
        return str(a).strip().lower().replace("₹", "inr").replace("$", "usd") == str(b).strip().lower().replace(
            "₹", "inr"
        ).replace("$", "usd")
    return str(a).strip() == str(b).strip()


def _line_total(row: dict):
    for k in ("line_total", "line_amount", "amount"):
        if not _nullish(row.get(k)):
            return row.get(k)
    return None


def _call(prompt: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "celeris-local-primary",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    assert body["usage"]["total_tokens"] > 0, "contract: usage must be populated"
    return json.loads(body["choices"][0]["message"]["content"])


@pytest.mark.skipif(not GOLDEN, reason="no golden cases yet — run node eval/datasets/flowgenix/assemble.js")
@pytest.mark.skipif(not API_KEY, reason="CELERIS_KEY_FLOWGENIX not set")
@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["name"])
def test_extraction_regression(case: dict):
    out = _call(case["prompt"])
    if case.get("adversarial"):
        return
    if isinstance(out, dict) and out.get("error") == "NO_DATA":
        pytest.fail("NO_DATA on a non-adversarial case")

    got = _remap(out)
    exp = _remap(case["expected"])
    got_h, exp_h = _header(got), _header(exp)

    for f in case.get("required_fields", []):
        ck = _canon(f)
        if ck in ("line_items", "lineitems"):
            assert [r for r in _lines(got) if isinstance(r, dict) and any(not _nullish(v) for v in r.values())], (
                "missing required lineItems"
            )
        else:
            assert not _nullish(got_h.get(ck)), f"missing required field: {f}"

    for k, want in exp_h.items():
        if _nullish(want):
            continue
        if k in ("po_number", "po_date", "invoice_number", "invoice_date", "currency", "total_amount"):
            assert _eq(k, want, got_h.get(k)), f"{k}: expected {want!r}, got {got_h.get(k)!r}"

    exp_lines = [r for r in _lines(exp) if isinstance(r, dict)]
    got_lines = [r for r in _lines(got) if isinstance(r, dict)]
    subset = case.get("lineItemsGoldScope") in ("page1_subset", "subset")
    if exp_lines:
        if subset:
            used: set[int] = set()
            for i, e in enumerate(exp_lines):
                et = _line_total(e)
                found = None
                for j, m in enumerate(got_lines):
                    if j in used:
                        continue
                    q_ok = _eq("quantity", e.get("quantity"), m.get("quantity"))
                    t_ok = _nullish(et) or _eq("line_total", et, _line_total(m))
                    if q_ok and t_ok:
                        found = j
                        break
                assert found is not None, f"line gold[{i}] unmatched in subset mode"
                used.add(found)
            assert len(got_lines) >= len(exp_lines), (
                f"line subset: model has fewer rows ({len(got_lines)}) than gold ({len(exp_lines)})"
            )
        else:
            assert len(got_lines) == len(exp_lines), f"line count: want {len(exp_lines)}, got {len(got_lines)}"
            for i, (e, m) in enumerate(zip(exp_lines, got_lines)):
                assert _eq("quantity", e.get("quantity"), m.get("quantity")), f"line[{i}].quantity"
                et, mt = _line_total(e), _line_total(m)
                if not _nullish(et):
                    assert _eq("line_total", et, mt), f"line[{i}].total"
