"""Goal 5, step 2: score cached OCR output against the hand labels.

For each file and engine it checks whether the labelled vendor, document number,
date and total appear anywhere in the output (all pages). Values are normalized
first, so 10/07/2026, 10-Jul-26 and 2026-07-10 all count as the same date, and
59.700,00 and 59,700.00 as the same amount.

Also scores a derived "router" engine: per page, use the text layer if it holds
at least MIN_CHARS characters, otherwise use docTR.

Prints only aggregate numbers and file indexes, never document text.
Writes reports/benchmark_results.csv (per file / field / engine, True or False).
"""
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

CACHE = Path("reports/ocr_cache")
MIN_CHARS = 25
ENGINES = ["textlayer", "tesseract", "doctr", "router"]
FIELDS = ["doc_number", "doc_date", "total_amount", "vendor"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
LEGAL = {"llc", "fze", "fzco", "fzc", "ltd", "limited", "pty", "the", "l", "c", "of", "and", "co", "on", "behalf"}


# ---------- matchers ----------
def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_number(text, want):
    w = alnum(want)
    if len(w) >= 6:
        return w in alnum(text)
    return re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", text.lower()) is not None


def match_date(text, iso):
    y, m, d = (int(x) for x in iso.split("-"))
    sep = r"[\s./\-]{1,3}"
    day = rf"0?{d}"
    mon = rf"(?:0?{m}|{MONTHS[m - 1]}[a-z]*)"
    yr = rf"(?:{y}|{y % 100:02d})"
    pats = [rf"(?<!\d){day}{sep}{mon}{sep}{yr}(?!\d)",           # d m y
            rf"(?<!\d){y}{sep}0?{m}{sep}{day}(?!\d)",            # y m d
            rf"(?<!\d)0?{m}{sep}{day}{sep}{yr}(?!\d)"]           # m d y
    t = text.lower()
    return any(re.search(p, t) for p in pats)


_NUM = re.compile(r"\d{1,3}(?:[,. ]\d{3})+(?:[.,]\d{1,6})?|\d+(?:[.,]\d{1,6})?")


def to_float(tok):
    tok = tok.replace(" ", "")
    if "." in tok and "," in tok:
        dec = "." if tok.rfind(".") > tok.rfind(",") else ","
        return float(tok.replace("," if dec == "." else ".", "").replace(dec, "."))
    for s in (",", "."):
        if s in tok:
            if tok.count(s) > 1:
                return float(tok.replace(s, ""))
            tail = tok.split(s)[1]
            return float(tok.replace(s, "")) if len(tail) == 3 else float(tok.replace(s, "."))
    return float(tok)


def match_total(text, want):
    target = float(want)
    for tok in _NUM.findall(text):
        try:
            if abs(to_float(tok) - target) < 0.005:
                return True
        except ValueError:
            pass
    return False


def match_vendor(text, want):
    toks = [t for t in re.findall(r"[a-z0-9]+", want.lower()) if t not in LEGAL and len(t) >= 3]
    if not toks:
        return None
    have = set(re.findall(r"[a-z0-9]+", text.lower()))
    hits = sum(1 for t in toks if t in have or difflib.get_close_matches(t, have, n=1, cutoff=0.85))
    return hits / len(toks) >= 0.7


MATCHERS = {"doc_number": match_number, "doc_date": match_date,
            "total_amount": match_total, "vendor": match_vendor}


# ---------- data ----------
def load_pages(idx, engine):
    if engine == "router":
        tl, dt = load_pages(idx, "textlayer"), load_pages(idx, "doctr")
        return [t if len(" ".join(t["words"])) >= MIN_CHARS else d for t, d in zip(tl, dt)]
    return json.loads((CACHE / f"{idx:02d}_{engine}.json").read_text(encoding="utf-8"))["pages"]


def group_of(kind, doc_type):
    if doc_type == "transport_chit":
        return "handwritten photo"
    if doc_type == "photo":
        return "plain photo"
    return {"digital": "digital pdf", "scanned": "scanned pdf", "mixed": "scanned pdf",
            "image": "plain photo"}[kind]


def main():
    cands = list(csv.DictReader(open("reports/gold_candidates.csv", encoding="utf-8")))
    labels = {r["path"]: r for r in csv.DictReader(open("reports/labels.csv", encoding="utf-8"))}
    rows = []
    speed = defaultdict(lambda: [0.0, 0])
    for idx, c in enumerate(cands, 1):
        lab = labels[c["path"]]
        grp = group_of(c["kind"], lab["doc_type"])
        for eng in ENGINES:
            pages = load_pages(idx, eng)
            text = " ".join(" ".join(p["words"]) for p in pages)
            if eng in ("tesseract", "doctr"):
                speed[eng][0] += sum(p["sec"] for p in pages)
                speed[eng][1] += len(pages)
            for f in FIELDS:
                want = lab[f].strip()
                if want in ("", "-"):
                    continue
                hit = MATCHERS[f](text, want)
                if hit is None:
                    continue
                rows.append({"idx": idx, "group": grp, "doc_type": lab["doc_type"],
                             "engine": eng, "field": f, "hit": hit})

    with open("reports/benchmark_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "group", "doc_type", "engine", "field", "hit"])
        w.writeheader()
        w.writerows(rows)

    def rate(sel):
        return f"{sum(r['hit'] for r in sel):3d}/{len(sel):<3d} {100 * sum(r['hit'] for r in sel) / len(sel):4.0f}%" if sel else "   n/a    "

    print("Field hit rate per engine (all labelled fields, all pages)\n")
    print(f"{'':14}" + "".join(f"{e:>14}" for e in ENGINES))
    for f in FIELDS:
        print(f"{f:14}" + "".join(f"{rate([r for r in rows if r['engine'] == e and r['field'] == f]):>14}" for e in ENGINES))
    print(f"{'ALL FIELDS':14}" + "".join(f"{rate([r for r in rows if r['engine'] == e]):>14}" for e in ENGINES))

    print("\nHit rate by document group (all fields)\n")
    print(f"{'':20}" + "".join(f"{e:>14}" for e in ENGINES))
    for g in ["digital pdf", "scanned pdf", "handwritten photo", "plain photo"]:
        print(f"{g:20}" + "".join(f"{rate([r for r in rows if r['engine'] == e and r['group'] == g]):>14}" for e in ENGINES))

    print("\nSpeed on laptop CPU (sec/page)")
    for e, (s, n) in speed.items():
        print(f"  {e:10} {s / n:5.1f}  ({n} pages)")

    print("\nFiles where the router missed a labelled field:")
    miss = defaultdict(list)
    for r in rows:
        if r["engine"] == "router" and not r["hit"]:
            miss[r["idx"]].append(r["field"])
    for i, fs in sorted(miss.items()):
        print(f"  #{i:02d} {cands[i - 1]['kind']:8} {labels[cands[i - 1]['path']]['doc_type']:20} {', '.join(fs)}")


if __name__ == "__main__":
    main()
