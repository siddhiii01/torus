"""Score DEB Excel-shaped predictions against validated gold.

Aligns rows by (person_key, address_type, address_source), with address_line1
as disambiguator on collisions. Field metrics use normalized structured
equality on schemas/deb/field_groups.json → scored — never OCR substring
presence (that is scripts/score.py and is out of scope here).

usage:
  python scripts/deb_score.py data/deb/cases/munuswamy/gold.json pred.json
  python scripts/deb_score.py --cases-dir data/deb/cases --pred-dir reports/deb/preds
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "deb"

ADDRESS_SOURCE_ALIASES = {
    "applicationform": "applicationform",
    "application form": "applicationform",
    "application_form": "applicationform",
    "voterid": "voterid",
    "voter id": "voterid",
    "voter_id": "voterid",
    "rationcard": "rationcard",
    "ration card": "rationcard",
    "ration_card": "rationcard",
    "aadhaarcard": "aadhaarcard",
    "aadhaar card": "aadhaarcard",
    "aadhaar": "aadhaarcard",
    "dl": "dl",
    "driving licence": "dl",
    "driving license": "dl",
    "agreement of guarantee": "agreement_of_guarantee",
    "aggreement of guarantee": "agreement_of_guarantee",
    "aggrement of guarantee": "agreement_of_guarantee",
    "agreement of gurantee": "agreement_of_guarantee",
    "dpn": "dpn",
    "guarantor's form": "guarantor_form",
    "guarantor form": "guarantor_form",
    "guarantor_form": "guarantor_form",
}

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def load_field_groups() -> tuple[list[str], set[str]]:
    data = json.loads((SCHEMA / "field_groups.json").read_text(encoding="utf-8"))
    scored = [item["key"] for item in data["scored"]]
    fuzzy = {item["key"] for item in data["scored"] if item.get("match") == "fuzzy"}
    return scored, fuzzy


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_text(v) -> str:
    if v is None:
        return ""
    return collapse_ws(str(v)).casefold()


def norm_person(v) -> str:
    s = norm_text(v)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return collapse_ws(s)


def norm_address_type(v) -> str:
    return re.sub(r"\s+", "", str(v or "")).upper()


def norm_address_source(v) -> str:
    raw = collapse_ws(str(v or "")).casefold()
    if not raw:
        return ""
    if raw in ADDRESS_SOURCE_ALIASES:
        return ADDRESS_SOURCE_ALIASES[raw]
    # strip punctuation to spaces then underscores
    cleaned = re.sub(r"[^a-z0-9]+", " ", raw)
    cleaned = collapse_ws(cleaned)
    if cleaned in ADDRESS_SOURCE_ALIASES:
        return ADDRESS_SOURCE_ALIASES[cleaned]
    return cleaned.replace(" ", "_")


def digits_only(v) -> str:
    return re.sub(r"\D", "", str(v or ""))


def norm_date(v) -> str:
    if v is None or v == "":
        return ""
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    s = str(v).strip()
    # already ISO
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # datetime dump
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    # d/m/y or d-m-y
    m = re.match(r"^(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        return f"{y:04d}-{mo:02d}-{d:02d}"
    # 16-Mar-1971 / 16 Mar 1971
    m = re.match(r"^(\d{1,2})[\s\-./]+([A-Za-z]{3,})[\s\-./]+(\d{2,4})$", s)
    if m:
        d = int(m.group(1))
        mon = MONTHS.get(m.group(2)[:3].lower())
        y = int(m.group(3))
        if mon:
            if y < 100:
                y += 2000 if y < 50 else 1900
            return f"{y:04d}-{mon:02d}-{d:02d}"
    return norm_text(s)


def normalize_field(key: str, v) -> str:
    if v is None or v == "":
        return ""
    if key == "dob":
        return norm_date(v)
    if key in {
        "mobile_no",
        "mobile_no_1",
        "mobile_no_2",
        "mobile_no_3",
        "mobile_no_4",
        "random_contact_no",
        "landline_no",
        "std_code",
        "address_pincode",
        "aadhaar_no",
        "page_count",
    }:
        return digits_only(v)
    if key == "address_type":
        return norm_address_type(v)
    if key == "address_source" or key.startswith("source_of_"):
        return norm_address_source(v) if key == "address_source" else norm_text(v).replace(" ", "")
    if key in {"person_name", "customer_name"}:
        return norm_person(v)
    if key == "email_id":
        return norm_text(v)
    if key in {"voter_id", "pan", "dl_no", "passport", "ration_card", "cin", "savings_account_no"}:
        return re.sub(r"[^a-z0-9]", "", norm_text(v))
    return norm_text(v)


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s) if t}


def fuzzy_match(pred: str, gold: str, threshold: float = 0.8) -> bool:
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pt, gt = tokens(pred), tokens(gold)
    if not gt:
        return not pt
    overlap = len(pt & gt) / len(gt)
    return overlap >= threshold


def values_equal(key: str, pred_v, gold_v, fuzzy_keys: set[str]) -> bool:
    p = normalize_field(key, pred_v)
    g = normalize_field(key, gold_v)
    if key in fuzzy_keys:
        return fuzzy_match(p, g)
    return p == g


def person_key(row: dict) -> str:
    return norm_person(row.get("person_name") or row.get("customer_name") or "")


def primary_key(row: dict) -> tuple[str, str, str]:
    return (
        person_key(row),
        norm_address_type(row.get("address_type")),
        norm_address_source(row.get("address_source")),
    )


def line1_key(row: dict) -> str:
    return normalize_field("address_line1", row.get("address_line1"))


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    raise ValueError(f"expected list or {{rows: [...]}} in {path}")


def align_rows(gold_rows: list[dict], pred_rows: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Return (pairs, unmatched_gold, unmatched_pred)."""
    gold_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for g in gold_rows:
        gold_by_key[primary_key(g)].append(g)

    pairs: list[tuple[dict, dict]] = []
    unmatched_pred: list[dict] = []
    used_gold_ids: set[int] = set()

    for p in pred_rows:
        key = primary_key(p)
        candidates = [g for g in gold_by_key.get(key, []) if id(g) not in used_gold_ids]
        if not candidates:
            unmatched_pred.append(p)
            continue
        if len(candidates) == 1:
            g = candidates[0]
        else:
            pl1 = line1_key(p)
            by_l1 = [g for g in candidates if line1_key(g) == pl1]
            if len(by_l1) == 1:
                g = by_l1[0]
            elif by_l1:
                g = by_l1[0]  # stable first
            else:
                unmatched_pred.append(p)
                continue
        pairs.append((g, p))
        used_gold_ids.add(id(g))

    unmatched_gold = [g for g in gold_rows if id(g) not in used_gold_ids]
    return pairs, unmatched_gold, unmatched_pred


def score_case(gold_rows: list[dict], pred_rows: list[dict], scored: list[str], fuzzy: set[str]) -> dict:
    pairs, unmatched_gold, unmatched_pred = align_rows(gold_rows, pred_rows)
    n_gold, n_pred = len(gold_rows), len(pred_rows)
    n_aligned = len(pairs)
    row_recall = n_aligned / n_gold if n_gold else 0.0
    row_precision = n_aligned / n_pred if n_pred else 0.0
    row_f1 = (
        2 * row_precision * row_recall / (row_precision + row_recall)
        if (row_precision + row_recall)
        else 0.0
    )

    field_stats: dict[str, dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0, "compared": 0} for k in scored
    }
    field_details: list[dict] = []

    for g, p in pairs:
        for key in scored:
            gv = g.get(key)
            pv = p.get(key)
            g_empty = gv is None or gv == ""
            p_empty = pv is None or pv == ""
            if g_empty and p_empty:
                continue
            field_stats[key]["compared"] += 1
            ok = values_equal(key, pv, gv, fuzzy)
            if ok:
                field_stats[key]["tp"] += 1
            elif g_empty and not p_empty:
                field_stats[key]["fp"] += 1
            elif p_empty and not g_empty:
                field_stats[key]["fn"] += 1
            else:
                # both present but mismatch: count as fp+fn for F1 symmetry
                field_stats[key]["fp"] += 1
                field_stats[key]["fn"] += 1
            field_details.append(
                {
                    "person_key": person_key(g),
                    "address_type": norm_address_type(g.get("address_type")),
                    "address_source": norm_address_source(g.get("address_source")),
                    "field": key,
                    "match": ok,
                    "gold_norm": normalize_field(key, gv),
                    "pred_norm": normalize_field(key, pv),
                }
            )

    per_field = {}
    tot_tp = tot_fp = tot_fn = 0
    for key, st in field_stats.items():
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_field[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "compared": st["compared"],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "rows": {
            "gold": n_gold,
            "pred": n_pred,
            "aligned": n_aligned,
            "unmatched_gold": len(unmatched_gold),
            "unmatched_pred": len(unmatched_pred),
            "precision": round(row_precision, 4),
            "recall": round(row_recall, 4),
            "f1": round(row_f1, 4),
        },
        "fields_micro": {
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "per_field": per_field,
        "field_details": field_details,
    }


def score_paths(gold_path: Path, pred_path: Path, scored: list[str], fuzzy: set[str]) -> dict:
    result = score_case(load_rows(gold_path), load_rows(pred_path), scored, fuzzy)
    result["gold"] = str(gold_path)
    result["pred"] = str(pred_path)
    return result


def print_summary(result: dict, case_id: str | None = None) -> None:
    label = case_id or Path(result.get("gold", "?")).parent.name
    rows = result["rows"]
    fields = result["fields_micro"]
    print(
        f"{label}: rows P/R/F1 "
        f"{rows['precision']:.3f}/{rows['recall']:.3f}/{rows['f1']:.3f} "
        f"(aligned {rows['aligned']}/{rows['gold']} gold, {rows['pred']} pred) | "
        f"fields micro P/R/F1 {fields['precision']:.3f}/{fields['recall']:.3f}/{fields['f1']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", nargs="?", type=Path, help="gold.json path")
    parser.add_argument("pred", nargs="?", type=Path, help="prediction json path")
    parser.add_argument("--cases-dir", type=Path, default=None)
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write full JSON report")
    parser.add_argument("--csv", type=Path, default=None, help="write per-field CSV")
    args = parser.parse_args(argv)

    scored, fuzzy = load_field_groups()
    results: list[dict] = []

    if args.cases_dir and args.pred_dir:
        for gold_path in sorted(args.cases_dir.glob("*/gold.json")):
            case_id = gold_path.parent.name
            pred_path = args.pred_dir / f"{case_id}.json"
            if not pred_path.exists():
                pred_path = args.pred_dir / case_id / "pred.json"
            if not pred_path.exists():
                print(f"{case_id}: missing prediction at {args.pred_dir}", file=sys.stderr)
                continue
            result = score_paths(gold_path, pred_path, scored, fuzzy)
            result["case_id"] = case_id
            results.append(result)
            print_summary(result, case_id)
    elif args.gold and args.pred:
        result = score_paths(args.gold, args.pred, scored, fuzzy)
        results.append(result)
        print_summary(result)
    else:
        parser.error("provide gold+pred paths, or --cases-dir and --pred-dir")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # drop bulky details from aggregate unless single case
        payload = []
        for r in results:
            slim = {k: v for k, v in r.items() if k != "field_details"}
            payload.append(slim)
        args.out.write_text(json.dumps(payload if len(payload) > 1 else payload[0], indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    if args.csv and results:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "field", "tp", "fp", "fn", "compared", "precision", "recall", "f1"])
            for r in results:
                case_id = r.get("case_id") or Path(r.get("gold", "")).parent.name
                for field, st in r["per_field"].items():
                    if st["compared"] == 0 and st["tp"] == 0:
                        continue
                    w.writerow(
                        [
                            case_id,
                            field,
                            st["tp"],
                            st["fp"],
                            st["fn"],
                            st["compared"],
                            st["precision"],
                            st["recall"],
                            st["f1"],
                        ]
                    )
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
