"""OCR-stage bake-off for DEB: baseline vs comb_merge postprocess.

Does not print document text. Writes scores under reports/deb/scores/.

usage:
  python scripts/deb_bakeoff_ocr.py --cases binani
  python scripts/deb_bakeoff_ocr.py --cases binani anbazhakan munuswamy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deb_extract import extract_rows  # noqa: E402
from deb_score import load_field_groups, load_rows, normalize_field, score_case  # noqa: E402
from ocr.comb_merge import merge_comb_document  # noqa: E402

CASES = ROOT / "data" / "deb" / "cases"
OCR_DIR = ROOT / "reports" / "deb" / "ocr_cache"
PRED_DIR = ROOT / "reports" / "deb" / "preds"
SCORE_DIR = ROOT / "reports" / "deb" / "scores"


def pages_blob(ocr: dict) -> str:
    parts = []
    for page in ocr.get("pages", []):
        words = page.get("words") or []
        if words and isinstance(words[0], dict):
            parts.append(" ".join(w.get("text", "") for w in words))
        else:
            parts.append(" ".join(str(w) for w in words))
    return "\n".join(parts)


def recoverability(gold_rows: list[dict], ocr: dict, scored: list[str]) -> dict:
    """Fraction of non-empty scored gold values whose normalized form appears in OCR text."""
    blob = pages_blob(ocr)
    # normalize blob coarsely for substring checks
    blob_norm = " ".join(blob.casefold().split())
    blob_digits = "".join(ch for ch in blob if ch.isdigit())
    blob_alnum = "".join(ch for ch in blob_norm if ch.isalnum())

    hit = miss = 0
    per_field: dict[str, list[int]] = {}
    for row in gold_rows:
        for key in scored:
            gv = row.get(key)
            if gv is None or gv == "":
                continue
            g = normalize_field(key, gv)
            if not g:
                continue
            ok = False
            if key in {
                "mobile_no",
                "mobile_no_1",
                "mobile_no_2",
                "landline_no",
                "std_code",
                "address_pincode",
                "page_count",
                "aadhaar_no",
            }:
                ok = g in blob_digits
            elif key in {"person_name", "customer_name"}:
                # all tokens present
                toks = [t for t in g.split() if len(t) > 1]
                ok = all(t in blob_norm for t in toks) if toks else g in blob_norm
            elif key in {"voter_id", "pan", "dl_no", "email_id"}:
                ok = g.replace("@", "").replace(".", "") in blob_alnum or g in blob_norm
            else:
                ok = g in blob_norm or g.replace(" ", "") in blob_alnum
            per_field.setdefault(key, [0, 0])
            if ok:
                hit += 1
                per_field[key][0] += 1
            else:
                miss += 1
                per_field[key][1] += 1
    total = hit + miss
    return {
        "hit": hit,
        "miss": miss,
        "recall": round(hit / total, 4) if total else 0.0,
        "per_field_hit_miss": {k: {"hit": v[0], "miss": v[1]} for k, v in per_field.items()},
    }


def word_stats(ocr: dict) -> dict:
    n = singles = 0
    for page in ocr.get("pages", []):
        words = page.get("words") or []
        for w in words:
            t = w.get("text", "") if isinstance(w, dict) else str(w)
            n += 1
            if len(t.strip()) == 1:
                singles += 1
    return {"words": n, "singles": singles, "single_ratio": round(singles / n, 4) if n else 0.0}


def run_case(case_id: str, scored: list[str], fuzzy: set[str]) -> dict:
    ocr_path = OCR_DIR / f"{case_id}.json"
    gold_path = CASES / case_id / "gold.json"
    if not ocr_path.exists():
        return {"case_id": case_id, "error": f"missing OCR cache {ocr_path}"}
    if not gold_path.exists():
        return {"case_id": case_id, "error": f"missing gold {gold_path}"}

    baseline = json.loads(ocr_path.read_text(encoding="utf-8"))
    merged = merge_comb_document(baseline)
    (OCR_DIR / f"{case_id}_comb_merge.json").write_text(
        json.dumps(merged), encoding="utf-8"
    )

    gold_rows = load_rows(gold_path)
    variants = {"baseline": baseline, "comb_merge": merged}
    out = {"case_id": case_id, "variants": {}}

    for name, ocr in variants.items():
        pred = extract_rows(ocr, case_id=case_id)
        pred["variant"] = name
        pred_path = PRED_DIR / f"{case_id}_{name}.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
        scored_result = score_case(gold_rows, pred["rows"], scored, fuzzy)
        out["variants"][name] = {
            "word_stats": word_stats(ocr),
            "ocr_recoverability": recoverability(gold_rows, ocr, scored),
            "extract_rows": scored_result["rows"],
            "extract_fields_micro": scored_result["fields_micro"],
            "pred": str(pred_path),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["binani"])
    args = parser.parse_args()

    scored, fuzzy = load_field_groups()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"{'case':<12} {'variant':<12} {'words':>6} {'1char%':>7} {'ocrR':>6} {'rowF1':>6} {'fldF1':>6}")
    for case_id in args.cases:
        result = run_case(case_id, scored, fuzzy)
        results.append(result)
        if "error" in result:
            print(f"{case_id}: {result['error']}", file=sys.stderr)
            continue
        for name, v in result["variants"].items():
            ws = v["word_stats"]
            print(
                f"{case_id:<12} {name:<12} {ws['words']:6d} {ws['single_ratio']:7.3f} "
                f"{v['ocr_recoverability']['recall']:6.3f} "
                f"{v['extract_rows']['f1']:6.3f} "
                f"{v['extract_fields_micro']['f1']:6.3f}"
            )

    out = SCORE_DIR / "bakeoff_ocr_v0.json"
    # strip per_field detail bulk if huge — keep summary
    slim = []
    for r in results:
        if "error" in r:
            slim.append(r)
            continue
        vr = {}
        for name, v in r["variants"].items():
            vr[name] = {
                "word_stats": v["word_stats"],
                "ocr_recoverability": {
                    "hit": v["ocr_recoverability"]["hit"],
                    "miss": v["ocr_recoverability"]["miss"],
                    "recall": v["ocr_recoverability"]["recall"],
                },
                "extract_rows": v["extract_rows"],
                "extract_fields_micro": v["extract_fields_micro"],
            }
        slim.append({"case_id": r["case_id"], "variants": vr})
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
