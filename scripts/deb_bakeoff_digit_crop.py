"""Bake-off: layout vs layout+digit_crop vs layout+digit_crop+crop_vlm (FN only).

Uses existing dpi300 OCR cache (no full-page re-OCR). Re-renders PDF pages only
for crop pixels. Stdout is metadata only (no document text).

Keep crop_vlm only when micro-F1 rises without precision drop vs digit_crop.

usage:
  python scripts/deb_bakeoff_digit_crop.py --cases binani
  python scripts/deb_bakeoff_digit_crop.py --cases binani anbazhakan munuswamy
  python scripts/deb_bakeoff_digit_crop.py --cases binani --crop-vlm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deb_extract import (  # noqa: E402
    extract_rows,
    iter_tokens,
    load_ocr,
    locate_digit_crops,
    locate_text_crops,
    render_pdf_pages,
)
from deb_score import load_field_groups, load_rows, score_case  # noqa: E402

CASES = ROOT / "data" / "deb" / "cases"
OCR_DIR = ROOT / "reports" / "deb" / "ocr_cache"
PRED_DIR = ROOT / "reports" / "deb" / "preds"
SCORE_DIR = ROOT / "reports" / "deb" / "scores"

DIGIT_FIELDS = ("dob", "address_pincode", "std_code", "landline_no", "mobile_no", "email_id")


def resolve_ocr(case_id: str) -> Path | None:
    for name in (f"{case_id}_dpi300_none.json", f"{case_id}.json"):
        path = OCR_DIR / name
        if path.exists():
            return path
    return None


def digit_hit_miss(gold_rows: list[dict], pred_rows: list[dict], recovery: dict) -> dict:
    """Per digit-ish field: gold present / pred match / crop reader fired."""
    from deb_score import align_rows, values_equal

    _scored, fuzzy = load_field_groups()
    pairs, _, _ = align_rows(gold_rows, pred_rows)
    table = {}
    for g, p in pairs:
        for key in DIGIT_FIELDS:
            gv = g.get(key)
            if gv is None or gv == "":
                continue
            ok = values_equal(key, p.get(key), gv, fuzzy)
            table.setdefault(key, {"gold": 0, "hit": 0, "miss": 0})
            table[key]["gold"] += 1
            if ok:
                table[key]["hit"] += 1
            else:
                table[key]["miss"] += 1
    return {
        "per_field": table,
        "crop_readers_fired": list(recovery.keys()),
        "crop_fields": sorted({v.get("field") for v in recovery.values()}),
        "escalated_keys": sorted(k for k, v in recovery.items() if v.get("escalated")),
    }


def _variant_block(score: dict, pred_path: Path, recovery: dict | None = None, crops=None) -> dict:
    block = {
        "extract_rows": score["rows"],
        "extract_fields_micro": score["fields_micro"],
        "digit_fields": {
            k: score["per_field"].get(k, {})
            for k in DIGIT_FIELDS
            if score["per_field"].get(k, {}).get("compared", 0)
        },
        "pred": str(pred_path),
    }
    if recovery is not None:
        block["digit_recovery_meta"] = recovery
        block["crops_located"] = crops or []
    return block


def _keep_vs(base: dict, cand: dict) -> dict:
    bf = base["fields_micro"]
    cf = cand["fields_micro"]
    f1_up = cf["f1"] > bf["f1"] + 1e-9
    prec_ok = cf["precision"] >= bf["precision"] - 1e-9 and cf["precision"] >= 0.99
    return {
        "field_f1_up": f1_up,
        "precision_ok": prec_ok,
        "recommend": bool(f1_up and prec_ok),
        "base_f1": bf["f1"],
        "cand_f1": cf["f1"],
        "base_p": bf["precision"],
        "cand_p": cf["precision"],
        "base_r": bf["recall"],
        "cand_r": cf["recall"],
    }


def run_case(
    case_id: str,
    scored: list[str],
    fuzzy: set[str],
    readers: tuple[str, ...],
    dpi: int,
    *,
    crop_vlm: bool,
    crop_vlm_readers: tuple[str, ...],
) -> dict:
    ocr_path = resolve_ocr(case_id)
    gold_path = CASES / case_id / "gold.json"
    pdf = CASES / case_id / "source.pdf"
    if ocr_path is None:
        return {"case_id": case_id, "error": "missing OCR cache"}
    if not gold_path.exists():
        return {"case_id": case_id, "error": "missing gold"}
    if not pdf.exists():
        return {"case_id": case_id, "error": "missing source.pdf"}

    ocr = load_ocr(ocr_path)
    gold_rows = load_rows(gold_path)
    page_indexes = [int(p.get("index") or i) for i, p in enumerate(ocr.get("pages") or [])] or [0]
    images = render_pdf_pages(pdf, page_indexes, dpi=dpi)

    baseline = extract_rows(ocr, case_id=case_id)
    digit = extract_rows(
        ocr,
        case_id=case_id,
        page_images=images,
        digit_readers=readers,
    )

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    base_path = PRED_DIR / f"{case_id}_layout.json"
    dig_path = PRED_DIR / f"{case_id}_digit_crop.json"
    base_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    dig_path.write_text(json.dumps(digit, indent=2, ensure_ascii=False), encoding="utf-8")

    base_score = score_case(gold_rows, baseline["rows"], scored, fuzzy)
    dig_score = score_case(gold_rows, digit["rows"], scored, fuzzy)
    recovery = digit.get("digit_recovery") or {}

    tokens = iter_tokens(ocr)
    # Approximate section bands for locate diagnostics (real extract uses detected ranges).
    crops = locate_digit_crops(tokens, (0.55, 0.90), (0.78, 1.05), page=0)
    text_crops = locate_text_crops(tokens, (0.55, 0.90), (0.78, 1.05), page=0)

    variants = {
        "layout": _variant_block(base_score, base_path),
        "digit_crop": {
            **_variant_block(dig_score, dig_path, recovery, sorted(crops.keys())),
            "digit_recoverability": digit_hit_miss(gold_rows, digit["rows"], recovery),
        },
    }
    keep = {
        "digit_crop_vs_layout": _keep_vs(base_score, dig_score),
    }

    if crop_vlm:
        from ocr.crop_vlm import crop_vlm_available

        ok, reason = crop_vlm_available(crop_vlm_readers[0] if crop_vlm_readers else "trocr_handwritten")
        if not ok:
            variants["crop_vlm"] = {"error": reason}
            keep["crop_vlm_vs_digit_crop"] = {"recommend": False, "error": reason}
        else:
            vlm = extract_rows(
                ocr,
                case_id=case_id,
                page_images=images,
                digit_readers=readers,
                crop_vlm=True,
                crop_vlm_readers=crop_vlm_readers,
            )
            vlm_path = PRED_DIR / f"{case_id}_crop_vlm.json"
            vlm_path.write_text(json.dumps(vlm, indent=2, ensure_ascii=False), encoding="utf-8")
            vlm_score = score_case(gold_rows, vlm["rows"], scored, fuzzy)
            vlm_recovery = vlm.get("digit_recovery") or {}
            variants["crop_vlm"] = {
                **_variant_block(
                    vlm_score,
                    vlm_path,
                    vlm_recovery,
                    sorted(set(crops) | set(text_crops)),
                ),
                "digit_recoverability": digit_hit_miss(gold_rows, vlm["rows"], vlm_recovery),
            }
            keep["crop_vlm_vs_digit_crop"] = _keep_vs(dig_score, vlm_score)
            keep["crop_vlm_vs_layout"] = _keep_vs(base_score, vlm_score)

    return {
        "case_id": case_id,
        "ocr_cache": str(ocr_path),
        "variants": variants,
        "keep": keep,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["binani"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--readers", default="tesseract,parseq")
    parser.add_argument(
        "--crop-vlm",
        action="store_true",
        help="add layout+digit_crop+crop_vlm_on_fn_only variant (downloads TrOCR on first use)",
    )
    parser.add_argument(
        "--crop-vlm-readers",
        default="trocr_handwritten,trocr_printed",
        help="comma list: trocr_handwritten,trocr_printed,florence2_base",
    )
    args = parser.parse_args()
    readers = tuple(r.strip() for r in args.readers.split(",") if r.strip())
    vlm_readers = tuple(r.strip() for r in args.crop_vlm_readers.split(",") if r.strip())

    scored, fuzzy = load_field_groups()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print(
        f"{'case':<12} {'variant':<12} {'rowF1':>6} {'fldP':>6} {'fldR':>6} {'fldF1':>6} "
        f"{'keep':>5}"
    )
    for case_id in args.cases:
        result = run_case(
            case_id,
            scored,
            fuzzy,
            readers,
            args.dpi,
            crop_vlm=args.crop_vlm,
            crop_vlm_readers=vlm_readers,
        )
        results.append(result)
        if "error" in result:
            print(f"{case_id}: {result['error']}", file=sys.stderr)
            continue
        for name, v in result["variants"].items():
            if "error" in v:
                print(f"{case_id:<12} {name:<12} ERROR {v['error']}")
                continue
            if name == "digit_crop":
                keep = result["keep"]["digit_crop_vs_layout"]["recommend"]
            elif name == "crop_vlm":
                keep = result["keep"].get("crop_vlm_vs_digit_crop", {}).get("recommend", "")
            else:
                keep = ""
            print(
                f"{case_id:<12} {name:<12} "
                f"{v['extract_rows']['f1']:6.3f} "
                f"{v['extract_fields_micro']['precision']:6.3f} "
                f"{v['extract_fields_micro']['recall']:6.3f} "
                f"{v['extract_fields_micro']['f1']:6.3f} "
                f"{str(keep):>5}"
            )
            if name in {"digit_crop", "crop_vlm"} and "digit_recoverability" in v:
                dr = v["digit_recoverability"]["per_field"]
                for field, st in sorted(dr.items()):
                    print(
                        f"  digit {field}: gold={st['gold']} hit={st['hit']} miss={st['miss']}"
                    )
                esc = v["digit_recoverability"].get("escalated_keys") or []
                print(
                    f"  crops_located={v.get('crops_located')} escalated={esc} "
                    f"fired={v['digit_recovery_meta']}"
                )

    out = SCORE_DIR / "bakeoff_digit_crop.json"
    slim = []
    for r in results:
        if "error" in r:
            slim.append(r)
            continue
        slim.append(
            {
                "case_id": r["case_id"],
                "ocr_cache": r["ocr_cache"],
                "keep": r["keep"],
                "variants": {
                    name: (
                        {"error": v["error"]}
                        if "error" in v
                        else {
                            "extract_rows": v["extract_rows"],
                            "extract_fields_micro": v["extract_fields_micro"],
                            "digit_fields": v.get("digit_fields"),
                            "digit_recoverability": v.get("digit_recoverability"),
                            "digit_recovery_meta": v.get("digit_recovery_meta"),
                            "crops_located": v.get("crops_located"),
                        }
                    )
                    for name, v in r["variants"].items()
                },
            }
        )
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
