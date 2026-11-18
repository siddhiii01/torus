"""Re-OCR DEB pages with preprocess/DPI variants; rank by gold-token recoverability.

Stdout is metadata only (no document text).

usage:
  python scripts/deb_bakeoff_ocr_variants.py binani --pages 0 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deb_bakeoff_ocr import recoverability, word_stats  # noqa: E402
from deb_extract import extract_rows  # noqa: E402
from deb_score import load_field_groups, load_rows, score_case  # noqa: E402
from ocr.engine import Word, _doctr_words, _finish  # noqa: E402
from ocr.preprocess import clahe_rgb, upscale  # noqa: E402

CASES = ROOT / "data" / "deb" / "cases"
CACHE = ROOT / "reports" / "deb" / "ocr_cache"
SCORE = ROOT / "reports" / "deb" / "scores"


def render_pages(pdf: Path, pages: list[int], dpi: int) -> list[np.ndarray]:
    images = []
    with pymupdf.open(pdf) as doc:
        for i in pages:
            pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            images.append(img)
    return images


def ocr_images(images: list[np.ndarray], page_indexes: list[int]) -> dict:
    pages = []
    for img, idx in zip(images, page_indexes):
        t0 = time.perf_counter()
        try:
            words, conf = _doctr_words(img)
            words = _finish(words)
            status, reason = "ok", ""
            if not words:
                status, reason = "needs_review", "empty"
            elif conf < 0.60:
                status, reason = "needs_review", "low_confidence"
        except Exception as exc:
            words, conf, status, reason = [], None, "needs_review", type(exc).__name__
        pages.append(
            {
                "index": idx,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "reader": "doctr",
                "status": status,
                "reason": reason,
                "seconds": round(time.perf_counter() - t0, 3),
                "words": [
                    {"text": w.text, "bbox": list(w.bbox), "conf": w.conf, "line": w.line}
                    for w in words
                ],
                "mean_conf": None if conf is None else round(float(conf), 4),
            }
        )
        confs = [w.conf for w in words if w.conf is not None]
        mean = sum(confs) / len(confs) if confs else None
        print(
            f"  page {idx}: words={len(words)} conf={mean if mean is None else round(mean, 2)} "
            f"status={status} sec={pages[-1]['seconds']}"
        )
    return {"pages": pages}


def apply_preprocess(images: list[np.ndarray], name: str) -> list[np.ndarray]:
    if name == "none":
        return images
    if name == "clahe":
        return [clahe_rgb(im) for im in images]
    if name == "upscale15":
        return [upscale(im, 1.5) for im in images]
    if name == "clahe_upscale15":
        return [upscale(clahe_rgb(im), 1.5) for im in images]
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--pages", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--dpi", nargs="+", type=int, default=[200, 300])
    parser.add_argument(
        "--preprocess",
        nargs="+",
        default=["none", "clahe"],
        choices=["none", "clahe", "upscale15", "clahe_upscale15"],
    )
    args = parser.parse_args()

    pdf = CASES / args.case_id / "source.pdf"
    gold_rows = load_rows(CASES / args.case_id / "gold.json")
    scored, fuzzy = load_field_groups()
    SCORE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"{'variant':<28} {'words':>6} {'1char%':>7} {'ocrR':>6} {'rowF1':>6} {'fldF1':>6}")

    for dpi in args.dpi:
        print(f"render dpi={dpi} pages={args.pages}")
        base_images = render_pages(pdf, args.pages, dpi)
        for prep in args.preprocess:
            variant = f"dpi{dpi}_{prep}"
            print(f"OCR {variant}")
            images = apply_preprocess(base_images, prep)
            ocr = ocr_images(images, args.pages)
            ocr["variant"] = variant
            cache_path = CACHE / f"{args.case_id}_{variant}.json"
            cache_path.write_text(json.dumps(ocr), encoding="utf-8")

            pred = extract_rows(ocr, case_id=args.case_id)
            scored_result = score_case(gold_rows, pred["rows"], scored, fuzzy)
            row = {
                "variant": variant,
                "word_stats": word_stats(ocr),
                "ocr_recoverability": {
                    k: recoverability(gold_rows, ocr, scored)[k]
                    for k in ("hit", "miss", "recall")
                },
                "extract_rows": scored_result["rows"],
                "extract_fields_micro": scored_result["fields_micro"],
                "cache": str(cache_path),
            }
            results.append(row)
            ws = row["word_stats"]
            print(
                f"{variant:<28} {ws['words']:6d} {ws['single_ratio']:7.3f} "
                f"{row['ocr_recoverability']['recall']:6.3f} "
                f"{row['extract_rows']['f1']:6.3f} "
                f"{row['extract_fields_micro']['f1']:6.3f}"
            )

    out = SCORE / f"bakeoff_ocr_variants_{args.case_id}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    # pick best by ocr recoverability then field f1
    best = max(results, key=lambda r: (r["ocr_recoverability"]["recall"], r["extract_fields_micro"]["f1"]))
    print(f"best_so_far: {best['variant']} ocrR={best['ocr_recoverability']['recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
