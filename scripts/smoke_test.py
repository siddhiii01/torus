"""Goal 4: smoke test of the candidate engines on a few gold files.

Runs the PDF text layer, Tesseract and docTR on page 1 of three gold files and
prints only counts, timings and whether the labelled document number appears
in the output. It prints and saves no document text.

usage: smoke_test.py <sample_root>
"""
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pymupdf
import pytesseract

# 1-based indexes into reports/gold_candidates.csv (digital, clean scan, handwritten photo)
CASES = [(3, "digital invoice"), (34, "clean scan"), (6, "handwritten photo")]
DPI = 200


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def render(path: Path) -> np.ndarray:
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def text_layer(path: Path, _img):
    with pymupdf.open(path) as doc:
        words = [w[4] for w in doc[0].get_text("words")]
    return words, None


def tesseract(_path, img):
    d = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
    pairs = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    return [t for t, _ in pairs], (sum(c for _, c in pairs) / len(pairs) / 100 if pairs else 0.0)


def make_doctr():
    from doctr.models import ocr_predictor
    t0 = time.perf_counter()
    model = ocr_predictor(pretrained=True)
    load = time.perf_counter() - t0

    def run(_path, img):
        res = model([img])
        words = [w for pg in res.pages for b in pg.blocks for ln in b.lines for w in ln.words]
        confs = [w.confidence for w in words]
        return [w.value for w in words], (sum(confs) / len(confs) if confs else 0.0)
    return run, load


def main(root: Path):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"
    cands = list(csv.DictReader(open("reports/gold_candidates.csv", encoding="utf-8")))
    labels = {r["path"]: r for r in csv.DictReader(open("reports/labels.csv", encoding="utf-8"))}

    print("loading docTR models (first run downloads them)...")
    doctr_run, load_s = make_doctr()
    print(f"docTR model load: {load_s:.1f}s\n")

    engines = [("text-layer", text_layer), ("tesseract", tesseract), ("docTR", doctr_run)]
    print(f"{'file':20} {'engine':11} {'sec':>6} {'words':>6} {'conf':>5}  doc-number-found")
    for idx, tag in CASES:
        rel = cands[idx - 1]["path"]
        path = root / rel
        want = norm(labels[rel]["doc_number"])
        img = render(path)
        for name, fn in engines:
            t0 = time.perf_counter()
            try:
                words, conf = fn(path, img)
                sec = time.perf_counter() - t0
                found = want in norm("".join(words))
                c = "  n/a" if conf is None else f"{conf:5.2f}"
                print(f"#{idx} {tag:16} {name:11} {sec:6.1f} {len(words):6d} {c}  {'yes' if found else 'NO'}")
            except Exception as exc:
                print(f"#{idx} {tag:16} {name:11} FAILED: {type(exc).__name__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
