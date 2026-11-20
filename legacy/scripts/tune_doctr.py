"""Goal 6a: try docTR settings and preprocessing on a hard subset (handwritten chits).

Prints only per-config hit counts and timings, never document text.
usage: tune_doctr.py <sample_root> [subset]     subset = handwriting (default) | scans
"""
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "1")

import cv2
import numpy as np
import pymupdf

spec = importlib.util.spec_from_file_location("score", Path(__file__).with_name("score.py"))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

FIELDS = score.FIELDS


def render(path, dpi):
    with pymupdf.open(path) as doc:
        out = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
            out.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy())
        return out


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def make_model(det="fast_base", reco="crnn_vgg16_bn", straight=True):
    from doctr.models import ocr_predictor
    kw = {}
    if not straight:
        kw.update(assume_straight_pages=False, straighten_pages=True, detect_orientation=False)
    return ocr_predictor(det_arch=det, reco_arch=reco, pretrained=True, **kw)


# name -> (model kwargs, render dpi, image preprocessing)
CONFIGS = {
    "baseline (200dpi)": (dict(), 200, None),
    "300 dpi": (dict(), 300, None),
    "contrast (CLAHE)": (dict(), 200, clahe),
    "tilt correction": (dict(straight=False), 200, None),
    "det db_resnet50": (dict(det="db_resnet50"), 200, None),
    "reco parseq": (dict(reco="parseq"), 200, None),
    "reco master": (dict(reco="master"), 200, None),
}


def main(root, subset):
    cands = list(csv.DictReader(open("reports/gold_candidates.csv", encoding="utf-8")))
    labels = {r["path"]: r for r in csv.DictReader(open("reports/labels.csv", encoding="utf-8"))}
    if subset == "handwriting":
        idxs = [i for i, c in enumerate(cands, 1) if labels[c["path"]]["doc_type"] == "transport_chit"]
    else:
        idxs = [i for i, c in enumerate(cands, 1)
                if c["kind"] in ("scanned", "mixed", "image") and labels[c["path"]]["doc_type"] != "blank"]
    names = sys.argv[3:] or list(CONFIGS)
    print(f"subset '{subset}': {len(idxs)} files\n")
    print(f"{'config':20} {'hits':>9} {'sec/page':>9}   missed (file:fields)")
    images = {}
    for name in names:
        kw, dpi, prep = CONFIGS[name]
        try:
            model = make_model(**kw)
        except Exception as exc:
            print(f"{name:20} could not load: {type(exc).__name__}")
            continue
        hits = total = pages = 0
        secs = 0.0
        missed = []
        for i in idxs:
            path = root / cands[i - 1]["path"]
            key = (i, dpi)
            if key not in images:
                images[key] = render(path, dpi) if path.suffix.lower() == ".pdf" else None
            imgs = images[key]
            if imgs is None:
                continue
            words = []
            for img in imgs:
                x = prep(img) if prep else img
                t0 = time.perf_counter()
                res = model([x])
                secs += time.perf_counter() - t0
                pages += 1
                words += [w.value for pg in res.pages for b in pg.blocks for ln in b.lines for w in ln.words]
            text = " ".join(words)
            lab = labels[cands[i - 1]["path"]]
            bad = []
            for f in FIELDS:
                want = lab[f].strip()
                if want in ("", "-"):
                    continue
                h = score.MATCHERS[f](text, want)
                if h is None:
                    continue
                total += 1
                hits += bool(h)
                if not h:
                    bad.append(f.replace("total_amount", "total").replace("doc_", ""))
            if bad:
                missed.append(f"#{i}:{'+'.join(bad)}")
        print(f"{name:20} {hits:3d}/{total:<3d}    {secs / max(pages, 1):6.1f}   {' '.join(missed)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "handwriting")
