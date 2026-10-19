"""Goal 5, step 1: run the engines on every page of every gold file and cache the output.

Cache: reports/ocr_cache/<NN>_<engine>.json (git-ignored; contains document text,
stays local, is never printed). Resumable: finished (file, engine) pairs are skipped.

usage: run_ocr.py <sample_root>
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pymupdf
import pytesseract
from PIL import Image

CACHE = Path("reports/ocr_cache")
DPI = 200
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def pages_of(path: Path):
    """Yield (page_index, rgb_array, text_layer_words) for every page."""
    if path.suffix.lower() in IMAGE_EXTS:
        yield 0, np.array(Image.open(path).convert("RGB")), []
        return
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            words = [w[4] for w in page.get_text("words")]
            pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            yield i, img, words


def run_tesseract(img):
    d = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
    pairs = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    return [t for t, _ in pairs], (sum(c for _, c in pairs) / len(pairs) / 100 if pairs else 0.0)


def make_doctr():
    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True)

    def run(img):
        res = model([img])
        words = [w for pg in res.pages for b in pg.blocks for ln in b.lines for w in ln.words]
        confs = [w.confidence for w in words]
        return [w.value for w in words], (sum(confs) / len(confs) if confs else 0.0)
    return run


def main(root: Path):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"
    CACHE.mkdir(parents=True, exist_ok=True)
    cands = list(csv.DictReader(open("reports/gold_candidates.csv", encoding="utf-8")))
    doctr = make_doctr()
    engines = {"tesseract": run_tesseract, "doctr": doctr}

    for idx, row in enumerate(cands, 1):
        path = root / row["path"]
        todo = [e for e in ("textlayer", "tesseract", "doctr") if not (CACHE / f"{idx:02d}_{e}.json").exists()]
        if not todo:
            continue
        out = {e: [] for e in todo}
        for _, img, tl_words in pages_of(path):
            if "textlayer" in out:
                out["textlayer"].append({"words": tl_words, "conf": None, "sec": 0.0})
            for name, fn in engines.items():
                if name not in out:
                    continue
                t0 = time.perf_counter()
                try:
                    words, conf = fn(img)
                    out[name].append({"words": words, "conf": conf, "sec": time.perf_counter() - t0})
                except Exception as exc:  # keep going; record the failure
                    out[name].append({"words": [], "conf": None, "sec": time.perf_counter() - t0,
                                      "error": type(exc).__name__})
        for e, pages in out.items():
            (CACHE / f"{idx:02d}_{e}.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
        n = len(next(iter(out.values())))
        print(f"done #{idx:02d} ({n} page{'s' if n != 1 else ''}) engines: {', '.join(todo)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
