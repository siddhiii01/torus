"""Run the shared OCR engine on DEB packs and write a local review HTML.

Stdout is metadata only (case, page, reader, word count, status, seconds).
HTML/images go under reports/deb/review/ (gitignored; may contain PII).

usage:
  python scripts/deb_review.py binani
  python scripts/deb_review.py binani --pages 0 1
  python scripts/deb_review.py --all
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymupdf

from ocr.engine import DocumentRead, read_document

CASES = ROOT / "data" / "deb" / "cases"
OUT = ROOT / "reports" / "deb" / "review"
CACHE = ROOT / "reports" / "deb" / "ocr_cache"


def read_selected_pages(pdf: Path, pages: list[int] | None) -> tuple[DocumentRead, list]:
    """OCR full pack, or only the requested 0-based pages via a temp subset PDF."""
    if pages is None:
        return read_document(pdf)

    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with pymupdf.open(pdf) as src:
            subset = pymupdf.open()
            for i in pages:
                subset.insert_pdf(src, from_page=i, to_page=i)
            subset.save(tmp)
            subset.close()
        doc, images = read_document(tmp)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass

    remapped = []
    for page, orig_i in zip(doc.pages, pages):
        page.index = orig_i
        remapped.append(page)
    return DocumentRead(remapped), images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case slug(s); default with --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pages", nargs="+", type=int, default=None, help="0-based page indexes")
    args = parser.parse_args()

    slugs = sorted(p.name for p in CASES.iterdir() if p.is_dir()) if args.all else args.cases
    if not slugs:
        parser.error("provide case slug(s) or --all")

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    print(f"{'case':<12} {'page':>4} {'reader':<10} {'words':>6} {'conf':>5} {'status':<14} {'sec':>6}")

    for slug in slugs:
        pdf = CASES / slug / "source.pdf"
        if not pdf.exists():
            print(f"{slug}: missing {pdf}", file=sys.stderr)
            continue
        try:
            doc, images = read_selected_pages(pdf, args.pages)
        except Exception as exc:
            print(f"{slug}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        cache_path = CACHE / f"{slug}.json"
        payload = doc.to_json()
        if args.pages is not None and cache_path.exists():
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
            by_idx = {p["index"]: p for p in existing.get("pages", [])}
            for p in payload["pages"]:
                by_idx[p["index"]] = p
            payload = {"pages": [by_idx[i] for i in sorted(by_idx)]}
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

        sections.append(_file_section(slug, doc, images))
        for page in doc.pages:
            confs = [w.conf for w in page.words if w.conf is not None]
            conf = f"{sum(confs) / len(confs):4.2f}" if confs else " n/a"
            print(
                f"{slug:<12} {page.index:4d} {page.reader:<10} {len(page.words):6d} "
                f"{conf}  {page.status:<14} {page.seconds:6.1f}  {page.reason}"
            )

    (OUT / "index.html").write_text(_html(sections), encoding="utf-8")
    print(f"\nreview written to {OUT / 'index.html'}")
    return 0


def _file_section(slug: str, doc: DocumentRead, images: list[np.ndarray]) -> str:
    blocks = []
    for page, img in zip(doc.pages, images):
        name = f"{slug}_p{page.index}.jpg"
        Image.fromarray(img).save(OUT / name, quality=85)
        h, w = img.shape[:2]
        rects = []
        for word in page.words:
            x0, y0, x1, y1 = word.bbox
            rects.append(
                f'<rect x="{x0 * w:.1f}" y="{y0 * h:.1f}" width="{max((x1 - x0) * w, 1):.1f}" '
                f'height="{max((y1 - y0) * h, 1):.1f}" vector-effect="non-scaling-stroke">'
                f"<title>{html.escape(word.text)}</title></rect>"
            )
        lines: list[list[str]] = []
        for word in page.words:
            while word.line >= len(lines):
                lines.append([])
            lines[word.line].append(html.escape(word.text))
        ordered = "".join(f"<li>{' '.join(line)}</li>" for line in lines)
        blocks.append(
            f"<section><h2>{html.escape(slug)} · page {page.index + 1} · "
            f"{html.escape(page.reader)} · {html.escape(page.status)}</h2>"
            f'<div class="row">'
            f'<div class="page"><svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<image href="{html.escape(name)}" width="{w}" height="{h}"/>'
            f'{"".join(rects)}</svg></div>'
            f'<div class="words"><ol>{ordered}</ol></div>'
            f"</div></section>"
        )
    return "\n".join(blocks)


def _html(sections: list[str]) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DEB OCR review</title>
<style>
body {{ font-family: Georgia, serif; margin: 1rem 2rem; background: #f7f4ef; color: #1a1a1a; }}
h1,h2 {{ font-weight: 600; }}
.row {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 1rem; margin-bottom: 2rem; }}
.page svg {{ width: 100%; height: auto; background: #fff; border: 1px solid #ccc; }}
.page rect {{ fill: rgba(0, 90, 180, 0.12); stroke: #005ab4; stroke-width: 1; }}
.words {{ max-height: 80vh; overflow: auto; background: #fff; border: 1px solid #ccc; padding: 0.5rem 1rem; }}
ol {{ padding-left: 1.2rem; }}
li {{ margin: 0.15rem 0; }}
@media (max-width: 900px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>DEB OCR review</h1>
{"".join(sections)}
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
