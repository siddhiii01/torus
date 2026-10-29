"""Week 1 review: run the engine and write a local side-by-side view.

The HTML and images contain document text. They go to reports/review/, which
is git-ignored. Stdout is metadata only: index, reader, word count, status.

usage: review.py <sample_root> [gold_index ...]
default indexes: 3 digital, 34 scan, 6 handwriting, 37 blank
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.engine import DocumentRead, read_document

OUT = ROOT / "reports" / "review"
DEFAULTS = (3, 34, 6, 37)


def main(sample_root: Path, indexes: list[int]) -> None:
    cands = list(csv.DictReader(open(ROOT / "reports" / "gold_candidates.csv", encoding="utf-8")))
    labels = {r["path"]: r for r in csv.DictReader(open(ROOT / "reports" / "labels.csv", encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    print(f"{'gold':>6} {'type':<20} {'page':>4} {'reader':<10} {'words':>6} {'conf':>5} {'status':<14} {'sec':>6}")
    for idx in indexes:
        row = cands[idx - 1]
        kind = labels.get(row["path"], {}).get("doc_type") or row["kind"]
        try:
            doc, images = read_document(sample_root / row["path"])
        except Exception as exc:
            print(f"#{idx:02d}   {kind:<20} FAILED {type(exc).__name__}")
            continue
        (OUT / f"{idx:02d}.json").write_text(json.dumps(doc.to_json()), encoding="utf-8")
        sections.append(_file_section(idx, kind, doc, images))
        for page in doc.pages:
            confs = [w.conf for w in page.words if w.conf is not None]
            conf = f"{sum(confs) / len(confs):4.2f}" if confs else " n/a"
            rows = sum(len(t) for t in page.tables)
            print(f"#{idx:02d}   {kind:<20} {page.index:4d} {page.reader:<10} {len(page.words):6d} {conf}  {page.status:<14} {page.seconds:6.1f}  fields {len(page.fields):3d}  table-rows {rows:3d}  {page.reason}")
    page = _html(sections)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"\nreview written to {OUT / 'index.html'}")


def _file_section(idx: int, kind: str, doc: DocumentRead, images: list[np.ndarray]) -> str:
    blocks = []
    for page, img in zip(doc.pages, images):
        name = f"{idx:02d}_p{page.index}.jpg"
        Image.fromarray(img).save(OUT / name, quality=85)
        h, w = img.shape[:2]
        rects = []
        for word in page.words:
            x0, y0, x1, y1 = word.bbox
            rects.append(
                f'<rect x="{x0 * w:.1f}" y="{y0 * h:.1f}" width="{max((x1 - x0) * w, 1):.1f}" '
                f'height="{max((y1 - y0) * h, 1):.1f}" vector-effect="non-scaling-stroke">'
                f'<title>{html.escape(word.text)}</title></rect>'
            )
        lines: list[list[str]] = []
        for word in page.words:
            if word.line >= len(lines):
                lines.append([])
            lines[word.line].append(html.escape(word.text))
        ordered = "".join(f"<li>{' '.join(line)}</li>" for line in lines)
        field_rows = "".join(
            f"<tr><td>{html.escape(item.field)}</td><td>{html.escape(item.value)}</td></tr>"
            for item in page.fields
        )
        table_html = []
        for table in page.tables:
            if not table:
                continue
            heads = "".join(f"<th>{html.escape(col)}</th>" for col in table[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(row.get(col, ''))}</td>" for col in table[0]) + "</tr>"
                for row in table
            )
            table_html.append(f"<table><tr>{heads}</tr>{body}</table>")
        blocks.append(f"""
        <article>
          <h2>#{idx:02d} {html.escape(kind)} · page {page.index + 1} · {page.reader} · {page.status} {html.escape(page.reason)}</h2>
          <div class="page">
            <div class="stage">
              <img src="{name}" alt="page {page.index + 1}">
              <svg viewBox="0 0 {w} {h}">{''.join(rects)}</svg>
            </div>
            <ol>{ordered}</ol>
            <div class="fields">
              <h3>Fields</h3>
              <table><tr><th>Field</th><th>Value</th></tr>{field_rows}</table>
              <h3>Tables</h3>
              {''.join(table_html)}
            </div>
          </div>
        </article>""")
    return "\n".join(blocks)


def _html(sections: list[str]) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR review</title>
<style>
  body {{ margin: 24px; font: 15px/1.4 sans-serif; background: #1e1e1e; color: #e8e8e8; }}
  h1 {{ font-size: 20px; font-weight: 600; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 28px 0 8px; }}
  .page {{ display: flex; gap: 20px; align-items: flex-start; }}
  .stage {{ position: relative; flex: 0 0 auto; }}
  .stage img {{ display: block; width: 640px; height: auto; }}
  .stage svg {{ position: absolute; left: 0; top: 0; width: 640px; height: 100%; }}
  rect {{ fill: transparent; stroke: #4c8dff; stroke-width: 2; }}
  ol {{ margin: 0; padding-left: 20px; max-height: 85vh; overflow: auto; min-width: 240px; }}
  li {{ margin-bottom: 4px; }}
  .fields {{ max-height: 85vh; overflow: auto; min-width: 320px; }}
  .fields table {{ border-collapse: collapse; margin-bottom: 16px; }}
  .fields td, .fields th {{ border: 1px solid #444; padding: 4px 8px; text-align: left; vertical-align: top; }}
</style>
</head>
<body>
<h1>Week 1 review</h1>
<p>Boxes are the words. The middle list is the reading order. The right side is field = value, plus any table. This file stays in reports/ and is not committed.</p>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    root = Path(sys.argv[1])
    indexes = [int(a) for a in sys.argv[2:]] or list(DEFAULTS)
    main(root, indexes)
