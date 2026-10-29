"""Synthetic check: a made-up PDF, no customer files. Prints pass or fail only."""

import sys
import tempfile
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.engine import read_document


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Alpha line of this synthetic page")
        page.insert_text((72, 120), "Beta line sits below it")
        doc.save(path)
        doc.close()
        read, images = read_document(path)
    page = read.pages[0]
    texts = [w.text for w in page.words]
    ok = (
        page.reader == "textlayer"
        and page.status == "ok"
        and texts[:1] == ["Alpha"]
        and "Beta" in texts
        and all(0 <= v <= 1 for w in page.words for v in w.bbox)
        and page.words[0].bbox[1] < page.words[-1].bbox[1]
        and len(images) == 1
    )
    print("pass" if ok else "fail", page.reader, page.status, len(page.words))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
