"""Goal 1: read-only inventory of a document folder.

Records metadata only: file type, size, and for PDFs the page count and how
many characters each page's embedded text layer holds. It never prints or
saves document text.
"""
import csv
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf

# A page with fewer characters than this in its text layer is treated as scanned.
MIN_TEXT_CHARS = 25


def classify_pdf(path: Path):
    """Return (pages, digital_pages, scanned_pages, error). Counts only."""
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                return 0, 0, 0, "password-protected"
            digital = scanned = 0
            for page in doc:
                n_chars = len(page.get_text("text").strip())
                if n_chars >= MIN_TEXT_CHARS:
                    digital += 1
                else:
                    scanned += 1
            return len(doc), digital, scanned, ""
    except Exception as exc:  # corrupt files must not stop the run
        return 0, 0, 0, type(exc).__name__


def main(root: Path, out_csv: Path):
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        row = {
            "path": str(path.relative_to(root)),
            "ext": ext,
            "size_kb": round(path.stat().st_size / 1024, 1),
            "pages": "", "digital_pages": "", "scanned_pages": "",
            "zip_entries": "", "error": "",
        }
        if ext == ".pdf":
            pages, digital, scanned, err = classify_pdf(path)
            row.update(pages=pages, digital_pages=digital,
                       scanned_pages=scanned, error=err)
        elif ext == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:  # list names only, no extraction
                    row["zip_entries"] = len(zf.namelist())
            except Exception as exc:
                row["error"] = type(exc).__name__
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"Files: {len(rows)}")
    for ext, n in Counter(r["ext"] for r in rows).most_common():
        print(f"  {ext or '(none)':8} {n}")
    pdfs = [r for r in rows if r["ext"] == ".pdf" and not r["error"]]
    d = sum(r["digital_pages"] for r in pdfs)
    s = sum(r["scanned_pages"] for r in pdfs)
    print(f"\nPDF pages: {d + s}  digital: {d}  scanned/image-only: {s}")
    mixed = sum(1 for r in pdfs if r["digital_pages"] and r["scanned_pages"])
    print(f"PDFs fully digital: {sum(1 for r in pdfs if not r['scanned_pages'])}, "
          f"fully scanned: {sum(1 for r in pdfs if not r['digital_pages'])}, "
          f"mixed: {mixed}")
    errs = [r for r in rows if r["error"]]
    print(f"Files with errors: {len(errs)}")
    print(f"\nCSV written to {out_csv}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
