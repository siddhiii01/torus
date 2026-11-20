"""Run the engine on every gold file. Metadata on stdout. JSON stays local.

usage: run_gold.py <sample_root>
Writes reports/gold_run/<NN>.json and reports/gold_run/summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.engine import read_document

OUT = ROOT / "reports" / "gold_run"


def main(sample_root: Path) -> None:
    cands = list(csv.DictReader(open(ROOT / "reports" / "gold_candidates.csv", encoding="utf-8")))
    labels = {r["path"]: r for r in csv.DictReader(open(ROOT / "reports" / "labels.csv", encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'gold':>6} {'type':<20} {'pages':>5} {'words':>6} {'fields':>6} {'rows':>5} {'status'}")
    for idx, cand in enumerate(cands, 1):
        kind = labels.get(cand["path"], {}).get("doc_type") or cand["kind"]
        out = OUT / f"{idx:02d}.json"
        if out.exists():
            doc = json.loads(out.read_text(encoding="utf-8"))
            pages = doc["pages"]
            skipped = True
        else:
            skipped = False
            try:
                read, _ = read_document(sample_root / cand["path"])
                payload = read.to_json()
                out.write_text(json.dumps(payload), encoding="utf-8")
                pages = payload["pages"]
            except Exception as exc:
                print(f"#{idx:02d}   {kind:<20} FAILED {type(exc).__name__}")
                rows.append({
                    "idx": idx, "kind": cand["kind"], "doc_type": kind, "pages": 0,
                    "words": 0, "fields": 0, "table_rows": 0, "status": type(exc).__name__,
                    "readers": "", "needs_review": 0,
                })
                continue
        words = sum(len(p["words"]) for p in pages)
        fields = sum(len(p["fields"]) for p in pages)
        table_rows = sum(len(t) for p in pages for t in p["tables"])
        review = sum(1 for p in pages if p["status"] != "ok")
        readers = ",".join(sorted({p["reader"] for p in pages}))
        status = "needs_review" if review else "ok"
        mark = " (cached)" if skipped else ""
        print(f"#{idx:02d}   {kind:<20} {len(pages):5d} {words:6d} {fields:6d} {table_rows:5d} {status}{mark}")
        rows.append({
            "idx": idx, "kind": cand["kind"], "doc_type": kind, "pages": len(pages),
            "words": words, "fields": fields, "table_rows": table_rows, "status": status,
            "readers": readers, "needs_review": review,
        })
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nfiles {len(rows)}  ok {ok}  other {len(rows) - ok}")
    print(f"summary {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
