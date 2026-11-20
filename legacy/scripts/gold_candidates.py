"""Goal 2: dedupe the sample folder and propose a gold-set candidate list.

Inputs : reports/inventory.csv (from inventory.py) and the sample folder.
Outputs: reports/duplicates.csv and reports/gold_candidates.csv

Only file names, hashes and counts are written. No document content is read
beyond hashing the raw bytes.

Selection rules (deterministic):
  1. Unique files only (identical content counts once).
  2. Every unique scanned/mixed PDF and image is included: these are the hard
     cases and the engine matters most there.
  3. Digital PDFs fill the remaining slots, round-robin across case folders
     so no single vendor dominates, preferring different page counts.
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

TARGET_FILES = 40
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kind(row) -> str:
    if row["ext"] in IMAGE_EXTS:
        return "image"
    if row["ext"] != ".pdf":
        return "other"
    if int(row["scanned_pages"] or 0) == 0:
        return "digital"
    return "scanned" if int(row["digital_pages"] or 0) == 0 else "mixed"


def main(root: Path, inventory: Path, out_dir: Path):
    with open(inventory, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if r["ext"] == ".pdf" or r["ext"] in IMAGE_EXTS]

    # 1. Dedupe by content hash. The first path (sorted) is kept as canonical.
    groups = defaultdict(list)
    for r in rows:
        r["sha256"] = sha256(root / r["path"])
        r["kind"] = kind(r)
        r["case"] = Path(r["path"]).parts[0]
        groups[r["sha256"]].append(r)
    unique = [g[0] for g in groups.values()]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "copies", "paths"])
        for h, g in groups.items():
            if len(g) > 1:
                w.writerow([h[:12], len(g), " | ".join(x["path"] for x in g)])

    # 2. Hard cases first.
    chosen = [(r, "hard-case: " + r["kind"]) for r in unique
              if r["kind"] in ("scanned", "mixed", "image")]

    # 3. Fill with digital PDFs, round-robin across cases.
    by_case = defaultdict(list)
    for r in sorted((r for r in unique if r["kind"] == "digital"),
                    key=lambda r: (int(r["pages"]), r["path"])):
        by_case[r["case"]].append(r)
    cases = sorted(by_case)
    slot = max(0, TARGET_FILES - len(chosen))
    i = 0
    while slot > 0 and any(by_case.values()):
        case = cases[i % len(cases)]
        i += 1
        pool = by_case[case]
        if not pool:
            continue
        # Alternate between the longest and shortest remaining file to vary page counts.
        r = pool.pop(-1 if (i // len(cases)) % 2 == 0 else 0)
        chosen.append((r, "coverage: digital"))
        slot -= 1

    with open(out_dir / "gold_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "kind", "pages", "case", "reason", "sha256_12"])
        for r, why in sorted(chosen, key=lambda x: x[0]["path"]):
            w.writerow([r["path"], r["kind"], r["pages"], r["case"], why, r["sha256"][:12]])

    # Summary
    dup_files = sum(len(g) - 1 for g in groups.values())
    print(f"PDFs and images scanned: {len(rows)}")
    print(f"Unique by content: {len(unique)}  (duplicate copies: {dup_files})")
    print(f"Unique by kind: " + ", ".join(
        f"{k}={sum(1 for r in unique if r['kind'] == k)}"
        for k in ("digital", "scanned", "mixed", "image")))
    pages = sum(int(r["pages"] or 1) for r, _ in chosen)
    print(f"\nGold candidates: {len(chosen)} files, about {pages} pages")
    for k in ("digital", "scanned", "mixed", "image"):
        print(f"  {k:8} {sum(1 for r, _ in chosen if r['kind'] == k)}")
    print(f"  cases covered: {len({r['case'] for r, _ in chosen})}")
    print(f"\nWrote {out_dir / 'gold_candidates.csv'} and {out_dir / 'duplicates.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
