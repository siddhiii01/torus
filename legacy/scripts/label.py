"""Goal 3: hand-labelling tool for the gold set (ground truth).

Run it yourself in a terminal:
    .venv\\Scripts\\python scripts\\label.py E:\\final-recon-sets

For each unlabeled file in reports/gold_candidates.csv it opens the document in
your default viewer and asks for the true values. Progress is saved after every
file to reports/labels.csv, so you can quit (q) and resume later.

Per field:  type the value | Enter = leave empty | - = not present in document
Per file:   s = skip for now, q = save and quit
"""
import csv
import os
import sys
from pathlib import Path

REPORTS = Path("reports")
CANDIDATES = REPORTS / "gold_candidates.csv"
LABELS = REPORTS / "labels.csv"

FIELDS = [
    ("doc_type", "Document type (invoice / grv / po / packing_list / cargo / other)"),
    ("vendor", "Vendor / issuer name"),
    ("doc_number", "Document number (invoice no., GRV no., ...)"),
    ("doc_date", "Document date (YYYY-MM-DD)"),
    ("currency", "Currency (e.g. USD, AED, INR)"),
    ("total_amount", "Total amount (digits only, e.g. 12345.67)"),
    ("notes", "Notes (handwriting, stamps, rotated, poor quality, ...)"),
]
COLUMNS = ["path", "labeled"] + [name for name, _ in FIELDS]


def load_labels() -> dict:
    if not LABELS.exists():
        return {}
    with open(LABELS, encoding="utf-8") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def save_labels(rows: list):
    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(root: Path):
    with open(CANDIDATES, encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    existing = load_labels()
    rows = [existing.get(c["path"], {"path": c["path"], "labeled": "no"}) for c in candidates]
    for r in rows:
        for col in COLUMNS:
            r.setdefault(col, "")
    save_labels(rows)  # creates the template on first run

    todo = [r for r in rows if r["labeled"] != "yes"]
    print(f"{len(rows) - len(todo)} of {len(rows)} files labeled, {len(todo)} to go.\n")

    for i, row in enumerate(todo, 1):
        print(f"--- [{i}/{len(todo)}] {row['path']}")
        try:
            os.startfile(root / row["path"])  # Windows: default viewer
        except OSError as exc:
            print(f"(could not open the file: {exc})")
        choice = input("Press Enter to label, s to skip, q to quit: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        for name, prompt in FIELDS:
            row[name] = input(f"  {prompt}: ").strip()
        row["labeled"] = "yes"
        save_labels(rows)
        print("  saved.\n")

    save_labels(rows)
    done = sum(1 for r in rows if r["labeled"] == "yes")
    print(f"\n{done} of {len(rows)} files labeled. Saved to {LABELS}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
