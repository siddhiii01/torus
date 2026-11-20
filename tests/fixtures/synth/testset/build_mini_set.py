"""Build a tiny synthetic test set for eval layout tests (P1.1).

Creates PDFs and JSON under a caller-supplied directory (usually ``tmp_path``).
Nothing from this module is written into git.
"""

from __future__ import annotations

import json
from pathlib import Path

MINI_SET_ID = "mini"
MINI_TEMPLATE_ID = "starter_v1"

# Placeholder values only — not real people or documents.
_CASES: tuple[dict[str, object], ...] = (
    {
        "case_id": "case_dev_001",
        "split": "dev",
        "tags": {
            "year": 2024,
            "scan_type": "digital",
            "handwriting_level": "none",
        },
        "page_count_expected": 1,
        "rows": [
            {
                "person_name": "Synthetic Person A",
                "mobile_no": "9876543210",
                "address_pincode": "560001",
                "notes": "synth",
            }
        ],
    },
    {
        "case_id": "case_hidden_001",
        "split": "hidden",
        "tags": {
            "year": 2023,
            "scan_type": "scan",
            "handwriting_level": "light",
        },
        "page_count_expected": 1,
        "rows": [
            {
                "person_name": "Synthetic Person B",
                "mobile_no": "9123456780",
                "address_pincode": "110001",
                "notes": None,
            }
        ],
    },
)


def write_synth_pdf(path: Path, *, label: str = "SYNTH") -> None:
    """Write a one-page PDF with a short non-PII label (PyMuPDF)."""
    import fitz  # lazy: heavy / optional at import time for non-PDF tests

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), label, fontsize=14)
        doc.save(path)
    finally:
        doc.close()


def build_mini_testset(data_dir: Path) -> Path:
    """Create ``<data_dir>/testsets/mini/`` with one ``dev`` and one ``hidden`` case.

    Returns the ``data_dir`` path (suitable as a stand-in for ``TORUS_DATA_DIR``).
    """
    set_root = data_dir / "testsets" / MINI_SET_ID
    for spec in _CASES:
        case_id = str(spec["case_id"])
        case_dir = set_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "case_id": case_id,
            "split": spec["split"],
            "template_id": MINI_TEMPLATE_ID,
            "tags": spec["tags"],
            "page_count_expected": spec["page_count_expected"],
            "notes": "",
        }
        expected = {
            "template_id": MINI_TEMPLATE_ID,
            "case_id": case_id,
            "version": 1,
            "rows": spec["rows"],
        }
        (case_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        (case_dir / "expected.json").write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8"
        )
        # Optional gold page types for classification eval (P3.5).
        pages = {
            "case_id": case_id,
            "version": 1,
            "pages": [
                {"index": 0, "page_type": "application_form"},
                {"index": 1, "page_type": "id_card"},
            ],
        }
        (case_dir / "pages.json").write_text(
            json.dumps(pages, indent=2) + "\n", encoding="utf-8"
        )
        # Optional gold value-region boxes for location eval (P4.8).
        # Synthetic normalised boxes only — no OCR text or field values.
        locations = {
            "case_id": case_id,
            "version": 1,
            "fields": [
                {
                    "column_key": "person_name",
                    "row_index": 0,
                    "page_index": 0,
                    "bbox": {
                        "x0": 0.20,
                        "y0": 0.12,
                        "x1": 0.55,
                        "y1": 0.16,
                    },
                    "kind": "field_value",
                },
                {
                    "column_key": "mobile_no",
                    "row_index": 0,
                    "page_index": 0,
                    "bbox": {
                        "x0": 0.20,
                        "y0": 0.20,
                        "x1": 0.42,
                        "y1": 0.24,
                    },
                    "kind": "field_value",
                },
                {
                    "column_key": "address_pincode",
                    "row_index": 0,
                    "page_index": 0,
                    "bbox": {
                        "x0": 0.20,
                        "y0": 0.28,
                        "x1": 0.35,
                        "y1": 0.32,
                    },
                    "kind": "field_value",
                },
            ],
        }
        (case_dir / "locations.json").write_text(
            json.dumps(locations, indent=2) + "\n", encoding="utf-8"
        )
        write_synth_pdf(case_dir / "source.pdf", label=f"SYNTH {case_id}")

    return data_dir
