"""One-shot bootstrap for DEB schemas + gitignored cases. Not part of runtime."""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
ATT = Path(
    r"c:\Users\saraj\.cursor\projects\c-Users-saraj-Desktop-Taurus"
    r"\attachments\e270d26c-10e2-429e-9080-cefdd498f8c9"
)
DOWNLOADS = Path(r"c:\Users\saraj\Downloads")
XLSX = DOWNLOADS / "DEB File 60 Cases.xlsx"

# Stable keys in Excel column order (73). Col 5 duplicates Bank Lan header.
COLUMNS = [
    ("bank_lan", "Bank Lan", 1),
    ("system_lan", "SYSTEM_LAN", 2),
    ("trust_name", "TRUST_NAME", 3),
    ("existing_lan_no", "EXISTING_LAN_No", 4),
    ("bank_lan_dup", "Bank Lan", 5),
    ("customer_name", "CUSTOMER_NAME", 6),
    ("person_name", "Name (Borrwer,Co-applicant,Gurantor)", 7),
    ("id", "ID", 8),
    (
        "constitution",
        "Constitution ( Individual,Prop Firm, Partnership Firm, LLP,Corporate",
        9,
    ),
    (
        "customer_type",
        "Customer Type ( Coapplicant/Borrower /Partner/Promoter  Guarantor/ Co Gur",
        10,
    ),
    ("tcmbr_code", "TCMBR CODE", 11),
    ("gender", "Gender", 12),
    ("address_line1", "Address_LINE1", 13),
    ("address_line2", "Address_LINE2", 14),
    ("address_line3", "Address_LINE3", 15),
    ("address_state", "Address_STATE", 16),
    ("address_city", "Address_City", 17),
    ("address_pincode", "Address_PINCode", 18),
    ("address_type", "Type Of Address (Residence/Office/Workplace)", 19),
    ("address_source", "Address source", 20),
    ("mobile_no", "Mobile No", 21),
    ("mobile_no_1", "Mobile No 1", 22),
    ("mobile_no_2", "Mobile No 2", 23),
    ("mobile_no_3", "Mobile No 3", 24),
    ("mobile_no_4", "Mobile No 4", 25),
    ("random_contact_no", "Randam Contact No", 26),
    ("email_id", "Email Id", 27),
    ("std_code", "STDCode", 28),
    ("landline_no", "LandlineNo", 29),
    ("cin", "CIN ", 30),
    ("savings_account_no", "Savings Account No", 31),
    ("source_of_savings_acc_no", "Source of Savings acc No.", 32),
    ("dl_no", "DLNo", 33),
    ("source_of_dl_kyc_id", "Source of DL Kyc ID", 34),
    ("aadhaar_no", "AadhaarNo", 35),
    ("source_of_aadhaar_kyc_id", "Source of Aadhaar Kyc ID", 36),
    ("pan", "PAN", 37),
    ("source_of_pan_kyc_id", "Source of Pan Kyc ID", 38),
    ("passport", "Passport", 39),
    ("source_of_passport_kyc_id", "Source of Passport Kyc ID", 40),
    ("voter_id", "VoterId", 41),
    ("source_of_voterid_kyc_id", "Source of Voterid Kyc ID", 42),
    ("ration_card", "Ration Card", 43),
    ("source_of_ration_card_kyc_id", "Source of Ration card Kyc ID", 44),
    ("dob", "DOB-Dateofincoprotion", 45),
    ("remark", "Remark", 46),
    ("verification_remark", "Verification remark/ verified with available documents ", 47),
    ("sanction_date", "Sanction Date", 48),
    (
        "sanction_ticket",
        "Sanction Ticket/Sanction Advice/ Sanction Letter/Arrangement Letter",
        49,
    ),
    ("application_form", "Application Form (Loan Facility Remark)", 50),
    ("dpn", "DPN (Loan Facility Remark)", 51),
    ("agreement_of_loan", "Agreement of Loan ", 52),
    ("nature_of_facility_docs", "Nature of Facility As per Docs", 53),
    ("nature_of_facility_system", "Nature of Facility As per System", 54),
    ("checker_remark", "Checker Remark", 55),
    ("checking_date", "Checking  date", 56),
    ("updation_remarks", "Updation remarks", 57),
    ("checker_name", "Checker Name", 58),
    ("checker_status", "Checker Status", 59),
    ("roi", "ROI", 60),
    ("page_no_of_roi", "Page No of ROI", 61),
    ("page_count", "Page Count of Document", 62),
    ("security_value_fmv", "Security value of asset (FMV)", 63),
    ("security_value_distress", "Security value of asset (Distress Value)", 64),
    ("realizable_value", "Reliazable value of asset ", 65),
    ("security_value", "Security Value", 66),
    ("business_type", "Business Type", 67),
    ("business_industry_type", "Business industry type", 68),
    ("business_category", "Business category", 69),
    ("location_type", "Location type", 70),
    ("type_of_relationship", "Type of Relationship", 71),
    ("credit_type", "Credit Type", 72),
    (
        "tangible_security_type",
        "Tangible Security Type (Home / House , Land and building,"
        "agricultural land,machinary, hypo of stock,machinery)",
        73,
    ),
]

SCORED = [
    "person_name",
    "customer_name",
    "gender",
    "dob",
    # customer_type / constitution → passthrough until checkbox/role OCR exists
    "mobile_no",
    "mobile_no_1",
    "mobile_no_2",
    "mobile_no_3",
    "mobile_no_4",
    "random_contact_no",
    "email_id",
    "std_code",
    "landline_no",
    "address_line1",
    "address_line2",
    "address_line3",
    "address_city",
    "address_state",
    "address_pincode",
    "address_type",
    "address_source",
    "dl_no",
    "source_of_dl_kyc_id",
    "aadhaar_no",
    "source_of_aadhaar_kyc_id",
    "pan",
    "source_of_pan_kyc_id",
    "passport",
    "source_of_passport_kyc_id",
    "voter_id",
    "source_of_voterid_kyc_id",
    "ration_card",
    "source_of_ration_card_kyc_id",
    "savings_account_no",
    "source_of_savings_acc_no",
    "cin",
    "business_type",
    "business_industry_type",
    "business_category",
    "page_count",
    "application_form",
]
FUZZY = {"address_line1", "address_line2", "address_line3", "business_type"}
PASSTHROUGH = [k for k, _, _ in COLUMNS if k not in SCORED and k != "bank_lan_dup"]

CASES = {
    "munuswamy": {
        "pdf": "S0000010999523.pdf",
        "excel_rows": [10, 11, 12],
        "customer_name": "V Munuswamy",
    },
    "anbazhakan": {
        "pdf": "S0000010547942.pdf",
        "excel_rows": [25, 26, 27],
        "customer_name": "Basker Anbazhakan",
    },
    "binani": {
        "pdf": "S0000010456335.pdf",
        "excel_rows": [73, 74],
        "customer_name": "Sailesh Binani",
    },
}


def cell_value(v):
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        return v.strip()
    return v


def write_schemas() -> Path:
    schema_dir = ROOT / "schemas" / "deb"
    schema_dir.mkdir(parents=True, exist_ok=True)

    excel_columns = {
        "version": 1,
        "source_sheet": "56 accounts",
        "header_row": 2,
        "columns": [
            {
                "key": k,
                "excel_header": h,
                "excel_col": c,
                "duplicate_of": "bank_lan" if k == "bank_lan_dup" else None,
            }
            for k, h, c in COLUMNS
        ],
    }
    (schema_dir / "excel_columns.json").write_text(
        json.dumps(excel_columns, indent=2), encoding="utf-8"
    )

    field_groups = {
        "version": 1,
        "scored": [
            {"key": k, "match": "fuzzy" if k in FUZZY else "exact"} for k in SCORED
        ],
        "passthrough": PASSTHROUGH,
        "notes": {
            "scored": "Doc-extractable fields for bake-off ranking on aligned rows only.",
            "passthrough": "Ops/system fields not scored in phase 1. customer_type/constitution deferred until checkbox/role OCR exists.",
            "fuzzy": "Token-overlap match for long address/business strings; never OCR-substring presence.",
        },
    }
    (schema_dir / "field_groups.json").write_text(
        json.dumps(field_groups, indent=2), encoding="utf-8"
    )

    (schema_dir / "row_key.md").write_text(
        """# DEB row alignment key

Predicted rows are matched to gold rows with this key (after normalization).

## Primary key

`(person_key, address_type, address_source)`

1. **person_key** — normalize `person_name` (`Name (Borrwer,Co-applicant,Gurantor)`); if empty, fall back to `customer_name` (`CUSTOMER_NAME`).
2. **address_type** — `address_type` / Type Of Address (`H1`, `H2`, `W`, …), uppercased, whitespace stripped.
3. **address_source** — `address_source` canonicalized via known aliases (see below).

## Disambiguator

When the primary key collides within one account, also match on normalized `address_line1`.

## Unmatched rows

- Gold row with no prediction → **false negative** (hurts row recall).
- Prediction with no gold → **false positive** (hurts row precision).
- Field-level metrics run **only on aligned pairs**.

## Address-source aliases (canonical → forms)

| Canonical | Accepted forms |
|-----------|----------------|
| `applicationform` | Application form, Application Form, ApplicationForm, Applicationform |
| `voterid` | Voterid, Voter id, Voter ID, VoterId |
| `rationcard` | Rationcard, Ration card, Ration Card |
| `aadhaarcard` | Aadhaarcard, Aadhaar card, Aadhaar |
| `dl` | DL, Driving Licence, Driving License |
| `agreement_of_guarantee` | Agreement of Guarantee, Aggreement of Guarantee, Aggrement of Guarantee, Agreement Of Gurantee |
| `dpn` | DPN |
| `guarantor_form` | Guarantor's form, Guarantor form |

Unknown sources: casefold + collapse whitespace + strip non-alnum separators to underscores.
""",
        encoding="utf-8",
    )

    (schema_dir / "GOLDING.md").write_text(
        """# DEB golding checklist

Goal: validate Excel draft rows against the scanned pack, then freeze **gold v0** (3 accounts, diagnostic only — not a publishable accuracy number).

## Per case

1. Open `data/deb/cases/<slug>/source.pdf` and `gold.json`.
2. Confirm person identity (name / DOB / gender) against the application form.
3. For each gold row, confirm:
   - `address_type` (H1 / H2 / W / …)
   - `address_source` (Application form, Voterid, …)
   - address LINE1–3, city, state, PIN
   - phones / email / STD / landline on that row
   - KYC IDs + sources when the supporting page exists in the pack
4. Confirm pack-level fields that appear on the primary (usually H1 Application form) row:
   - `page_count`, `application_form`, business-type signals
5. Fix `gold.json` in place. Do not invent fields that are not visible on the pack.
6. Mark `meta.json` → `gold_status`: `draft` → `validated` → `frozen_v0`.

## Freeze rule

Gold v0 is frozen only after human review of all three cases. Bake-off ranking starts after freeze.
""",
        encoding="utf-8",
    )
    return schema_dir


def write_cases() -> None:
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["56 accounts"]

    raw_dir = ROOT / "data" / "deb" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(XLSX, raw_dir / "DEB_File_60_Cases.xlsx")

    for slug, info in CASES.items():
        case_dir = ROOT / "data" / "deb" / "cases" / slug
        (case_dir / "pages").mkdir(parents=True, exist_ok=True)
        src = ATT / info["pdf"]
        if not src.exists():
            src = DOWNLOADS / info["pdf"]
        dest = case_dir / "source.pdf"
        shutil.copy2(src, dest)

        rows = []
        for r in info["excel_rows"]:
            row = {"_excel_row": r}
            for key, _h, col in COLUMNS:
                if key == "bank_lan_dup":
                    continue
                val = cell_value(ws.cell(r, col).value)
                if val is not None:
                    row[key] = val
            rows.append(row)

        gold = {
            "case_id": slug,
            "gold_version": "draft",
            "source": {
                "excel_file": "DEB File 60 Cases.xlsx",
                "sheet": "56 accounts",
                "excel_rows": info["excel_rows"],
                "pdf": info["pdf"],
            },
            "rows": rows,
        }
        (case_dir / "gold.json").write_text(
            json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        meta = {
            "case_id": slug,
            "customer_name": info["customer_name"],
            "source_pdf": info["pdf"],
            "page_count_expected": next(
                (row.get("page_count") for row in rows if "page_count" in row), None
            ),
            "gold_status": "draft",
            "notes": "Draft gold exported from Excel; pending PDF validation.",
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"case {slug}: {len(rows)} rows, pdf {dest.stat().st_size} bytes")


def main() -> None:
    schema_dir = write_schemas()
    write_cases()
    for p in (
        ROOT / "reports" / "deb" / "ocr_cache",
        ROOT / "reports" / "deb" / "review",
        ROOT / "reports" / "deb" / "scores",
    ):
        p.mkdir(parents=True, exist_ok=True)
    print("schemas:", [p.name for p in schema_dir.iterdir()])
    print("cases:", [p.name for p in (ROOT / "data" / "deb" / "cases").iterdir()])


if __name__ == "__main__":
    main()
