# DEB golding checklist

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
