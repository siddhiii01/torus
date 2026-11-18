# DEB OCR failure classes (diagnostic)

Recorded from baseline docTR on Binani application form pages 1–2 (2026-11-17). Metadata only — no document text.

| Class | Signal | Impact on DEB extract |
|-------|--------|------------------------|
| Comb / character boxes | Many 1-char tokens; surname/city often missing as whole words | Name/address assembly brittle |
| Stamps / barcodes over fields | Extra tokens; some field glyphs partially covered | KYC / ID / date noise |
| Handwriting in grids | Digit sequences (DOB, phone, PIN) incomplete in OCR dump | Structured date/phone misses |
| Cover / non-form pages | (expected on full packs) | Need page routing before extract |
| Scan noise / skew | Lower conf on dense sections | `needs_review` risk on worse pages |

Baseline note: Binani p1–2 both `reader=doctr`, `status=ok`, ~525–649 words, mean conf ~0.81–0.83. Usable for bake-off inputs, not for presence-in-OCR scoring.

## Bake-off notes (gold v0 frozen)

- **comb_merge postprocess** (join 1-char tokens): singles 7.4%→6.0%, **ocr recoverability unchanged (0.103)**. Comb merge alone is not the fix — most scored gold tokens (name, address, DOB, phones) are simply absent/wrong in the OCR dump, not merely split.
- **DPI / CLAHE** (Binani p1–2): `dpi300_none` best so far (**ocrR 0.103 → 0.154**). CLAHE did not help. Extract row/field F1 still 0 with stub.
- **layout extract v1** on `binani_dpi300_none` (`heuristic_v1_layout`): **row F1 0 → 1.000** (2/2 aligned H1+W); **fields micro F1 0 → 0.632** (P 1.000 / R 0.462). Wins: person name (card + company corroboration), gender, LINE1 both rows, W LINE2 `NS Road`. Still missing from OCR: DOB, phones/STD, city/PIN/state, email (glued/wrong), H1 street/locality, KYC/business (other pages).
- **digit_crop v1** on same cache (`heuristic_v1_layout+digit_crop`, no full-page re-OCR): **fields micro F1 0.632 → 0.655** (P **1.000** / R 0.462 → 0.487). Keep. Recovered **H1 `address_pincode`** via label→comb crop + black-ink sep + Tesseract jitter majority. Still FN: DOB, W PIN, STD/landline (crops locate; readers disagree / miss — escalate crop reader, not full-page OCR). Helper: `ocr/digit_crop.py`; DEB locate/apply stays in `scripts/deb_extract.py`. Bake-off: `scripts/deb_bakeoff_digit_crop.py`.
- **crop_vlm escalate (hybrid)** — same locators; TrOCR → Florence only on empty/disagree digit crops (+ email/city/state field crops). Schema: `schemas/deb/crop_escalate.json` (scored Excel keys only). Gates: existing validators, prior-vote echo when specialized readers saw digits, **fill empties only**. Helper: `ocr/crop_vlm.py`. CLI: `--digit-crop --crop-vlm`. Bake-off third arm: `--crop-vlm`. Keep only if micro-F1 rises **without** precision drop. Not a layout replacement; printed name/gender/LINE1 stay deterministic. Qwen2-VL stays last-resort / bigger machine per `PLAN.md`.
- **PIN → city/state + blank-ink gate (v2)** — once a valid 6-digit PIN is recovered, fill `address_city` / `address_state` from `schemas/deb/pin_master.json` (`ocr/pin_master.py`) and **skip** those TrOCR crops. Free-text VLM runs only when PIN locality miss: `black_ink_mask` density gate first, then closed state list / PIN-scoped city vocab (exact/near-exact — not wide fuzzy). Dual-reader consensus stays deprioritized. Policy matches `crop_escalate.json` v2.
- **Bake-off re-score (binani, dpi300_none, after v2):** digit_crop **P 1.000 / R 0.486 / F1 0.654** (was ~0.600) — **keep**. `pin_master:exact` filled H1 city+state from recovered PIN; W PIN still miss. crop_vlm **P 0.944 / R 0.486 / F1 0.641** — city/state TrOCR no longer fires (only `h1_std` escalated), but STD escalate still drops precision → **do not keep** vs digit_crop. Next: tighten digit escalate (STD) or leave crop_vlm off for city/state-only wins. Section crops still optional.

## Routing (failure class → path)

| Keep deterministic | Escalate to crop VLM |
|---|---|
| Person name, gender, printed LINE1 / street cues | DOB, PIN, STD, landline, mobile when digit_crop empty or no agreement |
| Section headers → H1 / W row assembly | Email when OCR empty/glued **and** ink-density gate passes |
| City/state via PIN master when PIN valid | City/state VLM only if PIN miss + ink + strict vocab |
| Page routing (cover vs form vs KYC) | KYC IDs on supporting pages after page classify (later) |
