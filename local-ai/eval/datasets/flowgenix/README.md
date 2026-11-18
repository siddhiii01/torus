# datasets/flowgenix/ (v1)

Standard-single-prompt extraction bake-off for the **fictional eval pin**
(`pack.json`, `source: fictional-eval-pin`). Envelope gold: `{ fields, lineItems }`.

The previous 15-case flat-key suite lives at `../flowgenix_v0_flat/` (historical only).

## Flags (pinned)

- `ENABLE_LEGACY_EXTRACTION`: false
- `fieldConfidenceEnabled`: false
- Field plan: `dynamic-configured` (`buildDynamicTargetFieldsSchema`)
- Path: standard-single-prompt only
- Prompt assembly: A + B + C + D(PO only) + F + `DOCUMENT CONTENT:` — **no** Gemini/OpenAI adapter wrap

## Layout

| Path | Role |
| --- | --- |
| `pack.json` | Pinned Prompt docs (B + catalogs) |
| `constants/` | A, C_po, C_inv, D, F — frozen join strings |
| `cases/<name>/ocr.txt` | Synthetic OCR |
| `cases/<name>/gold.json` | Envelope gold |
| `cases/<name>/meta.json` | `docType`, `required_fields`, `source` |
| `assemble.js` | Writes `assembled_prompt.txt` + `golden.json` |
| `schema.json` | Envelope + `NO_DATA` scoring schema |

```
node eval/datasets/flowgenix/assemble.js
powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1 -View
```

Until a live Mongo Prompt Setup replaces B, keep `prompt_source: synthetic_assembly` in meta.

## Scoring notes

- Dates: printed (`04-Jul-2026`, `02/09/2026`) equals ISO (`2026-07-04` / `2026-09-02`).
  Ambiguous `DD/MM` vs `MM/DD` (both parts ≤12): either reading is accepted if it matches the other side.
- Latency: hang guard 20 min; Ollama request timeout 15 min; **maxConcurrency: 1**.
  Production Celeris cap remains 115s — record p50/p95 separately.

## Seeds (synthetic)

| Case | Type |
| --- | --- |
| `po-01` | Clean two-line PO |
| `invoice-01` | Clean two-line invoice (`lineItems` array) |
| `invoice-02` | Messy OCR (split table) |
| `invoice-03` | Adversarial: PO OCR + Invoice prompt |

## Real POs (`po-real-01` … `po-real-10`)

Imported from Flowgenix `backend/scratch/po-real-eval/` (scrubbed Document AI OCR).
Pin pack B+C; gold is human envelope. `po-real-09` sets `lineItemsGoldScope: page1_subset`
(scorer matches gold rows as a subset; extra model rows OK). `po-real-10` is adversarial.
