# Schema notes

## What this schema is (and is not)

`extraction.schema.json` is a **scoring** schema for the model bake-off
(`eval/promptfoo/asserts/schema_pass.js` in the Celeris Local AI Server repo). It is
**not** enforced by Flowgenix at runtime for this scene:

- The middleware profile `profiles/flowgenix.json` has `"schema": null`, so the Celeris
  server does **no** output validation for Flowgenix
  (`middleware/src/validate.js` — `if (!schema) return { ok: true }`).
- On the Flowgenix side, the standard single-prompt extraction path
  (`backend/queues/agenda.js:1987`) calls `LLMService.generateResponse(fullPrompt, callOptions)`
  where `callOptions` (`agenda.js:1854-1863`) **does not include `schema`**. The
  `openaiAdapter` only runs Ajv when `options.schema` is truthy
  (`backend/services/adapters/openaiAdapter.js:166`), so for this scene it never does.
- What the adapter *does* enforce is **syntactic JSON** only: `parseLlmJsonResponse`
  (`backend/utils/llmJsonParser.js:6`) → `validateAndFixJSON`
  (`backend/utils/jsonFixer.js:141`). Failure there → error code `INVALID_JSON_FORMAT`
  (`openaiAdapter.js:158`). See `../error_contract.md`.
- Structural / field-level conformance is applied **later and separately**, at the
  Agenda `"stage: validate"` step, by the canonical normalizers
  (`backend/canonical/documents/vendorInvoice.js`, `.../purchaseOrder.js`) and
  `backend/services/unifiedValidationEngine.js` — which are deliberately **tolerant**
  (fuzzy key match, many alias candidates, wrapper unwrapping — see
  `backend/canonical/canonicalResolver.js`, `backend/utils/normalizeExtractionEnvelope.js`).

So: use this schema to score "did the model return the right shape", not to reproduce a
runtime gate. This is the "SAP readiness schema unless that IS this scene" caveat from
the brief — it is **not** the SAP readiness schema
(`backend/ai/schema/readiness.schema.json`); that belongs to `/api/sap/analyze`, a
different scene.

## The two shapes

The scene has **two** prompts (super-admin-authored), one per document type, with
**different** output contracts:

| doc_type | prompt | `$defs` entry |
| --- | --- | --- |
| `invoice` (Flowgenix docType "Invoice") | `../prompt/flowgenix_invoice_prompt.txt` | `invoiceExtraction` |
| `purchase_order` (Flowgenix docType "Purchase Order") | `../prompt/flowgenix_po_prompt.txt` | `poExtraction` |

Pick the `$def` per `cases/<name>/meta.json.doc_type`. A `NO_DATA` /
wrong-document-type response validates against `noDataEnvelope`.

## Pilot document types

The MVP pilot covers **invoice** and **purchase order** only. GRN, Credit Note, SAP
Readiness, and the analysis-template types exist in the codebase
(`backend/services/documentClassificationService.js:47` `DOC_TYPE_READ_ALIASES`) but are
**out of scope** for this bundle. Sample GRN / credit-note PDFs were present in the
source material and were not converted into cases.

## Required vs optional

- **Invoice prompt:** *every* field is optional — the rule is "If a field is missing,
  return null." There is no field the model is required to produce a non-null value for.
  For scoring, treat `invoice_number`, `invoice_date`, `total_amount`, `currency`, and a
  non-empty `line_items` as the **key fields** (these are what the downstream canonical
  invoice normalizer treats as required — `backend/canonical/documents/vendorInvoice.js:111-142`:
  `invoiceNumber` required, `invoiceDate` required, ≥1 line item required, each line
  `quantity` required and `productCode || description` required). `meta.json.required_fields`
  per case lists the ones verifiable from that document.
- **PO prompt:** same — all nullable, `line_items` defaults to `[]`. Downstream PO
  normalizer required set (`backend/canonical/documents/purchaseOrder.js:118-149`):
  `documentNumber`, `documentDate`, ≥1 line item, each line `quantity` +
  (`productCode || description`).

## Conditional / either-or fields

- **GST split is conditional on place of supply.** Intra-state (seller state == buyer
  state) → `cgst_amount` + `sgst_amount` populated, `igst_amount` null. Inter-state →
  `igst_amount` populated, `cgst`/`sgst` null. Goldens follow whatever the document
  prints; the schema does not encode the exclusivity because malformed source documents
  exist (see case `invoice-02`).
- **PO GST Summary:** the prompt says extract the single populated HSN row into the
  `gst_*` fields; if the source genuinely has >1 populated HSN row, extract the first
  and dump the rest as raw text into `gst_summary_extra_rows_note` (string), not into
  more `gst_*` fields.
- **`taxable_amount` vs `subtotal`:** many Indian tax invoices print one "Taxable
  Amount" and no separate "Subtotal". Goldens set whichever label the document uses and
  leave the other null (do not copy — the PO prompt states this explicitly; applied to
  invoices too for consistency).
- **`seller_pan` / `buyer_pan`:** in India the PAN is the middle 10 characters of the
  GSTIN. The prompt says "extract only what is explicitly present" — if the document
  prints only a GSTIN and no standalone "PAN" label, `*_pan` stays null (goldens do
  **not** derive PAN from GSTIN).

## Enum-ish values

- `document_type` / `po_type` — free text as printed ("TAX INVOICE", "Invoice",
  "Purchase Order", "Standard PO", ...). Not constrained.
- `currency` — as shown: `INR`, `₹`, `USD`, `$`, `EUR`, `€`, ... The prompt says extract
  the symbol/code exactly as shown, so a document that prints `$` yields `"$"`, not
  `"USD"` (see cases `invoice-05`, `po-04`).
- `freight_terms` (PO) — free text ("NA - Not Applicable", "EXW - Ex-Warehouse", ...).

## Types: amounts and dates

- **Amounts** — no currency symbol. The **PO** prompt is strict: "plain numbers with no
  currency symbol or thousands separators" → JSON numbers (`poAmount` = `number|null`).
  The **invoice** prompt only says "Amounts should not include currency symbols" and
  "preserve … numbers … whenever possible" — it does **not** forbid a string or
  thousands separators, so `amount` = `number|string|null`. Goldens for invoices use
  **numeric strings without thousands separators and with the printed decimal places**
  (e.g. `"12331.00"`), because that is the most defensible reading of "preserve the
  exact … numbers"; a model that returns the JSON number `12331` should still be scored
  correct — apply the numeric tolerance in `../normalization.md`.
- **Dates** — **as printed**, verbatim string. Both prompts say do not reformat / do not
  convert to ISO. `"04-Jul-2026"` stays `"04-Jul-2026"`; `"17/06/2026"` stays
  `"17/06/2026"`. (Flowgenix's own canonical layer later parses these to ISO —
  `backend/canonical/canonicalResolver.js:574` `parseDateValueDetailed`, formats listed
  at `canonicalResolver.js:46` `SUPPORTED_DATE_FORMATS` — but that is downstream of the
  model and out of this schema's scope. See `../normalization.md`.)
- **`quantity`** — number where the source is clean; string allowed for units-bearing
  quantities the model keeps together (`"12,000.00"` in PO sources). Goldens use plain
  numbers for PO (prompt rule) and the printed form for invoices.

## `confidence`

The `openaiAdapter` wrapper appends "CRITICAL: You MUST include a `confidence` key at
the root" (`backend/services/adapters/openaiAdapter.js:89`). The schema requires it
(`number`, 0–1). Each `cases/<name>/expected.json` carries a representative
`confidence` so the file validates, but the **value is not match-scored** — only its
presence and range. Put `confidence` in a bake-off's presence check, never in the
exact-match set.

## `additionalProperties: true`

Kept open on every object. Real models nest (`{"invoice": {...}}`, `{"fields": {...}}`),
alias keys (`grand_total` for `total_amount`), and add commentary keys. Flowgenix
absorbs this downstream (`normalizeExtractionEnvelope.js` unwraps a single known
wrapper; `canonicalResolver.js` fuzzy-matches keys). The bake-off's field-match assert
(`asserts/field_match.js`) should resolve expected keys tolerantly the same way — see
`../normalization.md` §strings/keys.
