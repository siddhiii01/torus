# Normalization / equivalence rules for scoring extracted vs expected

These are the equivalence rules the bake-off field-match assert
(`eval/promptfoo/asserts/field_match.js`) and the DeepEval regression gate should apply
when comparing a model's output to `cases/<name>/expected.json`. Every rule is grounded
in code Flowgenix already runs on the extraction result **after** the LLM call, so
"equivalent under these rules" == "Flowgenix would treat these the same".

Primary sources:
- `backend/utils/llmJsonParser.js`, `backend/utils/jsonFixer.js` — JSON parse + repair
- `backend/canonical/canonicalResolver.js` — `coerceToNumber` (`:367`), `resolveAmount`
  (`:422`), `parseDateValueDetailed` (`:574`), `SUPPORTED_DATE_FORMATS` (`:46`)
- `shared/fieldResolution.cjs` — `keysMatch` (`:18`), `isEmptyScalar` (`:35`),
  `FIELD_ALIASES` (`:53`)
- `backend/utils/normalizeExtractionEnvelope.js` — wrapper unwrap + line-items key alias
- `backend/utils/numberUtils.js` — `parseCurrency` (locale-aware; see caveat below)
- Tests that pin the behaviour: `backend/tests/canonicalDateParsing.test.js`,
  `backend/tests/normalizeVendorInvoice.test.js`,
  `backend/tests/normalizePurchaseOrder.test.js`,
  `backend/tests/normalizeExtractionEnvelope.test.js`, `backend/tests/llmJsonParser.test.js`

---

## 1. Amounts

**Rule for scoring: compare numerically, not as strings.** Coerce both sides with the
Flowgenix rule and compare with a tiny epsilon.

Flowgenix's coercion (`canonicalResolver.js:367-382`, `coerceToNumber`):

1. If already a finite number → use as-is.
2. If a string → delete every `$ € £ ₹ ,` and whitespace char, then `Number(...)`.
3. If the result is finite → that number; else → `null` ("found but not numeric").

Consequences (all treated as **equal**):

| Printed / model output | Coerces to |
| --- | --- |
| `1,000.00` | `1000` |
| `1000` | `1000` |
| `1000.0` | `1000` |
| `"$1,000.00"` | `1000` |
| `"₹ 1,000"` | `1000` |
| `1000.00` (number) | `1000` |
| `"12,331.00"` | `12331` |

Scoring:
- Numeric epsilon: `abs(a - b) <= 0.01` → equal (covers `12331` vs `12331.00` vs
  `"12,331.00"`).
- **Rounding:** Flowgenix does **not** round amounts during coercion. Do not accept
  `1881` vs `1880.98` as equal — only true representational differences.
- **Currency symbol:** never part of an amount field. Both prompts forbid it in amount
  values; `coerceToNumber` strips `$ € £ ₹` anyway. The currency itself lives only in
  `currency` (see §3).
- **Sign:** negative amounts (deposits, discounts, credits) keep their sign —
  `("$10,000.00")` in a "Deposit Received (50%)" line is `-10000` if the document
  prints it parenthesised/negative. Goldens follow the printed sign.

### Caveat — European decimal comma is NOT handled on this path

`coerceToNumber` strips commas but **not** dots, so `"1.234,56"` → `Number("1.234.56")`
→ `NaN` → `null`. A separate helper `numberUtils.parseCurrency`
(`backend/utils/numberUtils.js:5`) *does* infer locale (`1.234,56` → `1234.56`), but it
is **not** in the extraction→canonical path. All bundle cases use `USD`/`INR` documents
with US-style grouping (`1,234.56`), so this does not bite here — but a bake-off that
adds `EUR` documents with `1.234,56` formatting must expect Flowgenix to null those
amounts. Flag, don't "fix".

---

## 2. Dates

**Two different rules, at two different stages:**

### 2a. At the model boundary (what `expected.json` stores)

Both prompts: **"Dates should be extracted exactly as they appear"** / "do not reformat,
do not convert to ISO". So `expected.json` stores the **verbatim printed string**:
`"04-Jul-2026"`, `"17/06/2026"`, `"August 14, 2026"`, `"13-Jul-2026"`.

Scoring a date field:
- Exact string match after **trim** and internal-whitespace collapse (§4).
- Also accept a match if **both sides parse (via §2b) to the same ISO date** — this
  lets a model that lightly reformats (`"04 Jul 2026"` for `"04-Jul-2026"`) still score
  correct. Do **not** accept a value that fails to parse.

### 2b. Downstream (Flowgenix's own canonical parse — informational)

After extraction, `canonicalResolver.parseDateValueDetailed` (`:574`) converts the
printed date to ISO `YYYY-MM-DD`. Accepted input formats
(`SUPPORTED_DATE_FORMATS`, `:46`):

```
YYYY-MM-DD        YYYY/MM/DD        YYYY.MM.DD
DD-MM-YYYY        DD.MM.YYYY                       <- dash & dot = DAY-first
D-MMM-YYYY   (11-Jul-2026)     D-MMM-YY  (25-Feb-25)
Month D, YYYY  (July 11, 2026)     D Month YYYY  (11 July 2026)
M/D/YYYY     (7/11/2026)                          <- slash = MONTH-first (US)
```

Component-order rules (`canonicalResolver.js:553-570`, pinned by
`canonicalDateParsing.test.js`):
- **Year-first** (`YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD`) — unambiguous.
- **Dash / dot** numeric (`11-07-2026`, `11.07.2026`) — **day-first**. `05-03-2026` →
  `2026-03-05`.
- **Slash** numeric (`7/11/2026`) — **month-first**. `11/07/2026` → `2026-11-07` **and
  flagged `ambiguous: true`** (both readings valid) — surfaced as a warning, not an
  error.
- If the primary order is an impossible calendar date but the swapped order is valid
  (`23/07/2026`), the swapped reading is used silently.
- **2-digit year** (`25-Feb-25`) — POSIX pivot: `00–68` → `20xx`, `69–99` → `19xx`
  (`resolveTwoDigitYear`, `:85`).
- Unparseable → canonical value `null`, `reason: "found but not a parseable date"`,
  `attemptedFormats` attached. For a **required** date (`invoice_date` / PO
  `purchase_order_date`) this becomes a validation error
  (`vendorInvoice.js:120`, `purchaseOrder.js:127`).

**Implication for model choice:** a model that emits `"2026-13-07"` or a localized month
name outside the list will make Flowgenix null the date and fail the document. Score
date fidelity strictly.

---

## 3. `currency`

- Extracted **exactly as shown**: a document printing `$` → `"$"`; printing `USD` →
  `"USD"`; printing `₹` → `"₹"`; printing `INR` → `"INR"`. The prompt does not
  normalize `$`→`USD`.
- Scoring: exact match after trim. Optionally treat the known symbol/code pairs as
  equal (`$`≈`USD`, `₹`≈`INR`, `€`≈`EUR`, `£`≈`GBP`) **only** if your bake-off wants to
  be lenient — the goldens store the literal printed token, and `meta.json.currency`
  gives the ISO code for filtering.

---

## 4. Strings

- **Trim** leading/trailing whitespace on both sides before comparing
  (`isEmptyScalar` treats `"  "` as empty; canonical resolvers `.trim()` string
  values throughout, e.g. `vendorInvoice.js:101`).
- **Collapse internal whitespace runs — including newlines — to a single space** before
  comparing. The `document.txt` in this bundle is Google Document AI output, which keeps
  its own newlines (one token/line group per line), so a model that emits a multi-line
  address as `"330 Hudson Street, Suite 800\nNew York, NY 10013"` and a golden that
  stores it space-joined must compare equal after `\s+ -> " "`. (A Tesseract-engine
  tenant's text would already be space-joined — `ocr_service.py clean_text_lines`.)
- **Case:** key matching is case-insensitive (§6). **Values** are compared
  case-sensitively by default — but for `document_type`, `po_type`, `payment_terms`,
  `freight_terms`, `transport_mode` use case-insensitive value comparison (free-text
  labels; a model returning `"purchase order"` vs `"Purchase Order"` is not wrong).
- **Punctuation / separators:** do not strip punctuation from identifiers
  (`invoice_number`, `purchase_order_number`, `hsn_code`, tax IDs) — `PO/2627/0208`,
  `INV-2026-04510`, `27AAQPN4417R1Z3` must match exactly (after trim).
- **Addresses:** compare after trim + whitespace-collapse; accept `,` vs ` ` line-break
  differences (`"Pune, Maharashtra - 411052"` == `"Pune Maharashtra - 411052"` is a
  judgement call — goldens keep the printed commas; a bake-off may score addresses with
  a token-set ratio rather than exact).
- **Placeholder → null:** the PO prompt says treat `"-"`, `"--"`, `"N/A"`, empty cell as
  `null`, **not** literal text. Apply this when scoring: an expected `null` and a model
  output of `"-"` / `"N/A"` / `""` are **equal**. (The invoice prompt is less explicit
  but the same rule is applied to its goldens.)

---

## 5. Nulls / absence

`isEmptyScalar` (`shared/fieldResolution.cjs:35`, and the identical
`canonicalResolver.js:93`): a value is "empty" iff it is `null`, `undefined`, or a
string that is `""` after `.trim()`.

Scoring equivalence — **all of these are equal to expected `null`:**

| Model output | Equal to `null`? |
| --- | --- |
| `null` | yes |
| key omitted entirely | yes |
| `""` or `"   "` | yes |
| `"-"`, `"--"`, `"N/A"`, `"NA"`, `"null"`, `"None"` | yes (placeholder rule, §4) |
| `0` | **no** — a real zero (`igst_amount: 0` on an intra-state invoice) is distinct from "not present" |
| `[]` for `line_items` | yes, equals "no line items" (PO prompt: absent line_items → `[]`) |
| `{}` | yes for object-typed fields |

Corollary: **do not** score "model returned `null`" as a miss when the field genuinely
is not on the document. `meta.json.required_fields` lists the fields that MUST be
non-null for that specific case; everything else is pass-if-null-or-correct.

---

## 6. Keys / envelope shape

Before field-by-field comparison, normalize the model's envelope the way Flowgenix does:

1. **Unwrap one known wrapper** (`normalizeExtractionEnvelope.js:64`): if the object has
   exactly one nested-object key and it is one of
   `po, purchase_order, purchaseorder, invoice, bill, document, doc, data, result,
   extraction, extracted, fields, payload` → hoist its contents up (siblings preserved,
   inner wins collisions). Idempotent. Multi-object envelopes are left alone.
2. **Alias the line-items array key** to a canonical one: any of
   `line_items, lineItems, line_item, lineItem, items, rows` (first array wins).
   The goldens use `line_items`.
3. **Key match is fuzzy** (`keysMatch`, `shared/fieldResolution.cjs:18`): case-insensitive
   **and** camelCase⇄snake_case equivalent. `invoiceNumber` == `invoice_number` ==
   `Invoice Number`→`invoice_number`. Score expected key `invoice_number` as satisfied
   by any of these.
4. **Value aliases** (`FIELD_ALIASES`, `:53`): `total_amount` is also matched by
   `grand_total`, `grandTotal`, `invoice_total`, `invoiceTotal`. Extend per your assert
   if you see other consistent aliasing.
5. **Extra keys the model adds are not penalised** (`additionalProperties: true`) —
   unless your bake-off separately measures verbosity / hallucinated fields.

---

## 7. Line items

- Compare as an **ordered list**; align by index (goldens preserve document row order).
- Per row, apply §1 (amounts), §4 (strings), §5 (nulls) to each field.
- **Quantity default:** the invoice canonical normalizer sets `quantity = 1` when a
  line has no quantity (`vendorInvoice.js:315-324`, "service-line default"). So for
  services invoices where a row prints no qty, a model output of `null` **and** `1` for
  that line's `quantity` are both acceptable (goldens store what the document prints:
  `null` if truly absent, the number if present).
- **Filler rows:** PO sources often have blank numbered rows (`3 4 5 6` with no data).
  Goldens include only the populated rows. A model that emits empty `{}` rows for the
  fillers should have them dropped before comparison (drop rows where every value is
  null/empty).
- A row-count mismatch is a real miss (missing or hallucinated line). Report it
  distinctly from field-level misses.

---

## 8. JSON parse / repair (applied before any of the above)

`parseLlmJsonResponse` (`llmJsonParser.js:6`) → `validateAndFixJSON` (`jsonFixer.js:141`):

1. `JSON.parse` as-is. If OK → done (`repairAttempted: false`).
2. Else `fixMalformedJSON` (`jsonFixer.js:8`): extract JSON from ```` ```json ```` fences
   or first `{`…last `}`; strip fence artifacts; quote unquoted keys; close unclosed
   arrays; drop trailing commas; quote bare string values. Then `JSON.parse` again.
3. Still failing → `success: false` → Flowgenix error `INVALID_JSON_FORMAT`, upload
   marked failed (see `../error_contract.md`).

For scoring: run the model output through the same parse+repair. A response that only
passes **after** repair still counts as a JSON-valid extraction for field scoring, but
record `repairAttempted` / `repairSuccess` as a model-quality signal (a model that needs
repair often is a latent `INVALID_JSON_FORMAT` risk in production).
See `reusable/llm_json_repair_cases.json` for the exact repair pairs pinned by the
Flowgenix test suite.
