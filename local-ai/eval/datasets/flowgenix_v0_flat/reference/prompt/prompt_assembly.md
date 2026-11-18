# Prompt assembly — Flowgenix structured field-extraction scene

All paths are relative to the Flowgenix repo root. Branch `reddot-poc`,
commit `704e91bf0742c226db5ba3c85a5521b109dc2b21`.

---

## 1. Where the prompt is built

| Step | Location |
| --- | --- |
| Scene entry (the model-agnostic seam the pilot routes through) | `backend/services/llm/LLMService.js:86` — `generateResponse(prompt, options)` |
| Job that drives extraction | `backend/queues/agenda.js` — Agenda job `"stage: ai_classify"` (defined ~`agenda.js:1319`) |
| Extraction-path selector | `backend/queues/agenda.js:1355` — `standard-single-prompt` is the branch this bundle targets (not `analysis-template`, `sap-prompt-chaining`, or `thermo-multi-prompt`) |
| Per-tenant / per-docType prompt lookup | `backend/services/prompt/PromptResolver.js:105` — `resolvePrompt({ tenantId, docType })`, called at `backend/queues/agenda.js:1809` |
| Field-plan resolution | `backend/services/promptFieldDerivationService.js:219` — `resolveFieldPlanForExtraction(...)`, called at `backend/queues/agenda.js:1904` |
| **Final prompt string assembly** | `backend/queues/agenda.js:1955-1968` (`fullPromptParts` → `fullPrompt`) |
| Handed to the LLM | `backend/queues/agenda.js:1987` — `result = await LLMService.generateResponse(fullPrompt, callOptions)` |
| Adapter that talks to the local server | `backend/services/adapters/openaiAdapter.js:9` — `execute(prompt, modelId, apiKey, baseUrl, endpoint, options)` |

`callOptions` for this scene (`backend/queues/agenda.js:1854-1863`):

```js
{
  tenantId, userId, docType,           // upload.docType — e.g. "Invoice" / "Purchase Order"
  uploadId,
  purpose: "extraction",
  jsonMode: true,                      // <-- drives JSON handling everywhere downstream
  documentComplexity: "low" | "medium" | "high",
  tokenCount: <estimate>,
}
```

Note: `callOptions` does **not** set `temperature`, `maxOutputTokens`, `schema`, `stop`,
or any `system` message. See `params.json`.

---

## 2. Assembly order (standard single-prompt path)

### 2a. Inside `agenda.js` — `fullPrompt`

`backend/queues/agenda.js:1955-1968` builds an array and joins the truthy entries with
`"\n\n"`, then appends the document text:

```
fullPromptParts = [
  JSON_ENFORCEMENT_PREFIX,          // (i)   backend/utils/aiConstants.js:1
  parsedPromptText,                 // (ii)  resolved Prompt.promptText for this tenant+docType
  promptFieldsContext,             // (iii) "" in the pinned pilot config (see §3) → dropped
  isPoNumberRelevant ? PO_NUMBER_GUARDRAILS : null,   // (iv) backend/queues/agenda.js:389
  fieldConfidenceEnabled ? FIELD_CONFIDENCE_INSTRUCTION : null,  // (v) OFF in pilot (see §3)
  JSON_ONLY_REMINDER,              // (vi)  backend/utils/aiConstants.js:10
]
// then:  fullPromptParts.push(`DOCUMENT CONTENT:\n${extractedText}`)   // (vii) agenda.js:1965
fullPrompt = fullPromptParts.filter(Boolean).join("\n\n")
```

So the emitted `fullPrompt` is:

```
(i)  JSON_ENFORCEMENT_PREFIX
     ⏎⏎
(ii) parsedPromptText            ← the invoice OR purchase-order prompt body
     ⏎⏎
(iv) PO_NUMBER_GUARDRAILS
     ⏎⏎
(vi) JSON_ONLY_REMINDER
     ⏎⏎
(vii) DOCUMENT CONTENT:
      <extractedText — the raw OCR string for the document>
```

`extractedText` is `upload.extractedText`, set by the OCR stage
(`backend/services/textAcquisition` → `backend/utils/ocrService.js`). For the pilot
tenant (`googleVision = "google document ai"`) it is Google Document AI's
`result.document.text`, `.strip()`ed — verbatim, newlines preserved, **no page
markers** (`backend/ocr_service_srs.py`). Each `cases/*/document.txt` in this bundle is
exactly that string. (A Tesseract-engine tenant would instead get per-page
`\n--- Page N ---\n` blocks with words whitespace-collapsed — `backend/ocr_service.py`.)

`(iii)` and `(v)` are empty/absent in the pinned pilot configuration (§3).
`${rawText}` interpolation (`agenda.js:1937`) does **not** fire — neither real prompt
body contains the literal `${rawText}`, so the document text is always appended as
block `(vii)`.

### 2b. Inside `openaiAdapter.js` — the wrapper the model actually sees

Because `options.jsonMode === true`, `backend/services/adapters/openaiAdapter.js:96-99`
wraps `fullPrompt` **again**:

```js
enhancedPrompt = `${JSON_ENFORCEMENT_PREFIX}\n\n${prompt}\n\n${JSON_ONLY_REMINDER}`;
```

where this adapter-local `JSON_ENFORCEMENT_PREFIX` / `JSON_ONLY_REMINDER` are defined at
`backend/services/adapters/openaiAdapter.js:83-94` and are **not byte-identical** to the
`aiConstants.js` versions — the adapter copy adds two lines:

```
CRITICAL: You MUST include a `confidence` key at the root of your JSON response with a number between 0.0 and 1.0 representing your confidence in the extraction.
If you are unable to extract structured data, respond with {"error":"NO_DATA","fields":{},"confidence":0,"message":"Unable to extract structured data"}.
```

(The `aiConstants.js` NO_DATA line has no `confidence` key and no language-translation
line ordering difference — compare `flowgenix_extraction.txt` lines 1-7 vs 9-14.)

**Final byte sequence sent as the single user message** (this is exactly
`prompt/flowgenix_extraction.txt`, and a fully-resolved instance is
`prompt/assembled_example.txt`):

```
<adapter JSON_ENFORCEMENT_PREFIX>          (openaiAdapter.js:83-91)
⏎⏎
<agenda fullPrompt from §2a>
⏎⏎
<adapter JSON_ONLY_REMINDER>               (openaiAdapter.js:93-94)
```

### 2c. Request shape

`backend/services/adapters/openaiAdapter.js:101-116`:

```jsonc
{
  "model": "<modelId from registry>",     // logical name, e.g. "local-gateway" / "celeris-local-primary"
  "messages": [
    { "role": "user", "content": "<the full byte sequence from §2b>" }
  ],
  "temperature": 0.1,                       // options.temperature ?? 0.1  (agenda sets nothing → 0.1)
  "response_format": { "type": "json_object" }   // added because jsonMode && supportsNativeJson
}
```

**There is no `system` role message.** Flowgenix sends one combined `user` message.
This matches the Celeris Local AI Server contract
(`docs/CONTRACT.md §5.1` — "Flowgenix: off (it sends its own complete prompt)") and the
`profiles/flowgenix.json` `"systemPrompt": null`. **Your harness must send the entire
assembled string as a single `user` message.**

`response_format: {"type":"json_object"}` is sent when the resolved model's registry
entry has `capabilities.supportsJsonMode === true`
(`backend/services/llm/LLMService.js:132` sets `options.supportsNativeJson`;
`openaiAdapter.js:114`). The `local-gateway` registry entry added for the pilot sets
`supportsJsonMode: true` (PROJECT_LOG §5.7), so **JSON mode is always on**.

---

## 3. Dynamic parts, and the single pilot configuration to target

| Part | Dynamic on | Pilot value to target |
| --- | --- | --- |
| `parsedPromptText` (ii) | tenant + docType (`Prompt` collection row) | The **super-admin invoice prompt** (`prompt/flowgenix_invoice_prompt.txt`) for invoice cases; the **super-admin PO prompt** (`prompt/flowgenix_po_prompt.txt`) for PO cases. These are the current global/default prompts set by the platform super-admin. |
| `promptFieldsContext` (iii) | `ENABLE_LEGACY_EXTRACTION` + whether the `Prompt` row has a configured `fieldsToExtract` catalog | **Empty string** — pin `ENABLE_LEGACY_EXTRACTION=true` (`backend/config/promptExtractionConfig.js:33`). The two real prompts carry their field list as prose / inline JSON schema in the body, not as a structured `Prompt.fieldsToExtract` catalog, so `buildCompactTargetFieldsContext([], [])` (`backend/utils/extractionPromptUtils.js:15`) returns `""` and the block is dropped. |
| `PO_NUMBER_GUARDRAILS` (iv) | `isPoNumberRelevant` (`agenda.js:1950`) | **Included.** With `ENABLE_LEGACY_EXTRACTION=true` (or `legacy-fallback` mode) `isPoNumberRelevant` is unconditionally `true` (`agenda.js:1951`). |
| `FIELD_CONFIDENCE_INSTRUCTION` (v) | `tenantProfile.confidenceScoreScope === "FIELD"` (`agenda.js:1342`) | **Omitted.** Target a pilot tenant whose `confidenceScoreScope` is not `"FIELD"` (the default). If you must eval the field-confidence variant, append the block from `backend/utils/fieldConfidence.js:82-91` after `parsedPromptText`. |
| `additionalInstruction` / `smartContext` | set only if `upload.classification` already exists (`agenda.js:1865-1867`) | **Omitted.** Run each case as a first-pass extraction with no prior classification blob. |
| model id echoed into `model` | registry resolution / tenant lock | `local-gateway` (registry) → server maps to `celeris-local-primary`. Irrelevant to prompt bytes. |
| `${rawText}` splice (`agenda.js:1937`) | prompt body containing the literal `${rawText}` | **N/A** — neither real prompt body contains it. |

### Why pin `ENABLE_LEGACY_EXTRACTION=true`

With the default (`false`), `resolveFieldPlanForExtraction`
(`backend/services/promptFieldDerivationService.js:219`) makes an **extra LLM call** to
derive a field catalog from the prompt text
(`deriveFieldCatalog`, `promptFieldDerivationService.js:117`) and, on success, injects a
`buildDynamicTargetFieldsSchema(...)` block (`backend/utils/extractionPromptUtils.js:60`)
and persists the derived catalog back onto the `Prompt` row — making the prompt bytes
**non-deterministic and stateful** across eval runs. Pinning the emergency-rollback flag
to `true` removes that call and yields a byte-stable prompt. If you instead want to eval
the default path, note that the injected block lists the same field keys already spelled
out in the prompt body, wrapped as `TARGET FIELDS (return exactly this JSON shape ...)`.

---

## 4. Runtime vs pilot — which adapter

- **Runtime default (production today):** `modelSelectionService` picks
  `Gemini / gemini-2.5-flash` (`backend/services/modelRegistry.js:8`,
  `getDefaultModelConfig` `modelRegistry.js:196`) → `geminiAdapter`. The Gemini adapter
  wraps with its own near-identical `JSON_ENFORCEMENT_PREFIX`
  (`backend/services/adapters/geminiAdapter.js:~90`).
- **Pilot (dev/test, what this bundle targets):** a `local-gateway` registry entry
  (`provider: "OpenAI"`, `baseUrl = LOCAL_AI_BASE_URL`, `supportsJsonMode: true`) plus
  either `AI_TARGET=local` in `modelSelectionService` or a per-tenant lock
  (`Tenant.LLMProvider="OpenAI"`, `modelId="local-gateway"`, `LLMUrl`, `LLMKey` —
  `backend/services/modelSelectionService.js:172`). This routes the scene through
  `openaiAdapter` (`backend/services/adapters/openaiAdapter.js`), which already accepts
  `baseUrl`/`endpoint` with no code change (PROJECT_LOG §5.3, §5.7). **`prompt/flowgenix_extraction.txt`
  is the `openaiAdapter` byte sequence** — this is the one your harness must match.

---

## 5. Possible drift between this doc and a live DB

- The two prompt bodies here are the **super-admin-authored** prompts provided for this
  bundle. `backend/seed.js:27-43` seeds *different, shorter* placeholder prompts for
  docTypes `invoice` / `po` on a "Demo Tenant"; a real environment resolves whichever
  row `PromptResolver` finds for the pilot tenant+docType (tenant-specific, then global
  `tenantId: null`). Confirm the pilot tenant resolves to these two bodies.
- `backend/scripts/update_db_prompts.js` appends a "multiple distinct documents"
  paragraph to any prompt row lacking that phrase. If that migration ran against the
  pilot's prompt rows, `parsedPromptText` will have that extra trailing paragraph.
  Neither super-admin body here contains the phrase, so treat a trailing
  multi-document paragraph as an expected, benign delta.
