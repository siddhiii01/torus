# Error contract — Flowgenix extraction scene

What each error can be, what triggers it, and what Flowgenix does next. This is the
behaviour the Celeris Local AI Server must not break: **overload / rate-limit must come
back as HTTP `429` / `503`, never `200` with an error body** (PROJECT_LOG §5.9 #6,
`docs/CONTRACT.md §2`).

Two layers produce error strings:
1. **Adapter layer** — `backend/services/adapters/openaiAdapter.js` (+ `apiKeyManager.handleApiError`)
   returns `{ error: "<CODE>", message, details, status }` instead of throwing.
2. **Extraction job** — `backend/queues/agenda.js` `"stage: ai_classify"` turns that into
   a persisted failure via `persistExtractionFailure`
   (`backend/services/extractionPipelineService.js:154`).

Common downstream path for every hard failure (`persistExtractionFailure`):
- upsert `ExtractedResult` with `extractedData: null`, `rawResponse` (kept if any),
  `errorInfo { code, message, parseError, repairFailureReason, repairAttempted, repairSuccess }`,
  and `validationWarnings: [{ field: "_extraction", message: "Extraction unavailable — validation skipped" }]`.
- `updateUploadStatus(uploadId, "failed", reason, { step: "AI Extraction", status: "failed" })`.
- `upload.status = "failed"`, `upload.autoProcessing = false`, `validationErrors/Warnings = []`.
- `resetAutoProcessing(uploadId)` — clears the auto-processing flag so it is not retried
  automatically.
- `emitPoExtractionLog(...)` — structured log line.
- **Validation, canonical mapping, approval routing, ERP sync are all skipped.** The
  upload sits in `failed` until a human re-triggers it.

---

## `INVALID_JSON_FORMAT`

| | |
| --- | --- |
| Trigger | Model returned text that is not valid JSON **and** `jsonFixer.validateAndFixJSON` could not repair it (`backend/utils/llmJsonParser.js:40`, `backend/utils/jsonFixer.js:141`). Also fires on empty/whitespace-only completion, and on a **truncated** completion (`finish_reason: "length"` — output hit the token cap; see `params.json` → `maxOutputTokens_detail`). |
| Raised at | `backend/services/adapters/openaiAdapter.js:150-162` (`error: "INVALID_JSON_FORMAT"`, `message` = `EXTRACTION_FAILURE_MESSAGE` = `"Extraction failed: invalid JSON returned by LLM"`, `details` = parse/repair reason). |
| Flowgenix does | `agenda.js:2163` → `persistExtractionFailure({ reason: EXTRACTION_FAILURE_MESSAGE })` (`agenda.js:2196-2206`). `errorInfo.code = "INVALID_JSON_FORMAT"`, `json.parseStatus = "FAIL"`, `repairAttempted`/`repairFailureReason` recorded. Upload → `failed`. |
| Bake-off signal | Highest-severity model-quality failure. A model that trips this on clean input is unusable for the pilot regardless of field accuracy (PROJECT_LOG §5.9 #3). Track raw rate and post-repair rate separately. |

## `NO_DATA`

| | |
| --- | --- |
| Trigger | The model itself returned the sentinel body `{"error":"NO_DATA","fields":{},"confidence":0,"message":"Unable to extract structured data"}` (instructed by both the `aiConstants` prefix and the `openaiAdapter` prefix), **or** the parsed object has no usable keys (`isNoDataExtraction`, `agenda.js:609`: object with `error === "NO_DATA"` or with zero keys outside `confidence/message/error/fields`). |
| Detected at | `isValidExtractedData` (`backend/services/extractionPipelineService.js:146` — returns `false` for `{ error: "NO_DATA" }` or empty object) and `isNoDataExtraction` (`agenda.js:609`). |
| Flowgenix does | **Non-AWS tenant (the pilot):** `!isValidExtractedData(extractedData)` at `agenda.js:2212` → `persistExtractionFailure({ reason: EXTRACTION_FAILURE_MESSAGE })`. Upload → `failed`. **AWS tenant only:** first a **retry** with a hard-coded fallback prompt (`agenda.js:2095-2141`, `purpose: "extraction:aws-fallback"`); if still NO_DATA → `persistExtractionFailure({ reason: "AI extraction returned no structured data for this AWS tenant document." })` (`agenda.js:2228-2253`). |
| Bake-off signal | Expected on the wrong-document-type case (`cases/invoice-09`). On a real invoice/PO it is a model failure — count it. |

## `INVALID_SCHEMA`

| | |
| --- | --- |
| Trigger | Ajv validation of the parsed JSON failed after one repair-retry — **only when `options.schema` was passed to the adapter**. |
| Raised at | `backend/services/adapters/openaiAdapter.js:186-196`. |
| This scene | **Cannot occur.** The standard single-prompt extraction path passes no `schema` in `callOptions` (`agenda.js:1854-1863`), and the Celeris middleware profile is `schema: null`. Listed for completeness — it *does* occur on the SAP path (`/api/sap/analyze`), which is out of scope. If a future middleware profile turns schema validation on, expect `x_celeris_validation: "failed"` in `metadata` and the model's last output still returned (`docs/CONTRACT.md §5.3`), i.e. no hard 5xx. |

## `RATE_LIMIT_EXCEEDED`

| | |
| --- | --- |
| Trigger | Upstream returned HTTP **429**. |
| Raised at | `apiKeyManager.handleApiError` (`backend/services/apiKeyManager.js:223-233`) → surfaced by `openaiAdapter.js:222-239` catch as `{ error: "RATE_LIMIT_EXCEEDED", status: 429, retryAfter }`. (Gemini adapter has the same code at `geminiAdapter.js:343-355`.) |
| Flowgenix does | `agenda.js:2163` treats it like any adapter error → `persistExtractionFailure({ reason: errorMessage })` (message = `"API rate limit exceeded for <provider>. Please try again later."`). Upload → `failed`. **No automatic retry on this scene** (the adapter's own retry loop does not cover 429 for OpenAI). `retryAfter` is recorded but not acted on here. |
| Server must | Return real HTTP `429` with `error.code: "rate_limit_exceeded"` when the queue/concurrency cap is hit (`docs/CONTRACT.md §2`). |

## `API_OVERLOADED`

| | |
| --- | --- |
| Trigger | Upstream returned HTTP **503** and the adapter's internal retry budget was exhausted. |
| Raised at | **Gemini adapter only** — `backend/services/adapters/geminiAdapter.js:303-340` (`maxRetries` exponential backoff, then `{ error: "API_OVERLOADED", retryAfter: 300 }`). |
| OpenAI path (the pilot) | A 503 does **not** become `API_OVERLOADED`. `handleApiError` (`apiKeyManager.js:236-244`) maps 503 → `{ code: "SERVICE_UNAVAILABLE", status: 503 }`, and `openaiAdapter.js` surfaces that. **Discrepancy to flag:** downstream Flowgenix code and dashboards that branch on the literal string `API_OVERLOADED` (PROJECT_LOG §5.9 #6) will not match a 503 that arrives via the OpenAI adapter — they'll see `SERVICE_UNAVAILABLE`. If parity matters, either (a) have the Celeris server keep 503s rare and fast, or (b) note that the pilot's overload signal is `SERVICE_UNAVAILABLE`, not `API_OVERLOADED`. |
| Flowgenix does | Same as any adapter error → `persistExtractionFailure`. Upload → `failed`. No auto-retry at the job level. |

## `SERVICE_UNAVAILABLE`

| | |
| --- | --- |
| Trigger | Upstream HTTP **503** via the OpenAI adapter (model loading, runtime down, health-check failing — `docs/CONTRACT.md §2` `model_unavailable`). |
| Raised at | `apiKeyManager.handleApiError` `:236-244`. |
| Flowgenix does | `persistExtractionFailure`, message `"The AI service is temporarily unavailable. Please try again in a few minutes."`. Upload → `failed`. |

## `API_AUTH_FAILED`

| | |
| --- | --- |
| Trigger | Upstream HTTP **401 or 403** — missing/invalid bearer key (`docs/CONTRACT.md §2` `invalid_api_key`). |
| Raised at | `apiKeyManager.handleApiError` `:198-220`; special-cased at `agenda.js:2168-2188` (leaked-key wording if `details` contains "leaked"/"reported"). |
| Flowgenix does | `persistExtractionFailure` with an auth-specific `reason` telling the operator to fix the key in tenant settings. Upload → `failed`. Not retryable. |

## `API_KEY_NOT_FOUND`

| | |
| --- | --- |
| Trigger | No API key resolvable for the call — `LOCAL_AI_API_KEY` / tenant `LLMKey` / env all absent (`backend/services/adapters/openaiAdapter.js:39-77`). Local to Flowgenix; no request is sent. |
| Flowgenix does | Adapter returns `{ error: "API_KEY_NOT_FOUND" }` → `persistExtractionFailure`. Upload → `failed`. |

## `API_ERROR` (generic)

| | |
| --- | --- |
| Trigger | Any other non-2xx from upstream — including **HTTP 400** (`invalid_request` — bad model name, `n>1`, malformed body) and **HTTP 408** (`timeout`) and **HTTP 502** (`upstream_error`). `handleApiError` has no dedicated 400/408/502 branch, so all fall through to `:246-254` → `{ code: "API_ERROR", status: <original> }`. |
| Flowgenix does | `persistExtractionFailure`, message `"An error occurred with the <provider> API. Please try again later."`, `errorInfo.code = "API_ERROR"` (the original HTTP status is kept in `result.status` / `details` but not in the code string). Upload → `failed`. |
| Note | **408 (timeout) is not distinguished from a generic API error** on the Flowgenix side — it will not auto-retry and the upload fails. The Celeris server should keep generations inside its 90 s default / 115 s cap and return `408` promptly rather than letting Flowgenix's timeout-less axios call hang (PROJECT_LOG §5.9 #5). |

## `NETWORK_ERROR`

| | |
| --- | --- |
| Trigger | No HTTP response at all — connection refused, DNS failure, socket reset (`apiKeyManager.js:262-268`). Likely if `LOCAL_AI_BASE_URL` is wrong or the server is down and not even returning 503. |
| Flowgenix does | `persistExtractionFailure`, message `"Unable to connect to the AI service. Please check your internet connection."`. Upload → `failed`. |

## `OPENAI_ADAPTER_ERROR`

| | |
| --- | --- |
| Trigger | Fallback code when `handleApiError` returns no `code` (`openaiAdapter.js:236` — `handledError.code || "OPENAI_ADAPTER_ERROR"`). Rare; unexpected exception shape. |
| Flowgenix does | `persistExtractionFailure`. Upload → `failed`. |

## `LLM_SERVICE_ERROR`

| | |
| --- | --- |
| Trigger | An exception thrown *inside* `LLMService.generateResponse` before/around the adapter call — model selection failure, credential resolution throw, adapter constructor throw (`backend/services/llm/LLMService.js:213-227` catch). |
| Flowgenix does | `generateResponse` returns `{ text: null, data: null, error: "LLM_SERVICE_ERROR", message: err.message }`. `agenda.js:2163` → `persistExtractionFailure`. Upload → `failed`. |

## `EXTRACTION_JSON_FAILED` / `EXTRACTION_FAILURE_MESSAGE`

Not an adapter code — the **persisted** `errorInfo.code` when a failure has no more
specific code: `result.error || "EXTRACTION_JSON_FAILED"`
(`backend/services/extractionPipelineService.js:204`). `EXTRACTION_FAILURE_MESSAGE`
(`"Extraction failed: invalid JSON returned by LLM"`, `llmJsonParser.js:3`) is the
human-readable `reason` used for `INVALID_JSON_FORMAT` and for the
`!isValidExtractedData` (empty / NO_DATA) case.

---

## Success, but still rejected

Even on adapter success (`result.error` falsy), the job rejects the extraction if
`!isValidExtractedData(extractedData)` (`agenda.js:2212`):
`null`, non-object, empty array, empty object, or `{ error: "NO_DATA" }`. Same
`persistExtractionFailure({ reason: EXTRACTION_FAILURE_MESSAGE })` path. So a model that
returns syntactically valid `{}` or `{"confidence":0.2}` still fails the upload.

## Not errors (no failure)

- `usage` missing / zero — **not** an error, but cost metering silently records nothing
  (PROJECT_LOG §5.9 #7). The contract requires `usage` always populated
  (`docs/CONTRACT.md §2`); a bake-off should assert `usage.total_tokens > 0`
  (the DeepEval skeleton does: `test_flowgenix.py` `_call`).
- `finish_reason: "stop"` with a valid but **incomplete** extraction (missing fields) —
  no error; shows up only as low field-match score.
- `ambiguous` date (`11/07/2026`) — canonical layer warns, does not fail
  (`canonicalResolver.js:543`).
