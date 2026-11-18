# Celeris Local AI Server — External API Contract (v1, frozen)

Status: **proposed — awaiting teammate sign-off** (Open Q B-4).
Everything downstream is built to this. Changes here are deliberate, versioned PRs.

The contract is **product-agnostic**. Flowgenix is just the first caller that
exercises it (see `PROJECT_LOG.md` §5). It is also **model-agnostic**: callers never
name Qwen / Mistral / Llama — they send a logical model name and the server maps it.

---

## 1. Transport

- Base URL: `http://<celeris-ai-server>:4100` (stable VPN name, never a hardcoded LAN IP).
- Protocol: **OpenAI Chat Completions**, JSON over HTTP.
- One MVP endpoint plus health. No streaming, embeddings, rerank, vision, or
  tool-calling in v1 — added only when a product actually needs them.

---

## 2. `POST /v1/chat/completions`

### Request

Standard OpenAI shape. Fields we commit to honouring:

| Field | Required | Notes |
| --- | --- | --- |
| `model` | yes | **Logical name.** MVP values: `celeris-local-primary` (bake-off winner), `celeris-local-fast` (small model). Server maps to the real model. Unknown name → `400`. |
| `messages[]` | yes | `role` ∈ `system\|user\|assistant`. Total prompt content **must accept ≥ 150,000 characters (~40k tokens)**. |
| `response_format` | no | **`{"type":"json_object"}` MUST be honoured** — forces valid-JSON output. Extraction quality depends on it. |
| `temperature` | no | Default `0.1` if absent. Pass-through. |
| `max_tokens` | no | Default server-side (`2048`), max `8192`. |
| `metadata` | no | Opaque object, echoed to logs only (never to the model). Use for `{ "purpose": "...", "trace_id": "..." }`. |

Ignored if sent (accepted, no error): `stream` (always treated as `false` in v1),
`tools`, `functions`, `logprobs`, `n` (>1 rejected with `400`).

### Response — success (`200`)

Standard OpenAI shape. Guarantees:

```jsonc
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "celeris-local-primary",           // logical name echoed back
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"                  // or "length"
    }
  ],
  "usage": {
    "prompt_tokens": 1234,                     // ALWAYS populated, never null/0-by-omission
    "completion_tokens": 567,                  // callers meter cost from this
    "total_tokens": 1801
  }
}
```

- `choices[0].message.content`: when `response_format` was `json_object`, this is a
  syntactically valid JSON string. The server does **not** guarantee it matches any
  caller schema — that's the caller's / middleware profile's job.
- `usage`: **all three fields always present and real.** If the underlying runtime
  can't supply counts, the server estimates and sets `"usage_source": "estimated"`.

### Response — errors

OpenAI error envelope: `{ "error": { "message", "type", "code" } }`.

| HTTP | `error.code` | When | Caller expectation |
| --- | --- | --- | --- |
| `400` | `invalid_request` | bad/missing `model`, `n>1`, malformed body | fix and retry |
| `401` | `invalid_api_key` | missing/unknown bearer key | not retryable |
| `408` | `timeout` | generation exceeded the request timeout | retryable |
| `429` | `rate_limit_exceeded` | queue full / concurrency cap hit | back off + retry |
| `503` | `model_unavailable` | model loading, runtime down, health-check failing | back off + retry |
| `502` | `upstream_error` | LiteLLM / Ollama returned an unexpected error | retryable |

Flowgenix (and likely others) branch on overload → **must** see `429`/`503`, not `200`
with an error string. Downstream error-code strings this maps to on the Flowgenix side:
`API_OVERLOADED`, `RATE_LIMIT_EXCEEDED`, `NO_DATA` (see `PROJECT_LOG.md` §5.9).

### Timeouts

- Server enforces a **per-request timeout**, default **90 s**, hard cap **115 s**
  (callers that set no client timeout must not hang for minutes — see the Agenda
  worker 30-min lock risk, `PROJECT_LOG.md` §5.9).
- Override per request via header `X-Celeris-Timeout-Ms` (clamped to `[5000, 115000]`).
- On timeout: `408`, and the in-flight generation is cancelled.

---

## 3. `GET /health`

```jsonc
{
  "status": "ok",                 // "ok" | "degraded" | "down"
  "gateway": "ok",                // middleware itself
  "litellm": "ok",                // reachable + ready
  "model_runtime": "ok",          // Ollama reachable + a model loaded
  "queue_depth": 0,
  "loaded_model": "celeris-local-primary",
  "version": "0.1.0"
}
```

- `200` when `status` is `ok` or `degraded`; `503` when `down`.
- No auth. Safe to poll from monitoring and from a product's own health check.

---

## 4. Auth & identity

- Header: `Authorization: Bearer <key>`.
- Key format: `clr_<productId>_<random>` (e.g. `clr_flowgenix_a1b2c3...`). The
  `productId` is **not** trusted from the token text — the full key is looked up
  (hashed) in the middleware's key table, which is the single source of
  key → `productId`.
- One key per product per environment. No anonymous access. No product identity is
  ever read from the request body.
- Keys live in `middleware/.secrets/keys.json` (git-ignored) and, for LiteLLM's own
  upstream key, in `.env`. Never commit either.

---

## 5. What the middleware adds on top (per-product, via `profiles/<product>.json`)

All optional and off unless the profile turns them on:

1. **System-prompt injection** — prepend `profile.systemPrompt` as a `system` message.
   *Flowgenix: off* (it sends its own complete prompt).
2. **RAG** — if `profile.has_rag`, call the external RAG service (`docs/architecture.md`
   §RAG) with that product's RAG project key and inject retrieved chunks.
   *Flowgenix: off* (`has_rag: false`, evidence-backed — `PROJECT_LOG.md` §5.6).
3. **Output validation** — if `profile.schema`, validate `message.content` against it;
   one retry with the validation error appended; then return the model's last output
   plus `"x_celeris_validation": "failed"` in `metadata` (never a hard 5xx — the caller
   decides).
   *Flowgenix: off* for the MVP (Flowgenix runs its own Ajv check).
4. **Structured logging** — every request logs `{ productId, model, latency_ms,
   prompt_tokens, completion_tokens, ok, attempts, error_code, purpose }`.
   **Never** the prompt or completion text.

---

## 6. Explicitly out of scope for v1 (decide later)

- **Gemini-compatible `…:generateContent` surface.** Only needed to route *every*
  Flowgenix scene without a Flowgenix-side refactor. Recommendation: keep the server
  single-protocol; revisit post-MVP (Open Q B-8). Check first whether LiteLLM can
  expose a Google-AI route cheaply.
- Streaming responses.
- `/v1/embeddings`, `/v1/rerank`, vision input, audio.
- Per-request model *selection* by the caller beyond the logical names above.
