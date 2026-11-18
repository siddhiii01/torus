# Architecture

Companion to [`CONTRACT.md`](CONTRACT.md) (the external API) and `PROJECT_LOG.md`
(background + decisions). This is the internal shape.

## Principles

1. **Model-agnostic.** Callers never name a model family. A logical name
   (`celeris-local-primary`) is mapped server-side. The bake-off winner can change
   without touching any product.
2. **Provider-agnostic surface.** One OpenAI-compatible protocol. If a local provider
   other than Ollama is ever used, it changes only LiteLLM config.
3. **Product code stays clean.** Products change a base URL + key in dev/test config.
   Nothing about this repo ships inside a product.
4. **The existing RAG service is not rebuilt.** We call its HTTP API.
5. **Thin middleware.** It does only what LiteLLM can't: per-product identity,
   profile, RAG orchestration, output validation, content-free structured logs.
6. **Provably no production impact.** New env vars are absent in prod; no shared
   config path with prod.

## Components & flow

```
Product (dev/test) ──POST /v1/chat/completions──►  MIDDLEWARE  (this repo, :4100)
                                                   1. identity   key → productId (hashed lookup)
                                                   2. profile    load profiles/<product>.json
                                                   3. rag        if profile.has_rag → RAG service
                                                   4. forward    → LiteLLM
                                                   5. validate   if profile.schema (1 retry)
                                                   6. log        content-free structured line
                                                        │
                                                        ▼
                                                   LiteLLM PROXY  (:4000, Docker, pinned)
                                                   · logical model → ollama/<model>
                                                   · concurrency queue (max_parallel 2–3)
                                                   · token accounting · 429/503 on overload
                                                        │
                                                        ▼
                                                   OLLAMA  (:11434, native, MLX/Metal)
                                                   · runs the bake-off winner
                                                   · launchd KeepAlive
```

### Middleware (`middleware/`)

| Module | Responsibility |
| --- | --- |
| `src/index.js` | HTTP server; wires the 6 steps; `GET /health` |
| `src/identity.js` | `Authorization: Bearer` → `productId` via hashed lookup in `.secrets/keys.json`. Never trusts the request body. |
| `src/profileLoader.js` | Loads + caches `profiles/<productId>.json`; validates its shape once at startup |
| `src/ragClient.js` | Called only when `profile.has_rag`. Talks to the external RAG service; collection/project resolved from the profile, never the request. |
| `src/validate.js` | Optional JSON-schema check of the model output; one retry with the error appended; annotates `metadata` on repeated failure (no hard 5xx) |
| `src/logger.js` | One structured JSON line per request. Fields in `CONTRACT.md` §5.4. **No prompt/response text, ever.** |
| `src/upstream.js` | Thin client to LiteLLM `/v1/chat/completions` with the upstream key + timeout |
| `src/config.js` | Env parsing, defaults, startup validation |

For the **Flowgenix MVP** every optional step is off, so the middleware is
effectively: identity → forward → log. Kept as its own service anyway because it is
the seam every later product plugs into.

### Gateway (`gateway/`)

LiteLLM proxy, official Docker image pinned to an exact tag (`PROJECT_LOG.md` notes
its CVE history — no `:latest`, upgrades are reviewed PRs). Config in
`gateway/config/litellm_config.yaml`:

- `model_list`: logical names → `ollama/<model>` with `api_base`.
- `router_settings.max_parallel_requests`: 2–3, matching the 64 GB machine.
- `litellm_settings.turn_off_message_logging: true` — no raw content in LiteLLM logs.
- Virtual keys: LiteLLM issues its own; the middleware holds one upstream key. Product
  keys are the middleware's concern, not LiteLLM's.

### Model runtime

Ollama, **native** (not Docker — needs MLX/Metal GPU access), kept alive by launchd
(`deploy/com.celeris.localai.plist`). Env: `OLLAMA_NUM_PARALLEL=2–3`,
`OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=1–2`. Model list settled by the
bake-off (`PROJECT_LOG.md` §4).

### RAG (external — reference only)

Already-deployed standalone service (`architecture-overview.md`). We call:

- `POST /v1/retrieve` → `{ chunks: [...] }` — preferred: middleware builds the prompt.
- (`POST /v1/ask` delegates generation to the RAG service's own LLM — not used by us
  for now.)

Auth: per-project bearer key. A "Local AI product" maps to a "RAG project" 1:1;
provisioning (project + key + config + ingested docs) is a manual admin step per
product. **Full contract pending the RAG codebase report** (`PROJECT_LOG.md` §8.2).
Not needed for Flowgenix.

### Evaluation (`eval/`)

- **Promptfoo** — one-time model bake-off. One config, N candidate models, the
  collected Flowgenix extraction cases. Run one model at a time (64 GB).
- **DeepEval** — CI regression gate. Runs on any `profiles/` change and nightly.
  Metric-based with tolerance bands, not exact match. Local judge model only.

### Monitoring (MVP)

`logger.js` writes JSON lines to `middleware/logs/requests.log`. A tiny read-only page
tails it: recent requests, pass/fail rate, running "cloud cost avoided" estimate
(tokens × equivalent cloud price). Post-MVP: Grafana + Loki, then the Super Admin
dashboard Nikhil asked for.

## Deployment

`deploy/docker-compose.yml` runs `litellm` + `middleware`. Ollama runs natively via
launchd. `GET /health` on the middleware is the single health signal. Secrets via
`.env` + `middleware/.secrets/` on the host — never in the image, never committed.
