# Celeris Local AI Development Platform — Project Log

> Living document. Single source of truth for decisions and open threads.
> **Rule for this project: no assumptions.** Anything uncertain goes into
> "Open Questions" and gets answered before it drives code.

Last updated: 2026-08-31 (v3 — teammate corrections + Flowgenix codebase analysis received)

---

## 0. Status at a glance

| Item | Status |
| --- | --- |
| Initiative | **Approved by Nikhil.** Fresh implementation on the proposed architecture. |
| Owner / POC | **Siddhi** (primary), coordinating with Khuzaima + Heramb. Teammate + Claude build this repo together. |
| Hardware | **UNCERTAIN again (2026-09-02).** 64 GB MacBook Pro not confirmed; team may buy a small Windows/NVIDIA box (RTX 4060 8 GB / 32 GB / 1 TB) or nothing yet. **Interim dev host = a contributor's desktop: Ryzen 5 7600X, 16 GB RAM, RTX 50-series 16 GB VRAM, Windows + Docker Desktop.** Model strategy now tiered by VRAM budget — see ADR 0002. §4 (64 GB / 27–32B) paused, reactivates only if a ≥48 GB machine lands. |
| Repo | This repo = the standalone platform (`celeris-local-ai`). **No product code lives here.** Products call it over the network. |
| Language | Open — clean + maintainable is the only rule. Not yet ratified (Open Q F-23). |
| Pilot scope now | **Flowgenix only.** AI Email Agent deferred (teammate will hand it over later). ProcAgility/Tender later. |
| First product E2E | **Flowgenix — the structured field-extraction scene**, via its existing `LLMService` seam. Rationale in §5 + §10. |
| Week-1 work | All hardware-independent: contract freeze, repo scaffold, eval harness, Flowgenix test cases. Start now. |
| VPN | Teammate owns this decision separately. Not on our critical path. |
| RAG | **Already exists as a separate deployed product** (see §3.2). We do NOT rebuild it. RAG analysis prompt pending; teammate to run it. |
| Missing input | "Shortlisted models" PDF (teammate has it, compare later) · RAG codebase report. |

---

## 0.1 Corrections locked in (2026-08-31, from teammate)

1. **The Local AI Server is MODEL-AGNOSTIC and PROVIDER-AGNOSTIC.** It is not designed
   around Gemini or anything Flowgenix uses today. Products call a neutral API; the
   server picks the model/provider underneath; that is swappable by config with **no
   change to any product**. The recommended models (§4) are **bake-off candidates
   only**, never an architectural dependency.
2. **The RAG service already exists** as an independently deployed product (FastAPI,
   hexagonal, multi-tenant — see `architecture-overview.md`, §3.2). We do **not**
   rebuild, absorb, or duplicate it. The Local AI Server calls its HTTP API when a
   product needs retrieval. Its Node/Python/vector-store choices are its own.
3. **The Build Guide's "FIL" claim was wrong.** It said Flowgenix's only LLM call is
   the FIL (Forgery Intelligence Layer) semantic-validation scene on Gemini Flash.
   The Flowgenix codebase analysis (§5) shows FIL is **legacy and not wired into any
   live backend path**, and Flowgenix has **many** live AI calls, none of them FIL.
4. Scope is **Flowgenix first, in isolation.** Other products come after Flowgenix is
   integrated and understood.

---

## 1. What we were asked to do

### 1.1 Nikhil — original challenge
- Rising AI dev-test cost across Flowgenix, Memry.AI, ProcAgility, Tender, AIMS, AI
  Email Agent + others will throttle testing. Unacceptable for an AI company.
- Priority: **build a common local AI capability** for all teams to test against,
  before major further product progress.
- **Not** in scope: reducing Cursor; replacing OpenAI/Gemini in production / customer
  deployments.
- Customer demos stay on OpenAI/Gemini initially; **long-term goal** = demo locally
  once quality is proven.

### 1.2 Nikhil — approval + constraints (2nd mail)
Approach approved. Start hardware-independent work now; hardware in parallel via Heramb.

1. AI Gateway **completely independent** of product code — consumed like a common API.
2. Local-AI config must **not affect Cursor** or any production OpenAI/Gemini config.
3. Start with 3 pilot products but **complete ≥1 end-to-end** in the 2-week MVP; other
   2 = parallel / stretch.
4. **Fictional / non-sensitive data only.**
5. **No final model choice on generic benchmarks** — real use cases + measured quality.
6. **Deterministic validation = primary.** Any AI-based eval must be configurable and
   must **not call paid cloud models by default**.
7. Record: response quality, hallucinations, schema-validation success, latency,
   concurrency, machine-resource utilisation.
8. **Short milestone updates** throughout.
9. Hardware: 64 GB / 1 TB start; 128 GB future-ready; Mac Studio possible future.
10. Post-MVP progressively: LiteLLM routing depth, dashboard, broader CI regression,
    more onboarding, possible fine-tuning (only if eval shows gaps prompting +
    validation + RAG can't fix).
11. **P.S. future Super Admin dashboard** (models, active/default model, product access,
    usage, queue, latencies, failures, health). MVP = read-only view / structured logs.
12. First version of the plan in 1–2 days: POC approach, model eval, infra, security,
    deployment, performance benchmarks, responsibilities, timelines.

### 1.3 Nikhil — secure remote access (3rd + 4th mails)
- Secure office + remote access, no public port forwarding; must survive the server
  laptop changing networks.
- Doc uses **Tailscale**; free tier is trial-limited. Only 1–2 remote users now →
  **also evaluate NetBird Free, ZeroTier, self-hosted WireGuard.**
- Evaluate on: permanent free commercial use · ease of setup/admin · stable access on
  network change · SSH + RDP support · user/device access control · security without
  port forwarding.
- **Teammate owns this decision.** Not blocking our build.

---

## 2. The four PDFs — consolidated (with corrections applied)

### 2.1 Proposal (PDF 1) — strategic framing
Common local AI dev capability first; Cursor untouched; OpenAI/Gemini stay in
prod/demo until local quality independently proven. Components: gateway, common base
model, product profiles, per-product knowledge retrieval, validation layer,
evaluation framework, ops dashboard. Evidence-led bake-off. No fine-tuning in Phase 1.

### 2.2 Implementation Plan (PDF 2) — tech stack + 2-week MVP
- Gateway = **self-hosted LiteLLM** (OpenAI-compatible, version-pinned Docker).
- Middleware = thin custom layer (identity → optional profile → optional retrieval →
  validate → log).
- Product profile = JSON per product (system prompt, schema, few-shot). **Optional per
  product** — Flowgenix builds its own full prompt, so its profile is minimal.
- Retrieval = call the existing RAG service, **only for products that need it**.
- Validation = schema check + one retry + hard fail.
- Eval = **Promptfoo** (bake-off) + **DeepEval** (CI regression).
- Runtime = **Ollama** native on the Mac (MLX/Metal); Docker for the rest.
- Auto-start via **launchd**; `GET /health`.
- Monitoring MVP = JSON logs → simple read-only page (requests, pass/fail,
  cost-avoided). Post-MVP = Grafana + Loki → Super Admin dashboard.
- Candidate models as written are **stale — see §4**.

### 2.3 Secure Access Guide (PDF 3) — network + ops
- Pattern: dedicated AI server → private mesh VPN → authenticated middleware → local
  Ollama. No public exposure. Stable server name, never a hardcoded LAN IP.
- Don't expose Ollama/DB/SSH/RDP. Per-person VPN identities; app-level auth even over
  VPN; audit everything; revoke departed staff.
- Ops: AC power + UPS, no sleep, reliable broadband, auto-start, cooling, backups,
  health checks.
- **Laptop = POC only**; dedicated workstation/server for continuous multi-user use.
- **VPN product choice = teammate's, separate track.**

### 2.4 Build Guide (PDF 4) — partially superseded
- Hardware in hand: MacBook Pro 64 GB / 1 TB. ✔ still valid.
- ~~"Flowgenix's only LLM call is the FIL semantic-validation scene on Gemini Flash;
  validates a document against itself; `has_rag: false`."~~ **WRONG** — see §0.1(3)
  and §5. FIL is legacy/unwired; Flowgenix has many live AI calls. `has_rag: false`
  for Flowgenix is still the right outcome, but now for a different, evidence-backed
  reason (§5.6): the codebase has **no retrieval anywhere**.
- ~~"RAG service is built fresh, standalone, owned by Siddhi."~~ **WRONG** — the RAG
  service already exists (§0.1(2), §3.2).
- `has_rag` per-product conditional flag — **keep** (design is sound; just Flowgenix
  happens to be `false`).
- Bake-off = one model at a time (64 GB can't hold all candidates) — **keep**.
- Metrics list (quality, schema success, hallucination via local judge, latency,
  concurrency, CPU/mem) — **keep**.

---

## 3. Architecture (working understanding, v3)

### 3.1 Shape

```
Celeris products — DEV/TEST config only          PRODUCTION unchanged:
Flowgenix (+ later others)                        products keep their current
     │  point AI calls at the local server        provider (Gemini/OpenAI) — the
     │  via a neutral, model-agnostic API         local env vars are simply absent
     ▼
[ Mesh VPN — teammate's choice ]   stable name, no public ports
     ▼
Celeris AI Gateway  — LiteLLM (Docker, version-pinned)
  · OpenAI-compatible /v1/chat/completions   · per-product virtual keys = identity
  · model routing (logical name → local model)   · concurrency queue
  · usage/token logging   · 429/503 on overload   · no raw message logging
     ▼
Middleware (thin, custom) — only the value LiteLLM doesn't give us
  · resolve productId from key (never from body)
  · OPTIONAL: inject a product profile (system prompt / schema / few-shot)
  · OPTIONAL: call the external RAG service (only if that product needs retrieval)
  · deterministic output validation + one retry + hard fail
  · structured logging: product, latency, tokens, success, attempts, model — no raw text
     │                                    │
     ▼                                    ▼
External RAG service (ALREADY EXISTS)    Ollama (native, MLX/Metal, launchd KeepAlive)
  · its own repo + deployment             · runs the bake-off winner
  · HTTP API, multi-tenant, bearer keys   · OLLAMA_NUM_PARALLEL ≈ 2–3
  · we CALL it, never embed it            · OLLAMA_KV_CACHE_TYPE=q8_0
                                          · OLLAMA_MAX_LOADED_MODELS 1–2
     ▼
Eval:  Promptfoo (one-time bake-off)  +  DeepEval (CI regression gate)
  · deterministic checks primary  · LLM judge = local only, configurable, off by default
Monitoring MVP: JSON logs → read-only page  →  later Grafana+Loki  →  Super Admin dashboard
```

### 3.2 The external RAG service (context, not ours to build)

From `architecture-overview.md`: a standalone RAG pipeline, hosted separately, used by
multiple internal projects over an HTTP API. Ports-and-adapters. FastAPI. Public routes
`GET /health`, `POST /v1/ingest`, `GET /v1/jobs/{id}`, `POST /v1/retrieve`,
`POST /v1/ask`, `POST /v1/complete`; admin routes behind `RAG_ADMIN_TOKEN`. Per-project
bearer API key → `project_id` (hashed lookup). Vector store = Chroma or pgvector;
embeddings = local `all-MiniLM-L6-v2` (384-dim); LLM port has Ollama/Claude/GPT
adapters. Multi-tenant config in MongoDB.

**Implication for us:** if/when a product needs retrieval, our middleware calls this
service's `/v1/retrieve` (or `/v1/ask`) with that product's RAG project key. Nothing
to build here except the client call + provisioning. **Full contract pending the RAG
codebase report.** Flowgenix does not need it (§5.6).

### 3.3 Non-negotiable rules
- Product identity resolved server-side from the key, never from the request body.
- Ollama / DB / SSH / RDP never publicly exposed; only the gateway port.
- No raw prompt/response content in logs.
- Fictional / non-sensitive data only.
- Production behaviour of any product must be **provably** unchanged (env vars absent
  in prod; no shared config path with prod).

---

## 3.5 Local AI Server — external API contract (derived from Flowgenix evidence, §5)

This is the contract we build and freeze. It is **product-agnostic** — Flowgenix just
happens to be the first caller that exercises it.

**Endpoint (MVP):**
- `POST /v1/chat/completions` — OpenAI Chat Completions shape.
  - Request: standard (`model`, `messages[]`, `temperature`, `max_tokens`,
    `response_format`).
  - **Must honour `response_format: {type: "json_object"}`** (JSON mode). Extraction
    quality depends on it; without it, callers fall back to fragile JSON-repair.
  - Response: standard (`choices[0].message.content`), and **`usage` with
    `prompt_tokens` / `completion_tokens` / `total_tokens` always populated** — callers
    meter cost from this; missing counts = silent zero.
- `GET /health` — liveness of gateway + model runtime.

**Behaviour:**
- Auth: `Authorization: Bearer <per-product key>` → identity. No anonymous access.
- `model` is a **logical name** (e.g. `celeris-local-primary`); the server maps it to
  the actual local model. Callers never name Qwen/Mistral/etc.
- Overload / rate-limit / queue-full → **HTTP 429 or 503** (callers branch on this).
- **Enforce a per-request timeout** with a sane default well under **120 s**. (Some
  callers set no client timeout; a slow model must not hang them for minutes.)
- Input size: accept **≥ ~150,000 characters (~40k tokens)** of prompt.
- Output size: up to **~8k tokens**.
- Concurrency: queue beyond `OLLAMA_NUM_PARALLEL`; never hard-fail on contention —
  queue then 429 only if the queue is full.

**NOT in the MVP contract (decide later — Open Q B-8):**
- A **Gemini-compatible `…:generateContent`** surface. Needed only to route *every*
  Flowgenix scene without a Flowgenix-side refactor. Recommendation: **keep the server
  single-protocol (OpenAI-compatible)**; treat "consolidate onto one AI client" as
  each product's own modernisation. Re-check whether LiteLLM can expose a Google-AI
  route cheaply before committing either way.
- Embeddings / rerank / vision / streaming / tool-calling — **no Flowgenix need**
  found. Add only when a product actually requires them.

---

## 4. Model strategy for the 64 GB MacBook Pro

> Version strings in the PDFs are stale. Recommendation is by **capability tier +
> memory budget**; confirm exact current Ollama tags on ollama.com/library right
> before pulling.

**Resolved:** Flowgenix sends **text only** to LLMs (OCR text concatenated into the
prompt — never image bytes; §5.2). → **No vision model needed for the pilot.** The
earlier "images or text?" fork is closed: text.

### 4.1 Memory budget on 64 GB (shared service)

| Consumer | Approx |
| --- | --- |
| macOS + background | 6–10 GB |
| Docker (LiteLLM + middleware) | 2–4 GB |
| RAG (if co-hosted; may live elsewhere) | 0–3 GB |
| Ollama runtime overhead | 1–2 GB |
| Safety headroom | 4–6 GB |
| **Left for model weights + KV cache** | **≈ 40–46 GB** (≈ 50 GB if pushed) |

Flowgenix needs ≥ ~40k-token context → KV cache is non-trivial; budget 4–8 GB across
parallel slots, halved with `OLLAMA_KV_CACHE_TYPE=q8_0`.
→ **Target model weights ≤ ~30–35 GB.**

### 4.2 Bake-off candidates (evaluate later — not architecture)

| Tier | Model class | Quant | ~Weights | Role |
| --- | --- | --- | --- | --- |
| **A — primary** | Current-gen **Qwen dense 27–32B** | Q5_K_M pref / Q4_K_M | ~18–25 GB | Expected winner; best structured-output in class. Q5 helps JSON reliability. |
| **B — stable alt** | Current **Mistral Small ~24B** (Apache-2.0) | Q5 / Q4 | ~14–17 GB | Team familiarity, strong instruction-following. |
| **C — speed / fallback / local judge** | **12–14B dense** | Q4 / Q5 | ~8–11 GB | Fast iteration, high concurrency, low-mem fallback, eval judge. |
| **D — optional, 1 run** | **30B-class MoE (~3B active)** | Q4 | ~18–22 GB | Near-30B quality, 2–3× faster. |
| **E — optional, 1 solo run** | **70B dense** | Q4 | ~40–42 GB | Only to check "does bigger cut hallucination?" Not a shared-service default on 64 GB. |

### 4.3 Deploy after the bake-off
- One primary (Tier A) kept loaded; optionally Tier C as a 2nd loaded model if total
  < ~48 GB.
- `OLLAMA_NUM_PARALLEL=2–3`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=1–2`.
- Load/unload per candidate during the bake-off — never all at once.
- Judge model: reuse Tier C or a 7–8B; never paid cloud by default.

### 4.4 Verification searches (2026-08-31) — treat version strings as "confirm on official pages"
- Ollama moved its Apple-Silicon backend to MLX around v0.19 (early 2026) — matches PDFs.
- Current open Qwen dense ~27–32B exists (blogs vary: 3.5 / 3.6 / 3.8-27B) — confirm tag.
- Mistral Small 4 (Mar 2026, Apache-2.0) is the current small open Mistral; "3.2" is older.
- 64 GB Mac runs 27–34B at 4–5 bit comfortably (~15–20 tok/s); 70B Q4 possible but tight/slow.

Sources: apxml.com/posts/best-local-llms-apple-silicon-mac ·
insiderllm.com/guides/best-local-llms-mac-2026 ·
insiderllm.com/guides/qwen-models-guide · promptquorum.com/local-llms/best-models-apple-silicon-2026 ·
releasebot.io/updates/mistral · serenitiesai.com/articles/mistral-ai-models-2026-complete-guide

---

## 5. Flowgenix codebase analysis — findings (report received 2026-08-31, branch `reddot-sid @ 90dc634`)

### 5.1 What Flowgenix is
One git repo, sibling services. **`backend/`** = Node.js + Express 4, Mongoose 7 /
MongoDB, Agenda job queue, CommonJS, npm — **all live AI is here**. Two processes:
`server.js` (HTTP API) + `worker.js` (Agenda worker runs the document pipeline).
Deploy = GitHub Actions → SSH → PM2 (`flowgenix-backend`, `flowgenix-worker`). AI
config lives only in `backend/.env` on the host. `frontend/` makes no AI calls.

### 5.2 Live AI usage (all text-only — no images ever sent to an LLM)
| Provider · model | How | Scenes |
| --- | --- | --- |
| **Google Gemini `gemini-2.5-flash`** via raw `axios` (no SDK) | model + URL **hardcoded in ~8 files** | document classification · **structured field extraction (core IDP, default)** · region (NA/EMEA) detection · analysis-template multi-step · NL→Mongo "Flowgenix AI Assistant" · pre-sales proposal generation |
| **OpenAI `gpt-4o`** via `openai` npm SDK | hardcoded | `/api/sap/analyze` · non-default branch of the SAP prompt-chain |
| **Google Document AI + Tesseract** | Python scripts via `child_process` | OCR only — deterministic, not an LLM concern |
| Claude adapter + registry entries | — | **dead** (no API key in `.env`) |

**No embeddings, vector store, RAG, rerank, streaming, tool-calling, or vision**
anywhere (confirmed by grep).

### 5.3 The one clean seam
`backend/services/llm/LLMService.js` → `generateResponse(prompt: string, options)` +
Gemini/OpenAI/Claude adapters. Model chosen by `modelSelectionService` over
`modelRegistry`, with a **per-tenant override** (`Tenant.LLMProvider`, `modelId`,
`LLMUrl`, `LLMKey`). Covers **4 scenes**: structured extraction, validation-rule
suggestion, extraction re-validation, AI field-plan discovery.
**The `openaiAdapter` already accepts `baseUrl` / `endpoint` from the registry** — so
pointing this path at a local OpenAI-compatible server needs **no adapter code**.

**Adapter contract a new provider must satisfy:** accept
`(prompt:string, modelId, apiKey, baseUrl, endpoint, options)`; honour
`options.jsonMode` (return `data` = parsed JSON), `maxOutputTokens`, `temperature`,
`options.schema` (Ajv, 1 retry); return `usage` as
`{promptTokens, completionTokens, totalTokens}`; surface errors as string codes
(`API_OVERLOADED`, `RATE_LIMIT_EXCEEDED`, `INVALID_JSON_FORMAT`, `INVALID_SCHEMA`,
`NO_DATA`).

### 5.4 Everything else bypasses the seam
~6 call sites (`utils/geminiService.js`, `utils/aiService.js`, `ai/llm.js`,
`services/geminiService.js`, `services/geminiHelper.js`,
`controllers/generateWizardController.js`) each **hardcode a vendor URL** and carry
their own key handling / retry / JSON-repair. They parse Gemini's envelope literally
(`candidates[0].content.parts[0].text` + `usageMetadata`).

### 5.5 No environment-level provider / base-URL switch exists
No `OPENAI_BASE_URL` / `AI_BASE_URL` / `LLM_BASE_URL` anywhere. `forceProvider` /
`forceModel` are per-call `options` keys no production path sets. Today, redirection =
per-tenant DB edit **or** a `modelRegistry` code change. Historically staging shared
prod `.env` + prod MongoDB → **prod-bleed risk** if we're careless.

### 5.6 RAG — definitively not needed for Flowgenix
Grep for embeddings / vector DBs / retrieval / rerank / knowledge-base across
`backend/` → nothing. The only "context" injected into a prompt is the document's own
prior classification blob. DB lookups that exist (`gl_vendor_lookup`, Thermo PO
amounts) are exact-key queries, not retrieval. **No current or code-present planned
feature needs RAG.** → Flowgenix profile `has_rag: false`, evidence-backed.

### 5.7 What it takes to route Flowgenix's extraction scene to our server (dev/test only)
~3 lines, gated by new env vars, prod unaffected (vars absent → unchanged Gemini):
- add a `modelRegistry` entry `local-gateway` (`baseUrl` = `LOCAL_AI_BASE_URL`,
  `supportsJsonMode: true`);
- one line in `modelSelectionService` — `if (process.env.AI_TARGET === "local") return getModelConfig("OpenAI","local-gateway")`;
- one line in `apiKeyManager` — fall back to `LOCAL_AI_API_KEY`.
Zero-code alternative: set `Tenant.LLMProvider="OpenAI"`, `modelId="local-gateway"`,
`LLMUrl`, `LLMKey` on selected **test** tenants (existing tenant-lock path).

Routing **all** of Flowgenix = a medium refactor Flowgenix owns (add
`AI_GEMINI_BASE_URL` / `AI_OPENAI_BASE_URL` + replace ~8 literals, **or** consolidate
the 6 direct-call modules onto `LLMService`). **Out of scope for our MVP.**

### 5.8 Reusable eval material already in the Flowgenix repo
`tests/llmJsonParser.test.js` (+ `utils/llmJsonParser.js`) — golden LLM-JSON-repair
cases · `tests/normalize*` — post-extraction golden outputs (vendor invoice / PO /
GRN / envelope) · `ai/schema/readiness.schema.json` — SAP golden schema ·
`tests/fieldContract.test.js`, `extractionPromptUtils.test.js`, `fieldResolution.test.js`
· sample docs `test-invoice-sample.pdf`, `usecase_samples/`, `ocr_demonstration/`,
`backend/fixtures/`. **AI is barely mocked in their tests — no recorded golden
real-LLM responses exist**, so we build the bake-off set.

### 5.9 Risks / coupling (from the report)
1. **Gemini-envelope lock-in** — 6 live sites parse Gemini's REST shape literally.
   Highest integration cost if we ever go beyond the `LLMService` scene.
2. **No env-level provider switch** + historical staging/prod `.env` sharing → prod-bleed.
3. **JSON-mode dependence** — weak local model → `INVALID_JSON_FORMAT` → upload marked
   failed.
4. **Strict schemas** — SAP path throws on Ajv failure; adapter gives 1 retry then
   `INVALID_SCHEMA`.
5. **Latency** — classification runs **synchronously inside the HTTP upload handler**;
   `LLMService` adapters set **no HTTP timeout** → a slow local model blocks the Agenda
   worker up to its 30-min lock.
6. **Error-string coupling** — downstream branches on `API_OVERLOADED` /
   `RATE_LIMIT_EXCEEDED` / `NO_DATA` → our server must return 429/503.
7. **Cost metering silently zeros** without token counts in the response.
8. **PII in every prompt** — full invoice / PO / bank-statement OCR text (names,
   addresses, tax IDs, bank details); the region call ships the entire structured
   record. Local routing is *better* for PII, but note it.

### 5.10 Residual unknowns (Flowgenix)
- Whether any **production tenant** already has `Tenant.LLMProvider/modelId/LLMUrl` set
  (would change effective provider) — not visible from code.
- Prod-host `.env` values; Agenda worker concurrency in prod.
- `flowgenics-ai-service/` deployment status (not referenced by backend).
- FIL internals — out of scope; confirmed **not wired** to any live backend path.

---

## 6. The Ollama question (deferred — AI Email Agent comes later)

Kept for when we pick up the AI Email Agent. Two separate "Ollamas":
1. Ollama as **our server's runtime** — decided; internal detail behind the gateway.
2. **"AI Email Agent uses Ollama"** — unknown: is it live or an experiment? how does it
   call (client lib / REST / LangChain)? which models? is there reusable eval
   material? does any product need Ollama's native API? → revisit with that codebase.

---

## 7. Open Questions

### A. Hardware / network
1. MacBook Pro chip generation (M4 Max / M5 Max) — affects tok/s expectations.
2. Who administers the Mac; macOS version; unattended 24/7 now or POC-only?
3. Static office IP / internal hostname for `celeris-ai-server`?

### B. Contract / model
4. Ratify §3.5 as the frozen external contract.
5. Confirm current exact Ollama tags for Tier A / B / C before pulling.
6. **RESOLVED:** vision not needed for the Flowgenix pilot (text only).
7. **RESOLVED:** OpenAI-compatible surface is enough for the Flowgenix *extraction*
   pilot.
8. **DECISION NEEDED:** does the Local AI Server expose a **Gemini-compatible** route
   at all? (Recommendation: no for MVP; verify LiteLLM's Google-AI route option first.)
9. "Shortlisted models" PDF — compare with Nikhil's list, then finalise candidates.

### C. Flowgenix integration
10. Does any **production tenant** have `Tenant.LLMProvider/modelId/LLMUrl` set? (Must
    be "no" for the zero-prod-impact guarantee, or those tenants must be excluded.)
11. Confirm prod `.env` will never carry `AI_TARGET` / `LOCAL_AI_BASE_URL` /
    `LOCAL_AI_API_KEY`.
12. Agree the pilot is scoped to **structured field extraction only** (not
    classification / region / SAP / assistant / proposals).
13. Who on the Flowgenix side makes the ~3-line dev/test change, and on which
    branch/instance do we prove it?

### D. RAG service
14. Run the RAG analysis prompt (§8.2); get the full HTTP contract + provisioning steps.
15. (Then) does any *future* pilot product need retrieval, and does it go through this
    RAG service?

### E. This repo / process
16. Ratify implementation language (Open — clean + maintainable).
17. Repo host + PR reviewers + CI runner (GitHub Actions assumed).
18. Keep LiteLLM as a separate hop (Nikhil blessed it) vs fold into the middleware for
    MVP. Leaning **keep** — it already provides the whole §3.5 contract out of the box.

### F. Scope / timeline
19. Milestone-update cadence; go/no-go date; "the party" target.
20. Test-case count for the MVP: ~50–100 solid Flowgenix extraction cases, growing to
    300+ cross-product post-MVP — acceptable?

---

## 8. Analysis prompts

### 8.1 Flowgenix — DONE (report received 2026-08-31, folded into §5).

### 8.2 RAG service — PENDING (teammate to run against the RAG repo)

```
You are performing a READ-ONLY analysis of the existing standalone RAG service
codebase. Do not modify any file. Read architecture-overview.md, CLAUDE.md,
docs/adr-001-configuration-and-multi-tenancy.md, and docs/testing-plan.md first, then
verify every claim against the actual code. Cite file:line for everything. Where code
and docs disagree, follow the code and flag the drift. Say "UNKNOWN" rather than guess.

CONTEXT: We are building a separate model-agnostic "Local AI Server" (a gateway that
internal products call for LLM inference). Some products will need retrieval over their
own knowledge base. This RAG service ALREADY EXISTS and is independently deployed — we
will NOT rebuild or absorb it; our server will call its HTTP API. We need the exact
integration contract and operational constraints.

REPORT:
1. SERVICE OVERVIEW & DEPLOYMENT — entrypoint, ASGI server, ports, containerisation;
   every env var (RAG_MONGO_URI, RAG_ADMIN_TOKEN, RAG_SECRET_<REF>, RAG_LOG_LEVEL,
   others) with purpose/required/default; runtime deps (MongoDB, Chroma, Postgres/
   pgvector, Ollama, cloud LLM) and which are mandatory per config; how/where it is
   currently deployed; network exposure.
2. PUBLIC API CONTRACT — for every public route (GET /health, POST /v1/ingest,
   GET /v1/jobs/{id}, POST /v1/retrieve, POST /v1/ask, POST /v1/complete, + any other):
   method, path, full request schema (field, type, required?, default, constraints),
   full response schema, status codes, error body. Paste the pydantic models + route
   signatures. Auth per route + exact header format.
3. AUTH & MULTI-TENANCY — bearer key → project_id (hashed lookup): show code, header
   name, failure responses. How project scoping isolates the vector store, keyword
   index, metadata store (collection/index/table naming). Does one deployment serve
   many projects at once? Per-project rate limits / quotas?
4. ADMIN API — every admin route (project CRUD, key issue/revoke, config-version
   create/list/activate, suspend/reactivate, audit log): request/response, auth,
   side effects. The exact step-by-step sequence to provision a NEW project and get a
   working API key, with payloads.
5. PER-PROJECT CONFIG MODEL — overridable ports; how an override is written (MongoDB
   config docs, versioning, activation) — paste the config-version schema;
   adapter_settings shape (extra="forbid"); the embedding↔vector_store coupling rule;
   secret-ref pattern. What MUST be configured for a new project to answer queries vs
   what falls back to the default profile. ADR-001 open questions and current default
   behaviour: (a) changing embedding adapter without re-index, (b) fail-closed vs
   soft-fallback on config-DB outage.
6. INGESTION CONTRACT — accepted formats (PDF/TXT/CSV/DOCX/raw text; Excel/JSON
   excluded); HOW a document is submitted (inline text / multipart upload /
   presigned-URL fetch) — show the request model; document + chunk metadata schema;
   chunking params; the job model (async in-process queue), polling GET /v1/jobs/{id},
   terminal states, error reporting; re-ingest / version / delete semantics.
7. RETRIEVE vs ASK vs COMPLETE — exactly what each does, inputs/outputs; top_k, RRF,
   reranker stage, the "no relevant information" short-circuit, the shape of returned
   sources/citations. Which endpoint fits "give me the relevant chunks so the Local AI
   Server builds its own prompt" vs "do the whole answer here", and the trade-offs.
8. LLM USAGE INSIDE RAG — the LLMService port + Ollama/Claude/GPT adapters: does the
   RAG service itself call an LLM for /v1/ask and /v1/complete? Which model per
   project, configured how? If Ollama — what host/endpoint does it expect? Could it be
   pointed at our Local AI Server instead, and what would that require?
9. OPERATIONAL CHARACTERISTICS — latency (retrieval / rerank / answer), known slow
   paths (reranker latency noted as unmeasured — confirm); cold start (lazy container
   build, no warm-up); caching (ConfigCache TTL / stale-while-error, ContainerCache
   LRU) and their config; failure modes + status codes (config-DB down →
   stale-while-error / 503, invalid config → 503, suspended project, empty index);
   concurrency model (in-process asyncio queue) and its limits.
10. INTEGRATION FIT — concrete call flow: Local AI Server middleware → RAG service for
    a product that needs retrieval (endpoint, auth, request body, how a "Local AI
    product" maps to a "RAG project" + key). What must be provisioned per product and
    by whom. What is MISSING or unclear (batch retrieve? multi-project key? embedding
    mismatch risk? per-project health signal?).
11. SUMMARY — the exact HTTP contract we code against (endpoints + payloads); the
    per-product provisioning steps; operational constraints to design around;
    doc-vs-code drift; every remaining UNKNOWN.
```

---

## 9. Decisions log

_(record as `YYYY-MM-DD — decision — rationale`)_

- **2026-08-31** — Local AI Server is model-agnostic + provider-agnostic; models are
  bake-off candidates, never an architectural dependency. — teammate direction.
- **2026-08-31** — RAG service is the existing standalone product; we call it, never
  rebuild it. — teammate direction.
- **2026-08-31** — First product E2E = Flowgenix, scoped to the structured
  field-extraction scene, via its existing `LLMService` seam. — lowest-risk slice with
  a clean seam + reusable eval material (§5).
- **2026-08-31** — Flowgenix `has_rag: false`, evidence-backed (no retrieval anywhere
  in the codebase, §5.6).
- **2026-08-31** — No vision model for the pilot; Flowgenix is text-only (§5.2).
- **2026-08-31** — Repo scaffolded. Contract written (`docs/CONTRACT.md`), architecture
  (`docs/architecture.md`), ADR 0001 (single-protocol OpenAI-compatible). Structure:
  `gateway/` (LiteLLM config), `middleware/` (Node/ESM/Express skeleton — identity,
  profileLoader, ragClient, validate, logger, upstream, index, mint-key tool),
  `profiles/flowgenix.json` (all-off), `eval/` (promptfoo + deepeval skeletons +
  dataset guide), `deploy/` (compose + launchd plist), CI regression workflow (disabled).
- **2026-08-31 (provisional)** — Middleware language = Node.js 20 / ESM / Express.
  Reversible; matches the PDFs + Flowgenix backend + Promptfoo. Ratify at E-16.
- **2026-08-31 (provisional)** — Keep LiteLLM as a pinned separate hop (E-18) — it
  already delivers the whole `CONTRACT.md`. Confirm.

- **2026-09-02** — Interim dev host = Windows/NVIDIA desktop (16 GB RAM, 16 GB VRAM);
  Ollama native on Windows (installer service, not launchd); middleware run natively,
  only LiteLLM in Docker (16 GB RAM is tight). Model strategy re-tiered by VRAM budget,
  bake-off designed around the 8 GB target so results transfer. — ADR 0002. Config +
  `scripts/dev-up.ps1` + `scripts/smoke-test.ps1` added; keys minted.

- **2026-09-02** — `docs/SETUP.md` written: platform-independent local-setup guide
  (prereqs, config, model tiers, split vs all-Docker bring-up, smoke test, eval,
  troubleshooting) so any contributor with the disk + a GPU can run the full stack
  pre-hardware. README quickstart + `deploy/README.md` point to it.

Pending: Gemini-compatible surface yes/no (B-8, recommend no — ADR 0001) ·
ratify language (E-16) · ratify LiteLLM hop (E-18) · RAG contract (D-14) ·
teammate sign-off on `docs/CONTRACT.md` (B-4).

---

## 10. Next steps — recommendation

**Why this order:** the Flowgenix report gives us (a) a precise, product-agnostic
contract we can freeze now, and (b) a single low-risk landing spot — the extraction
scene, already behind a clean provider-neutral seam, with reusable eval material. That
lets us hit "one product end-to-end in 2 weeks" without waiting on any Flowgenix
refactor, the Gemini-surface decision, the Mac, or the AI Email Agent. Everything
except the bake-off is hardware-independent.

1. ✅ **Contract written** = `docs/CONTRACT.md` (+ `docs/architecture.md`, ADR 0001).
   → **needs teammate sign-off (B-4).**
2. ✅ **Gemini-surface decision recorded** = ADR 0001: no Gemini route in the MVP,
   server stays single-protocol. Still TODO: verify whether LiteLLM can expose a
   Google-AI route cheaply (doesn't block MVP).
3. ✅ **Repo scaffolded** — `gateway/`, `middleware/` (Node skeleton, syntax-clean),
   `profiles/flowgenix.json` (all-off), `eval/`, `deploy/`, CI workflow (disabled).
4. ✅ **Whole path proven against a stand-in model** (2026-09-02, on the interim
   Windows box — ADR 0002). `scripts/dev-up.ps1` + `scripts/smoke-test.ps1`: 6/6 checks
   pass — `/health`, JSON-mode `/v1/chat/completions` returning valid JSON, `usage`
   populated, logical-model echo, 401 bad key, 400 unknown model. Content-free request
   log confirmed. Stand-in model still `llama3.2:3b`; bump to a 7–14B per ADR 0002 next.
   Scaffold bugs fixed en route: (a) middleware never loaded `middleware/.env` → added
   `node --env-file-if-exists`; (b) `gateway/config/litellm_config.yaml` used `${VAR}`,
   which LiteLLM 1.55.8 does not expand → `os.environ/VAR`, with the runtime prefix
   composed in `docker-compose.yml`; (c) LiteLLM `ollama/` provider throws
   `KeyError: 'name'` in JSON mode → switched to `ollama_chat/`.
   Still TODO: exercise timeout (408) and queue-full (429/503) paths under load.
5. ✅ **Flowgenix extraction bake-off set built + first model run** (2026-09-02).
   Bundle received (`C:\Users\91882\Flowgenix\celeris-eval-export`, Flowgenix
   `reddot-poc` @ `704e91bf`): 15 cases (10 invoice / 5 PO, 4 adversarial), real
   Google Document AI OCR, hand-verified goldens, JSON-schema, normalization + error
   contract, all fictional/scrubbed. Imported via `scripts/import-flowgenix-bundle.ps1`
   → `eval/datasets/flowgenix/`. Harness rewired to consume the bundle's `golden.json`
   directly (both promptfoo and deepeval). **`qwen2.5:7b` AND `qwen2.5:14b` both 8/15,
   0 JSON / 0 schema failures, POs 5/5 perfect, both fail invoice `line_items` on all 7
   non-trivial invoices — identical failure (55 flat top-level keys, no array). 14B
   ~2x slower (up to 103 s/call, over the 90 s default).** Conclusion: model size is
   not the lever — the **invoice prompt** under-specifies the `line_items` array (the
   PO prompt has an explicit schema block; POs pass on both). Write-up + recommendation:
   `docs/bake-off.md`. Actions: (a) Flowgenix adds a `line_items` array instruction to
   the invoice prompt; (b) re-run `qwen2.5:7b` after that; (c) keep the primary in the
   7-8B tier (14B latency unsafe for Flowgenix's synchronous path) — the 8 GB target
   box is likely sufficient after all.
6. ⏭ **Send Flowgenix a minimal change request** (not a refactor): the ~3-line
   dev/test integration + new env vars (§5.7), plus guardrail questions C-10 (any prod
   tenant with an LLM override?) and C-11 (prod `.env` will not carry the new vars).
7. ⏭ **Run the RAG analysis prompt** (§8.2) in parallel — for later products.
8. ⏭ **When the Mac lands:** bake-off one model at a time against the Flowgenix suite;
   pick on measured numbers; document; flip `AI_TARGET=local` on a dev Flowgenix
   instance; prove end-to-end.
9. ⏭ **Draft "first version of the plan" for Nikhil** — POC approach, model eval,
   infra, security, deployment, benchmarks, responsibilities, timelines — from this log.
