# ADR 0001 — Model-agnostic, single-protocol (OpenAI-compatible) external contract

- Status: proposed (awaiting teammate sign-off)
- Date: 2026-08-31
- Context: `PROJECT_LOG.md` §3.5, §5

## Decision

The Local AI Server exposes exactly **one** external protocol: **OpenAI Chat
Completions** (`POST /v1/chat/completions`). Callers pass a **logical model name**;
the server maps it to whatever local model the bake-off selects. No product ever names
a model family, and no product integration changes when the model changes.

The server will **not** expose a Gemini-compatible (`…:generateContent`) surface in
v1.

## Why

- The Flowgenix analysis (`PROJECT_LOG.md` §5) shows its provider-neutral seam
  (`LLMService` + `openaiAdapter`) already accepts an OpenAI-compatible `baseUrl`.
  Routing the **structured field-extraction** scene needs ~3 lines, dev/test only,
  zero production impact.
- OpenAI Chat Completions is the de-facto standard; LiteLLM provides it, plus
  per-key identity, a concurrency queue, token accounting, and 429/503 semantics,
  out of the box — i.e. most of `CONTRACT.md` for free.
- A Gemini-shaped surface would exist only to serve ~6 Flowgenix call sites that
  hardcode Gemini's REST envelope. Consolidating those onto one AI client is
  Flowgenix's own modernisation, not this server's responsibility. Owning a
  Gemini-envelope translation layer adds surface (and a second response shape to
  keep correct) for one product's legacy pattern.
- Keeping to one protocol keeps the server both model-agnostic **and**
  protocol-clean.

## Consequences

- Products whose AI calls are not already OpenAI-compatible must introduce their own
  thin adapter (or point their existing OpenAI-compatible client at us). For
  Flowgenix that means the MVP covers the extraction scene only; full coverage is a
  Flowgenix-side refactor, out of our scope.
- Revisit post-MVP: check whether LiteLLM's proxy can expose a Google-AI-compatible
  route with negligible effort. If yes and a product needs it, reopen this ADR.

## Alternatives considered

- **Expose both OpenAI + Gemini surfaces now.** Rejected for MVP: more surface, a
  second response contract to guarantee, motivated by one product's legacy shape.
- **Fold LiteLLM into the middleware.** Rejected: LiteLLM already delivers the
  contract; Nikhil explicitly blessed "LiteLLM-based routing". Kept as a pinned,
  reviewed dependency.
