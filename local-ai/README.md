# celeris-local-ai

A shared, **model-agnostic** local AI server for Celeris product development and testing.

Products call one neutral, OpenAI-compatible endpoint. The server decides which local
model runs underneath. Swapping the model never changes a product's integration.

- **Not** a replacement for Cursor.
- **Not** a replacement for OpenAI/Gemini in production or customer demos.
- **For** making broad AI dev-testing cheap enough that teams stop rationing it.

See [`PROJECT_LOG.md`](PROJECT_LOG.md) for the full background, decisions, and open
questions. See [`docs/CONTRACT.md`](docs/CONTRACT.md) for the frozen external API and
[`docs/architecture.md`](docs/architecture.md) for how the pieces fit.

## Status

Scaffold. Pilot target: **Flowgenix structured field-extraction**, dev/test only,
via Flowgenix's existing `LLMService` seam. See `PROJECT_LOG.md` §10.

## Layout

| Path | What |
| --- | --- |
| `gateway/` | LiteLLM proxy config — routing, per-product keys, concurrency queue, token accounting |
| `middleware/` | Thin custom service in front of LiteLLM — identity, optional profile, optional RAG call, validation, structured logging |
| `profiles/` | One JSON per product — system prompt, output schema, few-shot, `has_rag` flag |
| `eval/` | Promptfoo (one-time model bake-off) + DeepEval (CI regression gate) |
| `deploy/` | docker-compose (LiteLLM + middleware) + launchd plist (Ollama) |
| `scripts/` | Windows dev helpers — `dev-up.ps1`, `smoke-test.ps1`, `import-flowgenix-bundle.ps1`, `run-eval.ps1` |
| `docs/` | Contract, architecture, **[`SETUP.md`](docs/SETUP.md)** (run it locally), decision records |

## Request path

```
product (dev/test config)
  → POST /v1/chat/completions        (middleware, this repo — port 4100)
      identity → [profile] → [RAG] → forward → [validate] → log
  → POST /v1/chat/completions        (LiteLLM — port 4000)
      route logical model → local model, concurrency queue, token usage
  → Ollama                           (native, MLX/Metal — port 11434)
```

## Local quickstart (pre-hardware, against a stand-in model)

Full step-by-step for any OS — prerequisites, config, model choice, troubleshooting —
is in **[`docs/SETUP.md`](docs/SETUP.md)**. The short version:

```bash
ollama pull qwen2.5:7b            # or llama3.2:3b to just prove the path
cp .env.example .env              # + middleware/.env + middleware/.secrets/keys.json; set a shared LITELLM_MASTER_KEY
node middleware/src/tools/mint-key.js flowgenix dev   # paste the sha256 into keys.json

# Windows (RAM-limited): Ollama native + LiteLLM container + middleware native
powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1
powershell -ExecutionPolicy Bypass -File scripts\smoke-test.ps1 -Key clr_flowgenix_YOUR_RAW_KEY

# macOS / Linux / roomy: everything in Docker (Ollama still native)
docker compose --env-file .env -f deploy/docker-compose.yml up --build
curl -s http://localhost:4100/health
```

Status (2026-09-02): full path proven locally on a Windows/NVIDIA box against
`qwen2.5:7b` — 6/6 smoke checks. See `PROJECT_LOG.md` §10.
