# gateway/ — LiteLLM proxy

The routing + queue + token-accounting layer. Runs as the official LiteLLM Docker
image, pinned to an exact tag in `deploy/docker-compose.yml`.

- Config: [`config/litellm_config.yaml`](config/litellm_config.yaml)
- Listens on `:4000` (internal — only the middleware talks to it).
- Env it needs (from `.env`): `LITELLM_MASTER_KEY`, `OLLAMA_BASE_URL`,
  `LOCAL_PRIMARY_MODEL`, `LOCAL_FAST_MODEL`.

## Responsibilities

- Map `celeris-local-primary` / `celeris-local-fast` → `ollama/<model>`.
- `max_parallel_requests` concurrency cap; queue the rest; `429` when the queue is full.
- Token usage in every response (`usage.*`).
- `turn_off_message_logging: true` — never logs prompt/response text.

## Not its job

Per-product identity, profiles, RAG, output validation, structured per-product logs —
all middleware.

## Upgrades

LiteLLM has a CVE history (`PROJECT_LOG.md` §2.2 / Implementation Plan §8). Never
`:latest`. Bump the tag in `deploy/docker-compose.yml` only via a reviewed PR after
reading the changelog.
