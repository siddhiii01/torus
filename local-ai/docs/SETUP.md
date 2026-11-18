# Local setup — run the Celeris Local AI Server on your own machine

Interim guide until dedicated hardware exists. Anyone with the disk space and a
usable GPU can stand the whole stack up locally and exercise the full request path.
Model quality depends on your VRAM — see
[`decisions/0002-interim-hardware-and-model-tiers.md`](decisions/0002-interim-hardware-and-model-tiers.md).

## What runs where

| Piece | Port | How |
| --- | --- | --- |
| Ollama (model runtime) | `11434` | **native** on the host — needs GPU access (Metal / CUDA) |
| LiteLLM proxy | `4000` | Docker container, pinned tag |
| Middleware (this repo) | `4100` | native `node` **or** a second container |

Only `:4100` is meant to be called. `:4000` and `:11434` stay on localhost.

## Prerequisites

- **Docker Desktop** (running).
- **Node.js ≥ 20** (`node --version`).
- **Ollama** — <https://ollama.com/download>. Installs a background service on Windows
  and macOS; on Linux run `ollama serve` (or a systemd unit).
- **Disk**: 3–20 GB per model.
- **RAM**: 16 GB works if you run the middleware natively (not a 2nd container).
- **GPU / VRAM**: 8 GB runs a 7–8B model; 16 GB runs 14B. CPU-only works but is slow.

## 1. Configure

```bash
# repo root
cp .env.example .env                          # (PowerShell: copy .env.example .env)
cp middleware/.env.example middleware/.env
cp middleware/.secrets/keys.example.json middleware/.secrets/keys.json
```

Edit **`.env`** and **`middleware/.env`** — set the **same** `LITELLM_MASTER_KEY` in
both (any string, e.g. `sk-local-dev-<random>`).

Mint a product key and register its hash:

```bash
node middleware/src/tools/mint-key.js flowgenix dev
```

Paste the printed `sha256` object into `middleware/.secrets/keys.json` → `keys[]`.
Keep the **raw** `clr_flowgenix_…` key — that's what a caller / the smoke test uses.

`.env`, `middleware/.env`, and `keys.json` are git-ignored. Never commit them.

## 2. Pull a model

```bash
ollama pull qwen2.5:7b        # primary — decent structured extraction, ~4.7 GB, fits 8 GB VRAM
ollama pull llama3.2:3b       # fast / fallback, ~2 GB
```

Set them in **`.env`**:

```
LOCAL_PRIMARY_MODEL=qwen2.5:7b
LOCAL_FAST_MODEL=llama3.2:3b
```

For a first run you can point both at `llama3.2:3b` to prove the path in seconds.

## 3. Start it

### Option A — split stack (Windows, or any RAM-limited box)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1
```

Pulls the model if missing, starts LiteLLM via compose, `npm install`s and runs the
middleware in the foreground. Leave the window open. `Ctrl+C` stops the middleware;
the script clears a stale `:4100` listener on the next run.

### Option B — everything in Docker (macOS / Linux / roomy machines)

Ollama still runs natively (it needs the GPU); LiteLLM **and** the middleware run as
containers:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up --build
```

`--env-file .env` is required — compose does not pick up the repo-root `.env` on its
own.

### Manual (either OS, no scripts)

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up -d litellm
npm --prefix middleware install
npm --prefix middleware start
```

## 4. Smoke test

**Windows:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke-test.ps1 -Key clr_flowgenix_YOUR_RAW_KEY
```

**Any OS (curl):**

```bash
curl -s http://localhost:4100/health

curl -s http://localhost:4100/v1/chat/completions \
  -H "Authorization: Bearer clr_flowgenix_YOUR_RAW_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"celeris-local-primary","messages":[{"role":"user","content":"Return {\"ok\":true} as JSON"}],"response_format":{"type":"json_object"}}'
```

Expect `status: ok`, a valid-JSON `content`, and a populated `usage` block. The first
model call is slow (10–40 s) while Ollama loads weights into VRAM.

## 5. Run the Flowgenix eval

```powershell
node eval/datasets/flowgenix/assemble.js
powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1 -View
```

See [`../eval/datasets/flowgenix/README.md`](../eval/datasets/flowgenix/README.md).
The archived v0 suite is `eval/datasets/flowgenix_v0_flat/`.

## Stopping

```bash
# Ctrl+C the middleware window first, then:
docker compose --env-file .env -f deploy/docker-compose.yml down
```

Ollama keeps running as a service; `ollama stop <model>` unloads a model from VRAM.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `docker: command not found` (Windows) | Docker Desktop's CLI isn't on PATH. `scripts\dev-up.ps1` finds it from the running process; manually: prepend `…\DockerDesktop\resources\bin` to `PATH`. |
| `EADDRINUSE :::4100` | A previous middleware is still running. `Get-Process node | Stop-Process -Force` (or kill the PID on 4100), then restart. |
| LiteLLM `LITELLM_MASTER_KEY ... not set` warning | You ran compose without `--env-file .env`. |
| Chat call → `401 Authentication Error, No connected db.` | LiteLLM's master key didn't resolve — the gateway config must use `os.environ/NAME`, not `${NAME}` (already fixed in this repo). Recreate the container: `... up -d --force-recreate litellm`. |
| Chat call → `500` / `KeyError: 'name'` in LiteLLM logs | The `ollama/` provider crashes in JSON mode; the config uses `ollama_chat/` (already fixed). Recreate the container. |
| Chat call → `502 upstream_error` | LiteLLM can't reach Ollama. Check `ollama list` works on the host; on Linux the container reaches the host via `OLLAMA_BASE_URL`. |
| Model call very slow | Model doesn't fit VRAM and is spilling to CPU. Use a smaller tag (ADR 0002). |
