# deploy/

> Running it on your own machine? Follow **[`../docs/SETUP.md`](../docs/SETUP.md)** —
> this file is the deployment-topology reference.

| Piece | Runs as | Why |
| --- | --- | --- |
| LiteLLM | Docker (`docker-compose.yml`), pinned tag | routing/queue/token accounting; isolated |
| Middleware | Docker (`docker-compose.yml`) | our custom layer |
| Ollama | **Native** on the Mac, via `com.celeris.localai.plist` (launchd) | needs MLX/Metal GPU — Docker on macOS can't pass the GPU through |

## Bring-up order (on the Mac)

1. Install Ollama natively; install the launchd plist; verify auto-restart.
2. `ollama pull` the bake-off candidates (one-time; run one at a time — 64 GB).
3. `cp .env.example .env` at repo root + `middleware/.env`; fill keys.
4. `docker compose -f deploy/docker-compose.yml up --build`.
5. `curl http://localhost:4100/health` → expect `status: ok`.

## Bring-up on a Windows + NVIDIA dev box (interim — ADR 0002)

16 GB RAM is tight, so split the stack: Ollama native, LiteLLM in Docker, middleware
native.

1. `winget install Ollama.Ollama` — the installer runs Ollama as a background service
   (no launchd equivalent needed). CUDA is used automatically.
2. `copy .env.example .env` (repo root) and `copy middleware\.env.example middleware\.env`;
   set a matching `LITELLM_MASTER_KEY` in both.
3. `copy middleware\.secrets\keys.example.json middleware\.secrets\keys.json`, then
   `node middleware\src\tools\mint-key.js flowgenix dev` and paste the `sha256` in.
4. `powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1`
   (pulls the model, starts LiteLLM via compose, `npm install`, starts the middleware).
5. In a second shell: `powershell -ExecutionPolicy Bypass -File scripts\smoke-test.ps1`.

Docker Desktop's CLI is sometimes not on PATH; `dev-up.ps1` finds it from the running
process. A future dedicated Linux box gets a `systemd` unit for Ollama instead.

## Network (later, teammate's track)

Bind the middleware to the VPN interface only; keep LiteLLM (`:4000`) and Ollama
(`:11434`) bound to localhost. No public port forwarding. Stable server name, never a
hardcoded LAN IP (`PROJECT_LOG.md` §1.3, §2.3).
