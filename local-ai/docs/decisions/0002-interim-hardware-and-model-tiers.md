# ADR 0002 — Interim dev hardware, and model tiers by VRAM budget

- Status: accepted (supersedes `PROJECT_LOG.md` §4's "64 GB Mac / 27–32B" assumption
  for the interim phase; §4 stays the reference for if/when the Mac lands)
- Date: 2026-09-02
- Context: `PROJECT_LOG.md` §0 (Hardware), §4 (Model strategy), ADR 0001

## Context

The 64 GB MacBook Pro in `PROJECT_LOG.md` §0 is **not confirmed** — the team may
instead buy a small Windows/NVIDIA box (quoted: RTX 4060 8 GB VRAM, 32 GB RAM, 1 TB
SSD), or nothing new for a while. Development must not block on that decision, and the
codebase must run on whatever is available.

Interim host = a contributor's desktop:

| Part | Value | Consequence |
| --- | --- | --- |
| CPU | AMD Ryzen 5 7600X (6c/12t) | fine |
| RAM | **16 GB** (often ~99% used) | **binding constraint.** Docker's WSL2 VM already reserves ~7 GB. |
| GPU | NVIDIA RTX 50-series, **16 GB VRAM** | runs 7–14B fully on GPU; comfortably ahead of the 8 GB target |
| Disk | 300 GB+ free NVMe | fine |
| Runtime | Ollama **native on Windows** (CUDA) | not launchd; see below |

## Decision

1. **Runtime is Ollama, invoked over its HTTP API — unchanged.** Metal/MLX vs CUDA vs
   CPU is Ollama's problem, not ours. No code or contract change. The only
   platform-specific pieces are (a) how Ollama is kept running and (b) which model tag
   fits.

2. **Auto-start is per-platform, and not on the critical path for dev:**
   - macOS: `deploy/com.celeris.localai.plist` (launchd) — as written.
   - Windows: the Ollama installer registers a background service; nothing to add for
     the dev box.
   - Linux (a future dedicated box): a `systemd` unit — add when that box exists.

3. **Model choice is entirely env-driven** (`LOCAL_PRIMARY_MODEL` / `LOCAL_FAST_MODEL`
   in the root `.env`, already the case). No model tag appears in code. The bake-off
   (`eval/promptfoo`) picks the real one later.

4. **Design the bake-off around the 8 GB-VRAM pessimistic target** so results transfer
   to whatever hardware is bought. The interim box may run a larger model for
   day-to-day work, but the *chosen* primary must fit 8 GB.

5. **Footprint rule for RAM-limited hosts:** run the middleware **natively** (Node),
   containerise **only LiteLLM**. The full two-container `docker-compose.yml` stays
   valid for the eventual server. `scripts/dev-up.ps1` implements the split.

## Model tiers by VRAM budget (interim; confirm exact Ollama tags at pull time)

| Budget | Primary (`celeris-local-primary`) | Fast (`celeris-local-fast`) | Notes |
| --- | --- | --- | --- |
| **Smoke test** | `llama3.2:3b` | `llama3.2:3b` | prove the path; ~2 GB; seconds per call |
| **8 GB VRAM** (target box) | 7–8B instruct, Q4_K_M (`qwen3:8b` / `llama3.1:8b` class) | 3B, Q4 | the tier the bake-off must select within |
| **16 GB VRAM** (interim box) | 14B instruct, Q4_K_M (`qwen3:14b` class) | 7–8B or 3B | usable for real dev; not the deploy target |
| CPU-only fallback | 3B, Q4 | 3B, Q4 | slow but keeps the path testable anywhere |

Structured-extraction quality at 7–8B is materially below the 27–32B tier the earlier
plan assumed. **Flag to the team:** the pilot's pass/fail numbers will partly measure
"is an 8 GB-class model good enough for Flowgenix extraction", not just "does the
platform work". If quality falls short, options are (in order) a bigger target box,
prompt+schema tightening in the profile, or the fine-tuning path already noted in
`PROJECT_LOG.md` §1.2(10).

## Consequences

- `PROJECT_LOG.md` §4.2 candidate table (Qwen 27–32B etc.) is **paused**, not deleted
  — it reactivates only if a ≥48 GB machine arrives.
- `eval/promptfoo/promptfooconfig.yaml` provider list should be re-pointed at 7–14B
  tags before the bake-off runs (currently `qwen3:32b` — won't fit).
- No change to `CONTRACT.md`, the middleware, the gateway config, or ADR 0001.
