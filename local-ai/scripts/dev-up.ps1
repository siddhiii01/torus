<#
  dev-up.ps1 - bring the Celeris Local AI Server up on a Windows dev box.

  Layout for this box (16 GB RAM is tight, so keep the footprint small):
    - Ollama        : native on the host      (:11434)  - loads weights into VRAM
    - LiteLLM       : one Docker container     (:4000)   - pinned tag, from deploy/docker-compose.yml
    - Middleware    : native (node)            (:4100)   - avoids a second container + image build

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1
    powershell -ExecutionPolicy Bypass -File scripts\dev-up.ps1 -SkipMiddleware   # just infra
#>
param(
  [switch]$SkipMiddleware
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

# --- locate docker (Docker Desktop is often not on PATH) --------------------
$docker = "docker"
if (-not (Get-Command $docker -ErrorAction SilentlyContinue)) {
  $p = Get-Process docker -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Path
  if ($p) { $env:Path = (Split-Path $p) + ";" + $env:Path }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker CLI not found. Start Docker Desktop, or add its resources\bin to PATH."
}

# --- check Ollama ----------------------------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Ollama is not installed." -ForegroundColor Yellow
  Write-Host "  winget install Ollama.Ollama    (then reopen the shell)" -ForegroundColor Yellow
  throw "install Ollama first"
}

# read model tags from the root .env
$envFile = Join-Path $repo ".env"
if (-not (Test-Path $envFile)) { throw "missing $envFile (copy .env.example)" }
$primary = (Select-String -Path $envFile -Pattern '^\s*LOCAL_PRIMARY_MODEL\s*=\s*(.+)$').Matches.Groups[1].Value.Trim()
$fast    = (Select-String -Path $envFile -Pattern '^\s*LOCAL_FAST_MODEL\s*=\s*(.+)$').Matches.Groups[1].Value.Trim()

$have = (& ollama list | Out-String)
foreach ($m in @($primary, $fast) | Select-Object -Unique) {
  if ($have -notmatch [regex]::Escape($m)) {
    Write-Host "Pulling $m ..." -ForegroundColor Cyan
    & ollama pull $m
  } else {
    Write-Host "Ollama model present: $m" -ForegroundColor Green
  }
}

# --- LiteLLM container ---------------------------------------------------
# --env-file is explicit: compose otherwise looks for .env next to the compose
# file (deploy/), not at the repo root.
Write-Host "Starting LiteLLM (docker compose) ..." -ForegroundColor Cyan
# --force-recreate: cheap (~2s) and guarantees the container picks up the current
# .env — compose does not always detect an interpolated-var change on its own.
& docker compose --env-file (Join-Path $repo ".env") -f (Join-Path $repo "deploy\docker-compose.yml") up -d --force-recreate litellm

Write-Host "Waiting for LiteLLM :4000 ..." -NoNewline
for ($i = 0; $i -lt 40; $i++) {
  try {
    $r = Invoke-RestMethod -Uri "http://localhost:4000/health/liveliness" -TimeoutSec 2
    Write-Host " ok" -ForegroundColor Green; break
  } catch { Start-Sleep 2; Write-Host "." -NoNewline }
  if ($i -eq 39) { Write-Host ""; throw "LiteLLM did not come up - check: docker compose -f deploy/docker-compose.yml logs litellm" }
}

# --- middleware (native) ----------------------------------------------
if ($SkipMiddleware) {
  Write-Host "`nInfra up. Start the middleware yourself:  npm --prefix middleware start" -ForegroundColor Green
  return
}
# clear a stale middleware still holding :4100 (orphaned by a prior Ctrl+C)
$stale = Get-NetTCPConnection -LocalPort 4100 -State Listen -ErrorAction SilentlyContinue
if ($stale) {
  Write-Host "Port 4100 in use by PID $($stale.OwningProcess) - stopping it ..." -ForegroundColor Yellow
  $stale.OwningProcess | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  Start-Sleep 1
}

$mw = Join-Path $repo "middleware"
if (-not (Test-Path (Join-Path $mw "node_modules"))) {
  Write-Host "npm install (first run) ..." -ForegroundColor Cyan
  & npm --prefix $mw install
}
Write-Host "`nStarting middleware on :4100 (Ctrl+C to stop) ..." -ForegroundColor Green
& npm --prefix $mw start
