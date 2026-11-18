<#
  run-eval.ps1 - Flowgenix model bake-off via promptfoo.

  Talks straight to Ollama (:11434) - measures the model, not the gateway.
  Prereqs: Ollama running with the candidate model pulled; v1 golden built
  (node eval/datasets/flowgenix/assemble.js).

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1 -View
#>
param(
  [switch]$View,
  [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$pf = Join-Path $repo "eval\promptfoo"
$ds = Join-Path $repo "eval\datasets\flowgenix"

$need = @(
  (Join-Path $ds "golden.json"),
  (Join-Path $ds "schema.json")
)
$missing = $need | Where-Object { -not (Test-Path $_) }
if ($missing) {
  Write-Host "Missing - run: node eval/datasets/flowgenix/assemble.js" -ForegroundColor Red
  $missing | ForEach-Object { Write-Host "  $_" }
  exit 1
}

# Pin promptfoo in eval/package.json. Do NOT use promptfoo@latest — 0.122+
# requires Node >=22.22.0 and this machine is often on 22.16.x.
$evalDir = Join-Path $repo "eval"
$pfBin = Join-Path $evalDir "node_modules\.bin\promptfoo.cmd"
if (-not (Test-Path $pfBin)) {
  $pfBin = Join-Path $evalDir "node_modules\.bin\promptfoo"
}
if (-not $SkipInstall -and -not (Test-Path $pfBin)) {
  Write-Host "installing eval deps (ajv + promptfoo@0.120.27) ..." -ForegroundColor Cyan
  # cd into eval/ — npm --prefix breaks on paths with spaces on some Windows setups
  Push-Location $evalDir
  try {
    & npm install --no-fund --no-audit
  } finally {
    Pop-Location
  }
  $pfBin = Join-Path $evalDir "node_modules\.bin\promptfoo.cmd"
  if (-not (Test-Path $pfBin)) {
    $pfBin = Join-Path $evalDir "node_modules\.bin\promptfoo"
  }
}
if (-not (Test-Path $pfBin)) {
  Write-Host "promptfoo binary missing under eval/node_modules. Run: npm --prefix eval install" -ForegroundColor Red
  exit 1
}

Push-Location $pf
try {
  & $pfBin eval -c promptfooconfig.yaml
  if ($View) { & $pfBin view }
} finally {
  Pop-Location
}
