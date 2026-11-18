<#
  import-flowgenix-bundle.ps1 - HISTORICAL. Copies a Flowgenix v0 flat-key export
  into eval/datasets/flowgenix_v0_flat/. Live bake-off uses eval/datasets/flowgenix/
  (v1 assemble.js). Do not overwrite v1 with this script.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\import-flowgenix-bundle.ps1 -BundlePath C:\Users\91882\Flowgenix\celeris-eval-export
#>
param(
  [Parameter(Mandatory)] [string]$BundlePath
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$ds = Join-Path $repo "eval\datasets\flowgenix_v0_flat"

if (-not (Test-Path $BundlePath)) { throw "bundle not found: $BundlePath" }
$B = (Resolve-Path $BundlePath).Path
foreach ($p in @("harness\golden.json", "schema\extraction.schema.json", "cases")) {
  if (-not (Test-Path (Join-Path $B $p))) {
    throw "bundle missing $p - point -BundlePath at the celeris-eval-export folder"
  }
}

New-Item -ItemType Directory -Force -Path `
  (Join-Path $ds "expected"), (Join-Path $ds "raw"), (Join-Path $ds "meta"), (Join-Path $ds "reference") | Out-Null

# --- golden.json (validate it parses + count) ---
$golden = Get-Content (Join-Path $B "harness\golden.json") -Raw | ConvertFrom-Json
Copy-Item (Join-Path $B "harness\golden.json") (Join-Path $ds "golden.json") -Force
Write-Host ("OK  golden.json  {0} cases" -f $golden.Count) -ForegroundColor Green

# --- schema ---
Get-Content (Join-Path $B "schema\extraction.schema.json") -Raw | ConvertFrom-Json | Out-Null
Copy-Item (Join-Path $B "schema\extraction.schema.json") (Join-Path $ds "schema.json") -Force
Write-Host "OK  schema.json" -ForegroundColor Green

# --- per-case expected / raw / meta ---
$n = 0
Get-ChildItem (Join-Path $B "cases") -Directory | Sort-Object Name | ForEach-Object {
  $name = $_.Name
  $doc = Join-Path $_.FullName "document.txt"
  $exp = Join-Path $_.FullName "expected.json"
  $met = Join-Path $_.FullName "meta.json"
  if (Test-Path $doc) { Copy-Item $doc (Join-Path $ds ("raw\" + $name + ".txt")) -Force }
  if (Test-Path $exp) { Copy-Item $exp (Join-Path $ds ("expected\" + $name + ".json")) -Force }
  if (Test-Path $met) { Copy-Item $met (Join-Path $ds ("meta\" + $name + ".json")) -Force }
  $n++
}
if (Test-Path (Join-Path $B "harness\expected")) {
  Copy-Item (Join-Path $B "harness\expected\*.json") (Join-Path $ds "expected") -Force
}
Write-Host ("OK  expected/ raw/ meta/  {0} cases" -f $n) -ForegroundColor Green

# --- reference material (not consumed by the harness, kept for humans) ---
foreach ($f in @("normalization.md", "error_contract.md", "params.json")) {
  if (Test-Path (Join-Path $B $f)) { Copy-Item (Join-Path $B $f) (Join-Path $ds "reference") -Force }
}
if (Test-Path (Join-Path $B "schema\schema_notes.md")) {
  Copy-Item (Join-Path $B "schema\schema_notes.md") (Join-Path $ds "reference") -Force
}
if (Test-Path (Join-Path $B "prompt")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $ds "reference\prompt") | Out-Null
  Copy-Item (Join-Path $B "prompt\*") (Join-Path $ds "reference\prompt") -Force
}
if (Test-Path (Join-Path $B "reusable\llm_json_repair_cases.json")) {
  Copy-Item (Join-Path $B "reusable\llm_json_repair_cases.json") (Join-Path $ds "reference") -Force
}
Write-Host "OK  reference/" -ForegroundColor Green

Write-Host ""
Write-Host "next:  powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1 -View" -ForegroundColor Cyan
