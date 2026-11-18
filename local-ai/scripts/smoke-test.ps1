<#
  smoke-test.ps1 - prove the CONTRACT.md path end to end.
  Run in a second shell while dev-up.ps1 holds the middleware.

  Checks:
    1. GET  /health                       -> status ok
    2. POST /v1/chat/completions          -> 200, JSON-mode content parses, usage.* populated
    3. bad key                            -> 401
    4. unknown model                      -> 400
#>
param(
  [string]$BaseUrl = "http://localhost:4100",
  [string]$Key = "clr_flowgenix_6cf8063f7602104c4416a921dbab8d321add9982949d430d"
)
$ErrorActionPreference = "Stop"
$pass = 0; $fail = 0
function Ok($m)  { Write-Host "PASS  $m" -ForegroundColor Green; $script:pass++ }
function Bad($m) { Write-Host "FAIL  $m" -ForegroundColor Red;   $script:fail++ }

# 1. health
try {
  $h = Invoke-RestMethod "$BaseUrl/health" -TimeoutSec 5
  if ($h.status -eq "ok") { Ok "health: status=ok litellm=$($h.litellm)" } else { Bad "health: status=$($h.status)" }
} catch { Bad "health: $($_.Exception.Message)" }

# 2. chat completion in JSON mode
$body = @{
  model           = "celeris-local-primary"
  messages        = @(@{ role = "user"; content = 'Return exactly {"ok": true} as JSON.' })
  response_format = @{ type = "json_object" }
  temperature     = 0.1
} | ConvertTo-Json -Depth 6
try {
  $r = Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post -Headers @{ Authorization = "Bearer $Key" } -ContentType "application/json" -Body $body -TimeoutSec 120
  $content = $r.choices[0].message.content
  try { $null = $content | ConvertFrom-Json; Ok "chat: content is valid JSON  ($content)" } catch { Bad "chat: content not JSON: $content" }
  if ($r.usage.total_tokens -gt 0) { Ok "chat: usage populated (p=$($r.usage.prompt_tokens) c=$($r.usage.completion_tokens))" } else { Bad "chat: usage missing/zero" }
  if ($r.model -eq "celeris-local-primary") { Ok "chat: logical model echoed" } else { Bad "chat: model=$($r.model)" }
} catch { Bad "chat: $($_.Exception.Message)" }

# 3. bad key -> 401
try {
  Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post -Headers @{ Authorization = "Bearer nope" } -ContentType "application/json" -Body $body -TimeoutSec 10
  Bad "auth: bad key was accepted"
} catch {
  if ($_.Exception.Response.StatusCode.value__ -eq 401) { Ok "auth: bad key -> 401" } else { Bad "auth: bad key -> $($_.Exception.Response.StatusCode.value__)" }
}

# 4. unknown model -> 400
$bad = ($body | ConvertFrom-Json); $bad.model = "gpt-4o"; $bad = $bad | ConvertTo-Json -Depth 6
try {
  Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post -Headers @{ Authorization = "Bearer $Key" } -ContentType "application/json" -Body $bad -TimeoutSec 10
  Bad "model: unknown model was accepted"
} catch {
  if ($_.Exception.Response.StatusCode.value__ -eq 400) { Ok "model: unknown -> 400" } else { Bad "model: unknown -> $($_.Exception.Response.StatusCode.value__)" }
}

Write-Host "`n$pass passed, $fail failed" -ForegroundColor $(if ($fail) { "Red" } else { "Green" })
exit $fail
