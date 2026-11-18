/**
 * Env parsing + defaults + startup validation.
 * Fail fast and loud if a required var is missing.
 */

function req(name) {
  const v = process.env[name];
  if (!v || !v.trim()) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return v.trim();
}

function num(name, def) {
  const v = process.env[name];
  if (v === undefined || v === "") return def;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`Env var ${name} must be a number, got "${v}"`);
  return n;
}

export const config = Object.freeze({
  port: num("MIDDLEWARE_PORT", 4100),

  // Upstream LiteLLM
  litellmBaseUrl: process.env.LITELLM_BASE_URL?.trim() || "http://localhost:4000",
  litellmMasterKey: req("LITELLM_MASTER_KEY"),

  // Request budget. Callers that set no client timeout must not hang for minutes.
  defaultTimeoutMs: num("REQUEST_TIMEOUT_MS", 90_000),
  maxTimeoutMs: num("REQUEST_TIMEOUT_MS_MAX", 115_000),
  minTimeoutMs: 5_000,

  // Output limits (CONTRACT.md §2)
  defaultMaxTokens: num("DEFAULT_MAX_TOKENS", 2048),
  hardMaxTokens: num("HARD_MAX_TOKENS", 8192),

  // Where product keys live (hashed lookup). Git-ignored.
  keysFile: process.env.KEYS_FILE?.trim() || "./.secrets/keys.json",

  // Profiles dir (relative to repo root by default)
  profilesDir: process.env.PROFILES_DIR?.trim() || "../profiles",

  // Structured request log (JSON lines, content-free)
  requestLogFile: process.env.REQUEST_LOG_FILE?.trim() || "./logs/requests.log",

  logLevel: process.env.LOG_LEVEL?.trim() || "info",

  // Valid logical model names (CONTRACT.md §2). Kept here so an unknown name is a 400
  // at the edge, not a confusing upstream error.
  allowedModels: (process.env.ALLOWED_MODELS?.trim() || "celeris-local-primary,celeris-local-fast")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
});
