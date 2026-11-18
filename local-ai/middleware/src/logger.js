/**
 * Structured logging. CONTENT-FREE — never the prompt or completion text.
 * (PROJECT_LOG.md §3.3, CONTRACT.md §5.4)
 *
 * Two sinks:
 *   - pino to stdout (ops)
 *   - one JSON line per request to REQUEST_LOG_FILE (the MVP dashboard tails this)
 */

import { createWriteStream, mkdirSync } from "node:fs";
import path from "node:path";
import pino from "pino";
import { config } from "./config.js";

export const log = pino({ level: config.logLevel });

mkdirSync(path.dirname(config.requestLogFile), { recursive: true });
const requestLog = createWriteStream(config.requestLogFile, { flags: "a" });

/**
 * @param {object} rec
 * @param {string} rec.productId
 * @param {string} rec.model            logical model name
 * @param {number} rec.latencyMs
 * @param {number} [rec.promptTokens]
 * @param {number} [rec.completionTokens]
 * @param {boolean} rec.ok
 * @param {number} rec.attempts
 * @param {string} [rec.errorCode]
 * @param {string} [rec.purpose]        from request metadata.purpose
 * @param {string} [rec.traceId]
 */
export function logRequest(rec) {
  const line = {
    ts: new Date().toISOString(),
    productId: rec.productId,
    model: rec.model,
    latency_ms: rec.latencyMs,
    prompt_tokens: rec.promptTokens ?? null,
    completion_tokens: rec.completionTokens ?? null,
    ok: rec.ok,
    attempts: rec.attempts,
    error_code: rec.errorCode ?? null,
    purpose: rec.purpose ?? null,
    trace_id: rec.traceId ?? null,
  };
  requestLog.write(JSON.stringify(line) + "\n");
  log[rec.ok ? "info" : "warn"](line, "request");
}
