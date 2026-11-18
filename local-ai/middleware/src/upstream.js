/**
 * Thin client to the LiteLLM proxy. One responsibility: forward an assembled
 * chat-completions request, enforce the timeout, normalise errors to CONTRACT.md §2.
 */

import { request } from "undici";
import { config } from "./config.js";

export class UpstreamError extends Error {
  constructor(httpStatus, code, message) {
    super(message);
    this.httpStatus = httpStatus;
    this.code = code;
  }
}

/**
 * @param {object} body   OpenAI chat-completions body (already assembled)
 * @param {number} timeoutMs
 * @returns {Promise<object>}  parsed OpenAI response
 */
export async function chatCompletion(body, timeoutMs) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  let res;
  try {
    res = await request(`${config.litellmBaseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.litellmMasterKey}`,
      },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new UpstreamError(408, "timeout", "generation timed out");
    throw new UpstreamError(502, "upstream_error", `cannot reach LiteLLM: ${e.message}`);
  } finally {
    clearTimeout(t);
  }

  const text = await res.body.text();
  if (res.statusCode === 200) {
    return JSON.parse(text);
  }

  // Map LiteLLM/OpenAI-ish upstream errors onto our contract.
  const code =
    res.statusCode === 429 ? "rate_limit_exceeded" :
    res.statusCode === 503 ? "model_unavailable" :
    res.statusCode === 400 ? "invalid_request" :
    res.statusCode === 401 ? "invalid_api_key" :
    "upstream_error";
  const httpStatus = ["rate_limit_exceeded", "model_unavailable", "invalid_request", "invalid_api_key"].includes(code)
    ? res.statusCode
    : 502;
  throw new UpstreamError(httpStatus, code, text.slice(0, 500));
}

export async function health() {
  try {
    const res = await request(`${config.litellmBaseUrl}/health/liveliness`, { method: "GET" });
    return res.statusCode < 500 ? "ok" : "down";
  } catch {
    return "down";
  }
}
