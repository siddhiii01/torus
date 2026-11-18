/**
 * Celeris Local AI Server — middleware.
 * Request lifecycle (docs/architecture.md): identity → profile → [rag] → forward →
 * [validate] → log. For the Flowgenix MVP the bracketed steps are off.
 */

import express from "express";
import { config } from "./config.js";
import { loadKeys, resolveIdentity } from "./identity.js";
import { getProfile, validateAll } from "./profileLoader.js";
import { checkOnce, retryMessage } from "./validate.js";
import { chatCompletion, health as litellmHealth, UpstreamError } from "./upstream.js";
import { retrieve } from "./ragClient.js";
import { log, logRequest } from "./logger.js";

const app = express();
app.use(express.json({ limit: "12mb" })); // ≥150k chars of prompt + overhead

function err(res, status, code, message) {
  return res.status(status).json({ error: { message, type: code, code } });
}

// ---- GET /health --------------------------------------------------------------
app.get("/health", async (_req, res) => {
  const litellm = await litellmHealth();
  const status = litellm === "ok" ? "ok" : "down";
  res.status(status === "down" ? 503 : 200).json({
    status,
    gateway: "ok",
    litellm,
    model_runtime: "unknown", // TODO: probe Ollama /api/tags via LiteLLM or directly
    queue_depth: 0,
    loaded_model: config.allowedModels[0],
    version: "0.1.0",
  });
});

// ---- POST /v1/chat/completions ----------------------------------------------
app.post("/v1/chat/completions", async (req, res) => {
  const started = Date.now();
  const traceId = req.get("x-celeris-trace-id") || cryptoRandom();
  let identity;

  try {
    identity = resolveIdentity(req.get("authorization"));
    if (!identity) return err(res, 401, "invalid_api_key", "missing or unknown API key");

    const body = req.body ?? {};
    const model = body.model;
    if (!config.allowedModels.includes(model)) {
      return err(res, 400, "invalid_request", `unknown model "${model}"; allowed: ${config.allowedModels.join(", ")}`);
    }
    if (body.n && body.n > 1) return err(res, 400, "invalid_request", "n>1 not supported");
    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      return err(res, 400, "invalid_request", "messages[] required");
    }

    const profile = await getProfile(identity.productId);

    // ---- assemble the upstream messages -------------------------------------
    let messages = [...body.messages];
    if (profile.systemPrompt) {
      messages = [{ role: "system", content: profile.systemPrompt }, ...messages];
    }
    if (profile.fewShotExamples.length) {
      const firstUser = messages.findIndex((m) => m.role === "user");
      const at = firstUser === -1 ? messages.length : firstUser;
      messages = [...messages.slice(0, at), ...profile.fewShotExamples, ...messages.slice(at)];
    }
    if (profile.has_rag) {
      // NOTE: provisional path — not exercised until the RAG report lands.
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      const { chunks } = await retrieve(profile.rag, lastUser?.content ?? "", process.env.RAG_PROJECT_KEY);
      const context = chunks.map((c) => c.text).join("\n---\n");
      messages = [{ role: "system", content: `Relevant context:\n${context}` }, ...messages];
    }

    // ---- timeout budget ---------------------------------------------------
    const hdrTimeout = Number(req.get("x-celeris-timeout-ms"));
    const timeoutMs = clamp(
      Number.isFinite(hdrTimeout) ? hdrTimeout : config.defaultTimeoutMs,
      config.minTimeoutMs,
      config.maxTimeoutMs,
    );

    const upstreamBody = {
      model,
      messages,
      temperature: body.temperature ?? profile.defaults.temperature,
      max_tokens: Math.min(body.max_tokens ?? config.defaultMaxTokens, config.hardMaxTokens),
      ...(body.response_format ? { response_format: body.response_format } : {}),
    };

    // ---- call + optional validate + one retry ---------------------------
    let attempts = 0;
    let completion = await chatCompletion(upstreamBody, timeoutMs);
    attempts++;
    let validationFailed = false;

    if (profile.schema) {
      const first = checkOnce(profile.schema, completion.choices?.[0]?.message?.content ?? "");
      if (!first.ok) {
        const retryBody = { ...upstreamBody, messages: [...messages, completion.choices[0].message, retryMessage(first.errors)] };
        completion = await chatCompletion(retryBody, timeoutMs);
        attempts++;
        const second = checkOnce(profile.schema, completion.choices?.[0]?.message?.content ?? "");
        validationFailed = !second.ok;
      }
    }

    const usage = completion.usage ?? {};
    logRequest({
      productId: identity.productId,
      model,
      latencyMs: Date.now() - started,
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      ok: true,
      attempts,
      purpose: body.metadata?.purpose,
      traceId,
    });

    // Echo the logical model name; annotate validation outcome without failing hard.
    completion.model = model;
    if (validationFailed) {
      completion.metadata = { ...(completion.metadata ?? {}), x_celeris_validation: "failed" };
    }
    return res.status(200).json(completion);
  } catch (e) {
    const isUpstream = e instanceof UpstreamError;
    const status = isUpstream ? e.httpStatus : 500;
    const code = isUpstream ? e.code : "internal_error";
    logRequest({
      productId: identity?.productId ?? "unknown",
      model: req.body?.model ?? "unknown",
      latencyMs: Date.now() - started,
      ok: false,
      attempts: 0,
      errorCode: code,
      purpose: req.body?.metadata?.purpose,
      traceId,
    });
    if (!isUpstream) log.error({ err: e, traceId }, "unhandled");
    return err(res, status, code, isUpstream ? e.message : "internal error");
  }
});

function clamp(n, lo, hi) {
  return Math.min(Math.max(n, lo), hi);
}
function cryptoRandom() {
  return Math.random().toString(36).slice(2, 12);
}

// ---- startup ---------------------------------------------------------------
const n = await loadKeys();
const profiles = await validateAll();
log.info({ keys: n, profiles }, "loaded");
app.listen(config.port, () => log.info(`middleware on :${config.port}`));
