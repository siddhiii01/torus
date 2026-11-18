/**
 * Loads and caches profiles/<productId>.json.
 *
 * Profile shape (see profiles/README.md):
 * {
 *   "productId":   "flowgenix",
 *   "has_rag":     false,
 *   "systemPrompt": null,                 // string | null  — prepended as a system message if set
 *   "schema":       null,                 // JSON Schema | null — validated against message.content if set
 *   "fewShotExamples": [],                // [{role,content}] pairs, injected before the user message
 *   "rag": null,                          // { project, topK } — required iff has_rag
 *   "defaults": { "model": "celeris-local-primary", "temperature": 0.1 }
 * }
 *
 * Unknown product -> throws (caller returns 401/400). Malformed profile -> throws at
 * startup via validateAll().
 */

import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { config } from "./config.js";

const cache = new Map(); // productId -> profile

function assert(cond, msg) {
  if (!cond) throw new Error(`profile: ${msg}`);
}

function normalise(raw, productId) {
  const p = {
    productId,
    has_rag: raw.has_rag === true,
    systemPrompt: raw.systemPrompt ?? null,
    schema: raw.schema ?? null,
    fewShotExamples: Array.isArray(raw.fewShotExamples) ? raw.fewShotExamples : [],
    rag: raw.rag ?? null,
    defaults: {
      model: raw.defaults?.model ?? "celeris-local-primary",
      temperature: raw.defaults?.temperature ?? 0.1,
    },
  };
  assert(raw.productId === productId, `productId mismatch: file says "${raw.productId}", filename says "${productId}"`);
  assert(typeof p.has_rag === "boolean", "has_rag must be boolean");
  if (p.has_rag) {
    assert(p.rag && typeof p.rag.project === "string", "has_rag is true but rag.project is missing");
  }
  if (p.systemPrompt !== null) assert(typeof p.systemPrompt === "string", "systemPrompt must be string or null");
  if (p.schema !== null) assert(typeof p.schema === "object", "schema must be an object or null");
  return Object.freeze(p);
}

export async function getProfile(productId) {
  if (cache.has(productId)) return cache.get(productId);
  const file = path.join(config.profilesDir, `${productId}.json`);
  const raw = JSON.parse(await readFile(file, "utf8"));
  const profile = normalise(raw, productId);
  cache.set(productId, profile);
  return profile;
}

/** Load + validate every profile on the disk. Call at startup. */
export async function validateAll() {
  const entries = await readdir(config.profilesDir);
  const loaded = [];
  for (const name of entries) {
    if (!name.endsWith(".json") || name.startsWith("_")) continue;
    const productId = name.slice(0, -".json".length);
    cache.delete(productId);
    await getProfile(productId);
    loaded.push(productId);
  }
  return loaded;
}
