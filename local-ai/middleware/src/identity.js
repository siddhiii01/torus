/**
 * Bearer key -> productId.
 *
 * The single source of truth for "which product is this". productId is NEVER read
 * from the request body and NEVER parsed out of the token text — the full key is
 * hashed and looked up in the key table.
 *
 * .secrets/keys.json shape:
 * {
 *   "keys": [
 *     { "productId": "flowgenix", "env": "dev", "sha256": "<hex of the raw key>", "label": "flowgenix dev" }
 *   ]
 * }
 * Generate a key + its hash with:  node src/tools/mint-key.js flowgenix dev
 */

import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { config } from "./config.js";

let table = null; // Map<sha256hex, {productId, env, label}>

export async function loadKeys() {
  const raw = await readFile(config.keysFile, "utf8").catch((e) => {
    throw new Error(`Cannot read keys file ${config.keysFile}: ${e.message}`);
  });
  const parsed = JSON.parse(raw);
  const next = new Map();
  for (const k of parsed.keys ?? []) {
    if (!k.productId || !k.sha256) continue;
    next.set(k.sha256.toLowerCase(), { productId: k.productId, env: k.env ?? "dev", label: k.label ?? k.productId });
  }
  if (next.size === 0) throw new Error(`No usable keys in ${config.keysFile}`);
  table = next;
  return next.size;
}

function sha256Hex(s) {
  return createHash("sha256").update(s, "utf8").digest("hex");
}

/**
 * @param {string|undefined} authHeader  raw "Authorization" header value
 * @returns {{productId:string, env:string, label:string} | null}
 */
export function resolveIdentity(authHeader) {
  if (!table) throw new Error("identity.loadKeys() not called");
  if (!authHeader || !authHeader.startsWith("Bearer ")) return null;
  const key = authHeader.slice("Bearer ".length).trim();
  if (!key) return null;
  return table.get(sha256Hex(key)) ?? null;
}
