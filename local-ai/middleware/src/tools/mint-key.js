/**
 * Mint a product API key and print the line to add to .secrets/keys.json.
 * Usage:  node src/tools/mint-key.js <productId> [env]
 * Example: node src/tools/mint-key.js flowgenix dev
 *
 * The raw key is printed ONCE. Store it in the product's dev/test config
 * (e.g. LOCAL_AI_API_KEY). Only the sha256 goes in keys.json.
 */

import { randomBytes, createHash } from "node:crypto";

const [productId, env = "dev"] = process.argv.slice(2);
if (!productId) {
  console.error("usage: node src/tools/mint-key.js <productId> [env]");
  process.exit(1);
}

const raw = `clr_${productId}_${randomBytes(24).toString("hex")}`;
const sha256 = createHash("sha256").update(raw, "utf8").digest("hex");

console.log("\nRAW KEY (give to the product, store once):\n  " + raw + "\n");
console.log("Add this object to .secrets/keys.json -> keys[]:\n");
console.log(JSON.stringify({ productId, env, sha256, label: `${productId} ${env}` }, null, 2));
console.log();
