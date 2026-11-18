/**
 * Assemble agenda-style extraction prompts for Flowgenix v1 eval cases.
 *
 * PO:     A + B + C_po + D + F + DOCUMENT CONTENT
 * Invoice: A + B + C_inv + F + DOCUMENT CONTENT  (no D)
 * Join with \n\n after filter(Boolean). No Gemini / OpenAI adapter wrap.
 *
 * Usage (from repo root):
 *   node eval/datasets/flowgenix/assemble.js
 */
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const CONST = path.join(ROOT, "constants");
const CASES = path.join(ROOT, "cases");

function readConst(name) {
  return fs.readFileSync(path.join(CONST, name), "utf8").replace(/\r\n/g, "\n").trimEnd();
}

const A = readConst("JSON_ENFORCEMENT_PREFIX.txt");
const F = readConst("JSON_ONLY_REMINDER.txt");
const D = readConst("PO_NUMBER_GUARDRAILS.txt");
const C_PO = readConst("C_PO.txt");
const C_INV = readConst("C_INVOICE.txt");
const pack = JSON.parse(fs.readFileSync(path.join(ROOT, "pack.json"), "utf8"));

function promptFor(docType) {
  const p = pack.prompts.find((x) => x.docType === docType);
  if (!p) throw new Error(`pack.json has no prompt for docType=${docType}`);
  return p;
}

function assemble(docType, ocrText) {
  const B = promptFor(docType).promptText;
  const C = docType === "Purchase Order" ? C_PO : C_INV;
  const parts = [
    A,
    B,
    C,
    docType === "Purchase Order" ? D : null,
    F,
    `DOCUMENT CONTENT:\n${ocrText.replace(/\r\n/g, "\n").replace(/\n+$/, "")}\n`,
  ];
  return parts.filter(Boolean).join("\n\n");
}

function main() {
  const dirs = fs
    .readdirSync(CASES, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();

  if (!dirs.length) throw new Error(`no cases under ${CASES}`);

  const golden = [];
  for (const name of dirs) {
    const dir = path.join(CASES, name);
    const ocr = fs.readFileSync(path.join(dir, "ocr.txt"), "utf8");
    const gold = JSON.parse(fs.readFileSync(path.join(dir, "gold.json"), "utf8"));
    const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
    const prompt = assemble(meta.docType, ocr);
    fs.writeFileSync(path.join(dir, "assembled_prompt.txt"), prompt, "utf8");

    const entry = {
      name,
      doc_type: meta.docType === "Purchase Order" ? "purchase_order" : "invoice",
      scene_prompt: meta.docType === "Purchase Order" ? "purchase_order" : "invoice",
      source: meta.source || pack.source,
      prompt_source: meta.prompt_source || meta.prompt_pack || "synthetic_assembly",
      adversarial: !!meta.adversarial,
      derived_from: meta.derived_from || meta.uploadId || "fictional",
      document_text: ocr.replace(/\r\n/g, "\n").replace(/\n+$/, "") + "\n",
      prompt,
      expected: gold,
      required_fields: meta.required_fields || [],
    };
    if (meta.bucket) entry.bucket = meta.bucket;
    if (meta.lineItemsGoldScope) entry.lineItemsGoldScope = meta.lineItemsGoldScope;
    if (meta.ocr_engine) entry.ocr_engine = meta.ocr_engine;
    golden.push(entry);
    console.log("assembled", name);
  }

  fs.writeFileSync(path.join(ROOT, "golden.json"), JSON.stringify(golden, null, 2) + "\n", "utf8");
  console.log(`wrote golden.json (${golden.length} cases)`);
}

main();
