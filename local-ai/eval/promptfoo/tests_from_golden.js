/**
 * promptfoo test generator — one test per entry in the imported Flowgenix bundle
 * (datasets/flowgenix/golden.json, produced by eval/datasets/flowgenix/assemble.js).
 *
 * Only two string vars are passed through — promptfoo serializes complex vars, so
 * the asserts re-read golden.json themselves, keyed by `case_name`.
 *
 * Optional filter: FLOWGENIX_EVAL_CASES=po-01,po-real-05,invoice-02
 */
const fs = require("fs");
const path = require("path");

const goldenPath = path.resolve(__dirname, "../datasets/flowgenix/golden.json");

module.exports = async () => {
  if (!fs.existsSync(goldenPath)) {
    throw new Error(
      `golden.json not found at ${goldenPath} — run node eval/datasets/flowgenix/assemble.js`,
    );
  }
  let golden = JSON.parse(fs.readFileSync(goldenPath, "utf8"));
  const filter = (process.env.FLOWGENIX_EVAL_CASES || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (filter.length) {
    const want = new Set(filter);
    golden = golden.filter((g) => want.has(g.name));
    if (!golden.length) {
      throw new Error(`FLOWGENIX_EVAL_CASES matched nothing: ${filter.join(",")}`);
    }
    console.error(
      `FLOWGENIX_EVAL_CASES: running ${golden.length} case(s): ${golden.map((g) => g.name).join(", ")}`,
    );
  }

  return golden.map((g) => ({
    description: g.name + (g.adversarial ? " [adversarial]" : ""),
    vars: {
      assembled_prompt: g.prompt,
      case_name: g.name,
    },
    assert: [
      { type: "is-json" },
      { type: "javascript", value: "file://asserts/schema_pass.js" },
      { type: "javascript", value: "file://asserts/field_match.js" },
      // Quality vs time are separate. Production Celeris cap is 115s.
      // Local 7B + real Document AI OCR often needs 5–15 min one-at-a-time.
      { type: "latency", threshold: 1200000 },
    ],
  }));
};
