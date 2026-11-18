/**
 * promptfoo javascript assert: does the (post-transform) model output validate
 * against the Flowgenix extraction scoring schema?
 *
 * datasets/flowgenix/schema.json is JSON Schema draft 2020-12: envelope
 * { fields, lineItems } or NO_DATA. Needs ajv (npm install in eval/).
 */
const fs = require("fs");
const path = require("path");

let Ajv;
try {
  Ajv = require("ajv/dist/2020");
} catch {
  try {
    Ajv = require("ajv");
  } catch {
    Ajv = null;
  }
}

const schemaPath = path.resolve(__dirname, "../../datasets/flowgenix/schema.json");
let validate;

function getValidator() {
  if (validate !== undefined) return validate;
  if (!fs.existsSync(schemaPath)) {
    validate = null;
    return validate;
  }
  const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"));
  if (!Ajv) {
    validate = "no-ajv";
    return validate;
  }
  const ajv = new Ajv({ allErrors: true, strict: false });
  validate = ajv.compile(schema);
  return validate;
}

module.exports = (output) => {
  let data;
  try {
    data = JSON.parse(output);
  } catch (e) {
    return { pass: false, score: 0, reason: `output is not JSON: ${e.message}` };
  }

  const v = getValidator();
  if (v === null) {
    return { pass: false, score: 0, reason: `schema not found at ${schemaPath} — import the bundle` };
  }
  if (v === "no-ajv") {
    return data && typeof data === "object"
      ? { pass: true, score: 1, reason: "object (ajv not installed — shallow check)" }
      : { pass: false, score: 0, reason: "not an object" };
  }
  if (v(data)) return { pass: true, score: 1, reason: "schema ok" };
  return {
    pass: false,
    score: 0,
    reason: (v.errors || [])
      .slice(0, 6)
      .map((e) => `${e.instancePath || "/"} ${e.message}`)
      .join("; "),
  };
};
