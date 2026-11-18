/**
 * Optional output validation.
 *
 * Only runs when a product's profile has a `schema`. For the Flowgenix MVP this is
 * OFF — Flowgenix runs its own Ajv check downstream (PROJECT_LOG.md §5). Kept here
 * because later products will want it.
 *
 * Policy (CONTRACT.md §5.3): validate message.content as JSON against the schema;
 * on failure do ONE retry with the validation error appended as a user message;
 * if it still fails, return the model's last output and let the caller decide
 * (annotate metadata, never a hard 5xx).
 */

import Ajv from "ajv";
import addFormats from "ajv-formats";

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);

const compiledCache = new WeakMap();

function compiled(schema) {
  let fn = compiledCache.get(schema);
  if (!fn) {
    fn = ajv.compile(schema);
    compiledCache.set(schema, fn);
  }
  return fn;
}

/**
 * @param {object|null} schema
 * @param {string} content  raw assistant message content
 * @returns {{ok:true,data:any} | {ok:false,errors:string}}
 */
export function checkOnce(schema, content) {
  if (!schema) return { ok: true, data: undefined };
  let data;
  try {
    data = JSON.parse(content);
  } catch (e) {
    return { ok: false, errors: `content is not valid JSON: ${e.message}` };
  }
  const validate = compiled(schema);
  if (validate(data)) return { ok: true, data };
  return {
    ok: false,
    errors: (validate.errors ?? []).map((e) => `${e.instancePath || "/"} ${e.message}`).join("; "),
  };
}

/** Build the retry message appended after a failed validation. */
export function retryMessage(errors) {
  return {
    role: "user",
    content: `Your previous response failed schema validation: ${errors}. Reply again with ONLY a corrected JSON object.`,
  };
}
