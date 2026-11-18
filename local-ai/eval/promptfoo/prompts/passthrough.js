/**
 * promptfoo prompt function: send the pre-assembled Flowgenix prompt verbatim as
 * the single `user` message (params.json: no system message).
 *
 * The prompt is agenda-assembled locally (A+B+C+D+F+OCR, no adapter wrap) and stored
 * in datasets/flowgenix/golden.json. No templating here — the string contains
 * characters Nunjucks would choke on.
 */
module.exports = (ctx) => {
  const vars = (ctx && ctx.vars) || ctx || {};
  return vars.assembled_prompt || "";
};
