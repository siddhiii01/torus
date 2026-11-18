/**
 * promptfoo output transform: pull the JSON object the model returned out of any
 * markdown fences / surrounding prose, so the downstream asserts see a clean JSON
 * string. If nothing parseable is found, return the raw output unchanged so the
 * `is-json` assert fails loudly instead of hiding the problem.
 */
module.exports = (output) => {
  if (typeof output !== "string") return output;
  let s = output.trim();

  const fence = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence) s = fence[1].trim();

  if (!s.startsWith("{")) {
    const first = s.indexOf("{");
    const last = s.lastIndexOf("}");
    if (first !== -1 && last > first) s = s.slice(first, last + 1);
  }

  try {
    return JSON.stringify(JSON.parse(s));
  } catch {
    return output;
  }
};
