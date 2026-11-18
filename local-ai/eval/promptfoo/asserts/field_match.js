/**
 * promptfoo javascript assert: score model extraction against golden expected
 * using the v1 dynamic envelope { fields, lineItems }.
 *
 * Gates:
 *   1. required_fields — header keys on `fields` must be non-null; `lineItems`
 *      must be a non-empty array of non-empty rows.
 *   2. key fields — ids / dates / currency / grand total when golden is non-null.
 *   3. line items — row count + per-row quantity + line amount/total.
 *
 * Adversarial cases always pass (informational). NO_DATA on adversarial also passes.
 */

const PLACEHOLDERS = new Set(["", "-", "--", "n/a", "na", "null", "none", "nil"]);
const AMOUNT_RE = /(amount|total|subtotal|tax|price|rate|charges|balance|net|gross|discount|paid|due|qty|quantity)/;
const DATE_RE = /(date|dated)/;
const LABEL_RE = /^(document_type|po_type|payment_terms|freight_terms|transport_mode)$/;

const CURRENCY_EQ = {
  $: "usd",
  usd: "usd",
  "us$": "usd",
  "₹": "inr",
  inr: "inr",
  rs: "inr",
  "rs.": "inr",
  "€": "eur",
  eur: "eur",
  "£": "gbp",
  gbp: "gbp",
};

function canonKey(k) {
  return String(k)
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function isNullish(v) {
  return v == null || (typeof v === "string" && PLACEHOLDERS.has(v.trim().toLowerCase()));
}

function remap(obj) {
  if (Array.isArray(obj)) return obj.map(remap);
  if (obj && typeof obj === "object") {
    const out = {};
    for (const [k, v] of Object.entries(obj)) out[canonKey(k)] = remap(v);
    return out;
  }
  return obj;
}

const WRAPPERS = new Set([
  "po",
  "purchase_order",
  "purchaseorder",
  "invoice",
  "bill",
  "document",
  "doc",
  "data",
  "result",
  "extraction",
  "extracted",
  "payload",
]);

function unwrap(obj) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return obj;
  const keys = Object.keys(obj);
  if (keys.length === 1 && WRAPPERS.has(keys[0]) && obj[keys[0]] && typeof obj[keys[0]] === "object") {
    return obj[keys[0]];
  }
  return obj;
}

function headerOf(obj) {
  if (!obj || typeof obj !== "object") return {};
  if (obj.fields && typeof obj.fields === "object" && !Array.isArray(obj.fields)) return obj.fields;
  return obj;
}

const LINE_KEYS = new Set(["line_items", "lineitems", "line_item", "lineitem", "items", "rows"]);
function lineItemsOf(obj) {
  if (!obj || typeof obj !== "object") return [];
  for (const [k, v] of Object.entries(obj)) {
    if (LINE_KEYS.has(k) && Array.isArray(v)) return v;
  }
  return [];
}
function rowIsEmpty(row) {
  return !row || typeof row !== "object" || Object.values(row).every(isNullish);
}

function num(v) {
  if (isNullish(v)) return null;
  const cleaned = String(v).replace(/[$€£₹,\s]/g, "");
  if (cleaned === "" || !/^-?\d*\.?\d+$/.test(cleaned)) return NaN;
  const n = Number(cleaned);
  return Number.isFinite(n) ? Math.round(n * 100) / 100 : NaN;
}
const MONTHS = {
  jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4,
  may: 5, jun: 6, june: 6, jul: 7, july: 7, aug: 8, august: 8, sep: 9, sept: 9,
  september: 9, oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12,
};

function ymd(y, m, d) {
  const yi = Number(y);
  const mi = Number(m);
  const di = Number(d);
  if (!yi || mi < 1 || mi > 12 || di < 1 || di > 31) return null;
  return `${String(yi).padStart(4, "0")}-${String(mi).padStart(2, "0")}-${String(di).padStart(2, "0")}`;
}

/** Calendar date only — never Date.toISOString() (timezone shifts printed dates). */
function toYmd(raw) {
  const s = String(raw).trim().replace(/\s+/g, " ");
  let m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (m) return ymd(m[1], m[2], m[3]);
  m = s.match(/^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$/);
  if (m) return ymd(m[1], m[2], m[3]);
  m = s.match(/^(\d{1,2})[ \-]([A-Za-z]{3,9})[ \-](\d{2,4})$/);
  if (m) {
    const month = MONTHS[m[2].toLowerCase()];
    if (!month) return null;
    let year = Number(m[3]);
    if (year < 100) year += year <= 68 ? 2000 : 1900;
    return ymd(year, month, m[1]);
  }
  m = s.match(/^([A-Za-z]{3,9}) (\d{1,2}),? (\d{4})$/);
  if (m) {
    const month = MONTHS[m[1].toLowerCase()];
    return month ? ymd(m[3], month, m[2]) : null;
  }
  m = s.match(/^(\d{1,2}) ([A-Za-z]{3,9}) (\d{4})$/);
  if (m) {
    const month = MONTHS[m[2].toLowerCase()];
    return month ? ymd(m[3], month, m[1]) : null;
  }
  m = s.match(/^(\d{1,2})([/\-.])(\d{1,2})\2(\d{4})$/);
  if (m) {
    const a = Number(m[1]);
    const b = Number(m[3]);
    const year = Number(m[4]);
    if (a > 12) return ymd(year, b, a); // 22/03/2026 → day-first
    if (b > 12) return ymd(year, a, b); // 03/22/2026 → month-first
    // Ambiguous (both ≤12): prefer day-first (INR / Flowgenix printed dates).
    // Callers that need both readings use dateCandidates().
    return ymd(year, b, a);
  }
  return null;
}

/** All plausible YYYY-MM-DD readings (ambiguous slash/dot gets day-first + month-first). */
function dateCandidates(raw) {
  const s = String(raw).trim().replace(/\s+/g, " ");
  const out = new Set();
  const primary = toYmd(raw);
  if (primary) out.add(primary);
  const m = s.match(/^(\d{1,2})([/\-.])(\d{1,2})\2(\d{4})$/);
  if (m) {
    const a = Number(m[1]);
    const b = Number(m[3]);
    const year = Number(m[4]);
    if (a <= 12 && b <= 12) {
      const dayFirst = ymd(year, b, a);
      const monthFirst = ymd(year, a, b);
      if (dayFirst) out.add(dayFirst);
      if (monthFirst) out.add(monthFirst);
    }
  }
  return [...out];
}

function datesEqual(a, b) {
  if (str(a).toLowerCase() === str(b).toLowerCase()) return true;
  const ca = dateCandidates(a);
  const cb = dateCandidates(b);
  if (!ca.length || !cb.length) return false;
  return ca.some((x) => cb.includes(x));
}
function str(v) {
  return isNullish(v) ? null : String(v).trim().replace(/\s+/g, " ");
}

function eqScalar(key, a, b) {
  if (isNullish(a) && isNullish(b)) return true;
  if (isNullish(a) || isNullish(b)) return false;
  if (AMOUNT_RE.test(key)) {
    const na = num(a);
    const nb = num(b);
    if (typeof na === "number" && typeof nb === "number" && !Number.isNaN(na) && !Number.isNaN(nb)) {
      return Math.abs(na - nb) <= 0.01;
    }
    return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
  }
  if (DATE_RE.test(key)) {
    return datesEqual(a, b);
  }
  if (key === "currency") {
    const ca = CURRENCY_EQ[str(a).toLowerCase()] || str(a).toLowerCase();
    const cb = CURRENCY_EQ[str(b).toLowerCase()] || str(b).toLowerCase();
    return ca === cb;
  }
  if (LABEL_RE.test(key)) return str(a).toLowerCase() === str(b).toLowerCase();
  return str(a) === str(b);
}

function lineTotalOf(row) {
  if (!row) return null;
  if (!isNullish(row.line_total)) return row.line_total;
  if (!isNullish(row.line_amount)) return row.line_amount;
  if (!isNullish(row.amount)) return row.amount;
  return null;
}

const fs = require("fs");
const path = require("path");
const GOLDEN_PATH = path.resolve(__dirname, "../../datasets/flowgenix/golden.json");
let _golden;
function goldenFor(name) {
  if (!_golden) _golden = JSON.parse(fs.readFileSync(GOLDEN_PATH, "utf8"));
  return _golden.find((g) => g.name === name);
}

const KEY_FIELDS = ["po_number", "po_date", "invoice_number", "invoice_date", "currency", "total_amount"];

module.exports = (output, context) => {
  const name = ((context && context.vars) || {}).case_name;
  const g = name && goldenFor(name);
  if (!g) return { pass: false, score: 0, reason: `no golden entry for case_name=${JSON.stringify(name)}` };

  const expectedRaw = g.expected;
  const required = (Array.isArray(g.required_fields) ? g.required_fields : []).map(canonKey);
  const adversarial = !!g.adversarial;

  if (!expectedRaw) return { pass: false, score: 0, reason: "golden entry has no `expected`" };

  let got;
  try {
    got = JSON.parse(output);
  } catch (e) {
    return { pass: adversarial, score: 0, reason: `output is not JSON: ${e.message}` };
  }

  if (got && got.error === "NO_DATA") {
    const reason = "NO_DATA envelope";
    return adversarial
      ? { pass: true, score: 1, reason: `${reason} [adversarial]` }
      : { pass: false, score: 0, reason };
  }

  const expRoot = unwrap(remap(expectedRaw));
  const modRoot = unwrap(remap(got));
  const exp = headerOf(expRoot);
  const mod = headerOf(modRoot);

  const missingRequired = [];
  for (const rk of required) {
    if (rk === "line_items" || rk === "lineitems") {
      if (lineItemsOf(modRoot).filter((r) => !rowIsEmpty(r)).length === 0) missingRequired.push("lineItems");
    } else if (isNullish(mod[rk])) {
      missingRequired.push(rk);
    }
  }
  const requiredGate = missingRequired.length === 0;

  const checks = [];
  const misses = [];
  for (const kf of KEY_FIELDS) {
    if (!(kf in exp) || isNullish(exp[kf])) continue;
    const ok = eqScalar(kf, exp[kf], mod[kf]);
    checks.push(ok);
    if (!ok) misses.push(`${kf}: want ${JSON.stringify(exp[kf])}, got ${JSON.stringify(mod[kf])}`);
  }

  const expLines = lineItemsOf(expRoot).filter((r) => !rowIsEmpty(r));
  const modLines = lineItemsOf(modRoot).filter((r) => !rowIsEmpty(r));
  // page1_subset: gold is a recoverable subset (e.g. multi-page). Every gold row
  // must match some unused model row (qty + line total); extra model rows OK.
  const subsetMode =
    g.lineItemsGoldScope === "page1_subset" || g.lineItemsGoldScope === "subset";
  if (expLines.length) {
    if (subsetMode) {
      const used = new Set();
      for (let i = 0; i < expLines.length; i++) {
        const e = expLines[i];
        const eTot = lineTotalOf(e);
        let found = -1;
        for (let j = 0; j < modLines.length; j++) {
          if (used.has(j)) continue;
          const m = modLines[j];
          const qOk = eqScalar("quantity", e.quantity, m.quantity);
          const tOk = isNullish(eTot) || eqScalar("line_total", eTot, lineTotalOf(m));
          if (qOk && tOk) {
            found = j;
            break;
          }
        }
        const ok = found >= 0;
        checks.push(ok);
        if (ok) used.add(found);
        else {
          misses.push(
            `line gold[${i}] unmatched (qty=${JSON.stringify(e.quantity)}, total=${JSON.stringify(eTot)}); model has ${modLines.length} rows [subset]`,
          );
        }
      }
      if (modLines.length < expLines.length) {
        checks.push(false);
        misses.push(`line subset: model has fewer rows (${modLines.length}) than gold (${expLines.length})`);
      }
    } else {
      const countOk = modLines.length === expLines.length;
      checks.push(countOk);
      if (!countOk) misses.push(`line count: want ${expLines.length}, got ${modLines.length}`);
      for (let i = 0; i < Math.min(expLines.length, modLines.length); i++) {
        const e = expLines[i];
        const m = modLines[i] || {};
        const qOk = eqScalar("quantity", e.quantity, m.quantity);
        checks.push(qOk);
        if (!qOk) misses.push(`line[${i}].quantity: want ${JSON.stringify(e.quantity)}, got ${JSON.stringify(m.quantity)}`);
        const eTot = lineTotalOf(e);
        const mTot = lineTotalOf(m);
        if (!isNullish(eTot)) {
          const tOk = eqScalar("line_total", eTot, mTot);
          checks.push(tOk);
          if (!tOk) misses.push(`line[${i}].total: want ${JSON.stringify(eTot)}, got ${JSON.stringify(mTot)}`);
        }
      }
    }
  }

  const matched = checks.filter(Boolean).length;
  const score = checks.length ? matched / checks.length : requiredGate ? 1 : 0;
  const pass = adversarial ? true : requiredGate && score === 1;

  const bits = [
    `required ${required.length - missingRequired.length}/${required.length}`,
    `fields ${matched}/${checks.length}`,
  ];
  if (missingRequired.length) bits.push(`missing: ${missingRequired.join(", ")}`);
  if (misses.length) bits.push(misses.slice(0, 6).join(" | ") + (misses.length > 6 ? " …" : ""));
  if (adversarial) bits.push("[adversarial — see meta.json]");

  return { pass, score, reason: bits.join("; ") };
};
