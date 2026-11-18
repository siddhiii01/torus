# Bake-off results — Flowgenix structured field-extraction

> Historical. This document describes the **v0 flat-key** suite now at
> `eval/datasets/flowgenix_v0_flat/`. Live eval is v1 (`eval/datasets/flowgenix/`,
> envelope `{ fields, lineItems }`). Re-run numbers after the v1 seeds before treating
> this table as current.

Suite (v0): 15 cases: 10 invoices, 5 POs. Run via `scripts/run-eval.ps1` → promptfoo → Ollama
direct (measures the model, not the gateway).

Machine: interim Windows box (ADR 0002) — Ryzen 5 7600X, RTX 50-series 16 GB VRAM.

---

## Summary — 2026-09-02

| Model | Pass | JSON valid | POs | Invoice line-items | Latency/call |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5:7b` (4.7 GB, 8 GB-class) | 8/15 | 15/15 | **5/5** | **0/7** — flattens the table | 13–49 s |
| `qwen2.5:14b` (9.0 GB, 16 GB-class) | 8/15 | 15/15 | **5/5** | **0/7** — flattens the table | 30–103 s |

**The 14B model does not fix the invoice line-item problem.** Identical pass/fail set,
identical failure mode (55 flat top-level keys instead of a `line_items: [...]` array),
and ~2× slower — several 14B calls exceed the 90 s default timeout, which in production
would be a `408`. Both models build the `line_items` array correctly for **purchase
orders** (5/5 each).

Since the same prompt yields perfect POs and broken invoices across two model sizes,
the cause is the **invoice prompt**, not model capacity. The PO prompt carries an
explicit inline SCHEMA block (`schema_notes.md`); the invoice prompt does not name a
`line_items` array with per-row keys.

### Recommendation

1. **Do not size up the model or the target box for this.** 7B ≈ 14B here; 7B is
   faster and fits the 8 GB target.
2. **Flowgenix: add a `line_items` array instruction to the invoice prompt** (mirror
   the PO prompt's schema block). This is the single highest-leverage change.
3. Re-run `qwen2.5:7b` after that change. If invoices then pass, `qwen2.5:7b` is the
   pick and the 8 GB-VRAM box is sufficient.
4. Keep the primary in the **7–8B** tier regardless — 14B latency (up to 103 s) is
   unsafe for Flowgenix's synchronous upload-handler path.

---

## `qwen2.5:7b` — run 2026-09-02 (clean, no cache)

**8 / 15 pass · 0 JSON failures · 0 schema violations · ~13–49 s per call**

| Group | Result |
| --- | --- |
| **JSON validity** | **15/15.** Never a fenced block, never invalid JSON, never a schema violation. The single biggest production risk (`INVALID_JSON_FORMAT`) does not materialise at this size. |
| **Purchase orders (po-01…05)** | **5/5, score 1.0.** Header, parties, GST/financial blocks, and `line_items` arrays all correct — including the flagship Consignor/Consignee layout (po-01), the 4-line/3-HSN PO (po-03), and the concatenated-duplicate adversarial (po-05). |
| **Invoice header fields** | Good. Invoice number, dates, currency, subtotal/tax/total mostly correct (field sub-score ~0.8). |
| **Invoice line items** | **Fails ~every non-trivial invoice (7/7 non-adversarial).** The model flattens the line-item table into **top-level scalars** (`item_description`, `quantity`, `unit_price`, `line_total`) instead of emitting a `line_items: [ … ]` array — one item where the document has 3–9. On invoice-06 it built the array in run 1 but not run 2 → **inconsistent** at temperature 0.1. |
| **Other invoice misses** | invoice-07 (no header invoice #, only a payment memo) → `invoice_number` null. |
| **Latency** | 40–49 s on multi-page invoices, 12–24 s on short ones. Under the 115 s cap but slow; several of these run synchronously inside Flowgenix's upload handler (`PROJECT_LOG.md` §5.9 #5). |

### Read

`qwen2.5:7b` is **viable for purchase orders and for invoice header extraction**, and
**not reliable for invoice line-item structure**. Because POs work and invoices don't —
against the same model, same machine — the likely cause is the **invoice prompt body**
under-specifying the `line_items` array shape (the PO prompt carries an explicit inline
SCHEMA block; per `schema_notes.md` the invoice prompt does not). Two things to try
before concluding the model is too small:

1. **Test `qwen2.5:14b`** on the identical suite (fits 16 GB VRAM; not the 8 GB target
   box — a positive result argues for a bigger target box, ADR 0002).
2. **Flag to Flowgenix**: does the production invoice prompt explicitly instruct a
   `line_items` array with per-row keys? If a one-line prompt change fixes 7B, no
   bigger model is needed. (Prompt is Flowgenix-owned; the bake-off runs it as-is.)

### How to run another candidate

Edit `eval/promptfoo/promptfooconfig.yaml` — comment the `qwen2.5:7b` provider,
uncomment one alternative — then:

```
ollama pull qwen2.5:14b
powershell -ExecutionPolicy Bypass -File scripts\run-eval.ps1 -View
```

---

## Harness notes

- Run 1 (2026-09-02, first attempt) reported 6/15 but reused stale cached provider
  outputs from an aborted run and had two scoring bugs (`quantity` not treated as
  numeric so `"5,000" ≠ 5000`; currency `$`/`₹` not accepted as `USD`/`INR` per
  `normalization.md` §3). Fixed in `eval/promptfoo/asserts/field_match.js`; run 2 is
  the clean baseline. Always `promptfoo cache clear` before a candidate's first run.
