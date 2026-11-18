# eval/

Two different jobs, two tools (Implementation Plan §3.7).

| | Bake-off — `promptfoo/` | Regression gate — `deepeval/` |
| --- | --- | --- |
| Question | Which model do we run? | Did a profile/RAG change break something? |
| Runs | Once, at model selection (one model at a time — 64 GB) | On every `profiles/` change + nightly |
| Input | The collected Flowgenix extraction cases, all candidates | Golden subset, current model only |
| Judging | Deterministic first (schema pass, field match). Any LLM judge = **local model only**, configurable, **off by default** (Nikhil constraint #6). |

## Datasets

`datasets/flowgenix/` — v1 envelope suite (`{ fields, lineItems }`, fictional-eval-pin).
See its README. Assemble with `node eval/datasets/flowgenix/assemble.js`.

`datasets/flowgenix_v0_flat/` — archived 15-case flat-key bake-off (historical).

## Metrics recorded per bake-off run (Build Guide §8)

response quality vs expected fields · schema-validation success rate · hallucination
rate (manual sample or local judge) · latency (p50/p95) · concurrency behaviour (2–3
simultaneous) · CPU / memory utilisation.

Output: one model chosen, with a short written justification against the numbers —
not a subjective call.
