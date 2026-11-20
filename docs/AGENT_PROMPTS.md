# Agent prompts for Torus phases

Copy a **task brief** into a new agent chat. Prefer **one task ID per agent**. Use a phase coordinator brief only to sequence or split work, not to implement an entire phase in one go.

Canonical rules: `AGENTS.md`. Task list and acceptance: `docs/IMPLEMENTATION_PLAN.md`. Recipes: `SKILLS.md`. Models/hardware: `docs/SETUP.md`.

---

## How to dispatch

1. Pick the next open task ID from `docs/IMPLEMENTATION_PLAN.md`.
2. Copy **Shared preamble** + the **task brief** (or phase coordinator + one task).
3. Fill the placeholders: `<BASE_BRANCH>`, `<TASK_IDS if several>`, any extra notes.
4. `@`-mention or open: `AGENTS.md`, the phase section of `IMPLEMENTATION_PLAN.md`, and the listed `SKILLS.md` recipes.
5. Tell the agent: first reply = plan (files + tests); then implement only the assigned IDs.

**Branch naming:** `agent/<task-id>-<short-slug>` (example: `agent/P0.3-core-models`).

**Parallelism:** Phases 0→1 sequential. Phases 2–7 may run in parallel if they only consume `torus.core` interfaces. Phase 8 needs 5–7. Phases 9–10 need 8.

---

## Shared preamble (paste first, every agent)

```text
You are an AI coding agent working in the Torus (Taurus) repository.

## Mission
Execute ONLY the task ID(s) named in the task brief below.
Do not start later tasks. Do not refactor unrelated packages.

## Required reading (do this before coding)
1. AGENTS.md (full — non-negotiable rules)
2. docs/IMPLEMENTATION_PLAN.md — only the phase section for your task
3. SKILLS.md — only the recipes listed in the brief
4. docs/SETUP.md — only if you enable or download a model
5. Code paths listed under In-scope / Depends on

## Product context (short)
Local engine: any PDF/image → fill the client's Excel template columns.
Confirmed values need independent agreeing signals; uncertain → review.
Quality target: zero silent errors (a wrong `confirmed` value).
Hardware is not a design constraint; readers that cannot run are skipped.

## Hard rules
- No hard-coded customer/vendor/bank/person/place names; no fixed page coordinates; no `page == 0` assumptions; no per-sample string fixes.
- Document/client knowledge lives in config (`config/`), not Python.
- Never commit or log real document content. Real data → TORUS_DATA_DIR. Tests → synthetic fixtures only.
- Depend on `torus.core` protocols/models only — not another stage's internals.
- Heavy imports (torch, paddle, transformers, …) only inside the reader/module that needs them.
- No network calls in the pipeline except configured local model servers.
- New top-level dependencies require human approval (stop and ask).
- Protocol or core schema changes: stop and ask.
- Feature branch from <BASE_BRANCH, default: dev-sid>. Never push to main unless a human asks. Commit only if the human asks.

## Execution workflow (AGENTS.md §6)
1. First reply: plan — files to create/change, interfaces touched, tests to add. Then wait is not required; implement after the plan in the same session unless blocked.
2. Tests first where practical, then code.
3. Run: ruff format + ruff check; mypy as required by AGENTS.md; pytest -m "not model".
4. If output can change: run evaluation and attach a summary table with NO document text.
5. Final report: what changed, commands run + results, risks, exact human check items.

## Definition of done (every task)
1. Tests written and passing (unit/component/integration as relevant). Model tests pass where installed.
2. Lint, format, and type checks clean.
3. Evaluation run if output can change; no silent-error increase; table in the report.
4. No hard-coded document knowledge, no real data, no document text in the diff.
5. Docs/config updated if behaviour changed.
```

---

## Phase coordinator briefs (optional)

Use when one agent should **split or sequence** a phase for others, not implement everything alone.

### Phase 0 — Foundation (coordinator)

```text
[PASTE SHARED PREAMBLE]

## Role
Phase 0 coordinator. Sequence tasks P0.1 → P0.10. Implement only if explicitly given a task ID; otherwise produce a dispatch order and handoff notes.

## Phase goal
Codebase where agents work safely and every change is tested.

## Order
P0.1 → P0.2 → P0.3 → P0.4 → P0.5 → P0.6 → P0.7 → P0.8 (STOP for owner approval on git-history purge) → P0.9 → P0.10.

## Skills
None specific beyond AGENTS.md; P0.7 → S8; P0.10 shaped like S1.

## Acceptance / human check
See IMPLEMENTATION_PLAN Phase 0.
```

### Phase 1 — Evaluation first (coordinator)

```text
[PASTE SHARED PREAMBLE]

## Role
Phase 1 coordinator. Sequence P1.1 → P1.6. Prefer one implementing agent per task.

## Depends on
Phase 0 complete (core models, config, synth fixtures, CI).

## Skills
S6 (eval/report). Port reference: legacy scoring / schemas/deb/row_key.md.

## Human gate
Developers must label ≥20 packs (5 hidden) — agent prepares format and tools; humans label.
```

### Phases 2–7 — Parallel track note

```text
[PASTE SHARED PREAMBLE]

## Role
You own ONLY the phase named below. Assume torus.core interfaces are frozen.
If you need a new field on Document/Page/Region/Reading/Candidate/FieldValue, STOP and ask — do not silently extend protocols.

## Parallel phases in flight
Other agents may own ingest, classify, layout, readers, consensus, validate.
Integrate only through torus.core and config.
```

### Phase 8+ coordinators

```text
[PASTE SHARED PREAMBLE]

## Role
Coordinator for Phase <N>. Do not implement the whole phase; dispatch task briefs P<N>.* in dependency order from IMPLEMENTATION_PLAN.
Phase 8 requires readers + consensus + validation usable.
Phases 9–10 require Phase 8 Excel output.
```

---

## Task briefs

For each task: paste **Shared preamble**, then the block below.

---

### Phase 0 — Foundation

#### P0.1 — Package layout and pyproject

```text
## Task
P0.1 — Create src/torus/ package and folder layout from AGENTS.md §3.1.
Add pyproject.toml (Python 3.12, base deps pinned, optional extras: paddle, doctr, vlm, train, review).

## Skills
—

## In-scope
src/torus/ (package skeleton + __init__), pyproject.toml, README install notes if required by existing docs style.

## Out-of-scope
Implementing stage logic; enabling models; moving real data.

## Depends on
—

## Produces
Installable package layout matching AGENTS.md §3.1.

## Stop and ask
New unexpected top-level dependencies beyond a minimal base set.
```

#### P0.2 — Tooling and CI

```text
## Task
P0.2 — ruff, mypy, pytest markers (model, slow), pre-commit, GitHub Actions: lint + types + non-model tests on push/PR.

## Skills
—

## In-scope
pyproject/tool configs, .pre-commit-config.yaml, .github/workflows/*, pytest.ini or pyproject pytest section.

## Out-of-scope
Application logic.

## Depends on
P0.1

## Produces
CI-green empty/skeleton test run.
```

#### P0.3 — torus.core models and protocols

```text
## Task
P0.3 — pydantic models: Document, Page, Region, Reading, Candidate, FieldValue, ExtractionResult.
Protocols: Reader, PageClassifier, Validator, FieldMapper. Typed errors. Unit tests.
mypy --strict on src/torus/core.

## Skills
—

## In-scope
src/torus/core/, tests/unit/ for core.

## Out-of-scope
Stage implementations.

## Depends on
P0.1

## Produces
Frozen interfaces for all later phases — treat as API.
```

#### P0.4 — torus.config

```text
## Task
P0.4 — typed loader for config/*.yaml with pydantic validation and config hash.
Starter: pipeline.yaml, readers.yaml, page_types.yaml, templates/.

## Skills
S1 (shape of field specs)

## In-scope
src/torus/config/, config/*.yaml starters, unit tests for load/validate/hash.

## Depends on
P0.3

## Produces
Validated config load used by every stage.
```

#### P0.5 — Metadata-only logger

```text
## Task
P0.5 — torus.logging: metadata-only logger with redaction. Test proves document text cannot reach log output.

## Skills
—

## In-scope
src/torus/logging.py (or logging/), tests that attempt to leak sample text and assert redaction.

## Depends on
P0.1

## Hard requirement
No document content in logs by default.
```

#### P0.6 — torus doctor

```text
## Task
P0.6 — CLI/doctor: detect OS, CPU, RAM, GPU, installed extras, model weights; print which readers can run.

## Skills
—

## In-scope
doctor command entrypoint, hardware probes, reader eligibility vs config/readers.yaml.

## Depends on
P0.4 (readers.yaml schema)

## Note
Must be usable on Windows and Linux.
```

#### P0.7 — Synthetic fixtures

```text
## Task
P0.7 — tests/fixtures/synth/ seeded generators: printed forms, comb rows, checkboxes, handwriting-like text (public fonts), rotation, fax-like degradation, stamps.

## Skills
S8

## In-scope
tests/fixtures/synth/, SOURCES.md for licences, unit tests for determinism.

## Out-of-scope
Committing generated images; real customer docs.

## Depends on
P0.3 (ground-truth shapes if returning Region/text)
```

#### P0.8 — Data safety

```text
## Task
P0.8 — Move data/deb/cases/* guidance to TORUS_DATA_DIR; tighten .gitignore; pre-commit blocks PDFs/images/xlsx/sensitive json under data paths.

## Skills
—

## In-scope
.gitignore, pre-commit hooks, docs note for TORUS_DATA_DIR layout, remove tracked customer artifacts from working tree if present.

## HARD STOP
Rewriting git history to purge past customer files needs explicit owner approval. Do NOT rewrite history unless the human message says the owner approved.

## Depends on
—

## Human check
Data-move plan approved.
```

#### P0.9 — Legacy move

```text
## Task
P0.9 — Move existing ocr/ and scripts/ into legacy/, mark read-only. Document which later phases port which parts. Do not extend legacy/.

## Skills
—

## In-scope
legacy/ move, short README in legacy/ mapping → future phases.

## Out-of-scope
Porting behaviour into src/torus/ (that is later phases).
```

#### P0.10 — deb_v1 template config

```text
## Task
P0.10 — Convert schemas/deb/excel_columns.json + field_groups.json into config/templates/deb_v1.yaml
(field specs: types, validators, look_in, cardinality, critical). Interim until client Excel (P8.1).

## Skills
S1

## In-scope
config/templates/deb_v1.yaml, loader acceptance tests.

## Read
schemas/deb/excel_columns.json, field_groups.json, row_key.md

## Out-of-scope
Implementing extractors for those columns.
```

---

### Phase 1 — Evaluation first

#### P1.1 — Test-set format

```text
## Task
P1.1 — Define case layout: TORUS_DATA_DIR/testsets/<set>/<case>/{source.pdf, expected.json, meta.json}
meta: split dev|hidden, tags (year, scan type, handwriting level).

## Skills
S6

## In-scope
Schema/docs for testset format, validation helper, synthetic mini-set under tests (not real PDFs in git).

## Depends on
P0.4, P0.10 (column keys)
```

#### P1.2 — torus.eval scoring

```text
## Task
P1.2 — Port legacy deb_score into torus.eval. Row alignment (row_key.md → configurable per template). Match rules exact|normalised|fuzzy. Metrics: accuracy, silent errors, status rates, automatic rate.

## Skills
S6

## In-scope
src/torus/eval/, unit tests with synthetic expected vs predicted (no real IDs).

## Read
legacy scoring / schemas/deb/row_key.md, field_groups.json

## Depends on
P1.1, P0.10
```

#### P1.3 — Report writer

```text
## Task
P1.3 — Markdown + HTML eval reports: per-column, per-page-type, timing, model/config versions. NO document text.

## Skills
S6

## In-scope
report writer + tests that assert absence of raw field values when forbidden fixtures used.

## Depends on
P1.2
```

#### P1.4 — eval diff gate

```text
## Task
P1.4 — torus.eval diff: compare two runs; fail if silent errors rise or a column drops beyond noise margin (config/pipeline.yaml).

## Skills
S6

## Depends on
P1.2, P1.3, P0.4
```

#### P1.5 — Labelling tool

```text
## Task
P1.5 — Local labelling tool: open PDF, fill template columns by hand, save expected.json. No cloud upload.

## Skills
—

## In-scope
Local-only UI or CLI under review/eval tooling; writes only under TORUS_DATA_DIR.

## Depends on
P1.1, P0.10
```

#### P1.6 — Baseline

```text
## Task
P1.6 — Run legacy pipeline through the new harness on existing cases; record starting numbers. PR summary table only (no document text).

## Skills
S6

## Depends on
P1.1–P1.4, P0.9 (legacy available)

## Note
Real cases live outside git; do not commit reports that embed document text.
```

---

### Phase 2 — Ingest and page clean-up

```text
## Phase ownership (all P2.*)
In-scope: src/torus/ingest/, src/torus/preprocess/
Skills: S8
Depends on: P0.3 Page model, P0.4 pipeline dpi/thresholds, P0.7 synth
Acceptance: component tests on rotated/skewed/degraded/blank synth pages.
Do not silently drop blank pages.
Human check (human, not you): contact sheet before/after on dev set.
```

#### P2.1 — Ingest PDF/images

```text
## Task
P2.1 — ingest/: PDF (PyMuPDF), jpg/png/tiff/multi-page tiff, EXIF rotation. Render at config dpi. Use embedded text only if quality check says usable.

## Skills
S8

## In-scope
src/torus/ingest/

## Depends on
P0.3, P0.4
```

#### P2.2 — Orientation

```text
## Task
P2.2 — Orientation 0/90/180/270 including handwriting-only pages. Classic detector + small classifier; if they disagree, keep both orientations for reading.

## Skills
S8

## In-scope
preprocess orientation; component tests with synth rotations.

## Depends on
P2.1
```

#### P2.3 — Deskew / perspective / crop

```text
## Task
P2.3 — Deskew, photo perspective correction, border/shadow cropping.

## Skills
S8

## Depends on
P2.1
```

#### P2.4 — Quality repair

```text
## Task
P2.4 — Fax-like/faded repair: stroke thicken, denoise, contrast norm. Each step switchable in config and measured.

## Skills
S8

## Depends on
P0.4, P2.1
```

#### P2.5 — Blank detection

```text
## Task
P2.5 — Blank/near-blank detection (ink density + OCR check). Record blanks; never silently drop.

## Skills
S8

## Depends on
P2.1
```

#### P2.6 — Page quality score

```text
## Task
P2.6 — Blur/contrast/resolution score on Page for later confidence/routing.

## Skills
—

## Depends on
P0.3 Page fields, P2.1
```

---

### Phase 3 — Page classification and pack splitting

```text
## Phase ownership (all P3.*)
In-scope: src/torus/classify/, config/page_types.yaml
Skills: S4
Depends on: clean pages (P2), P0.3 PageClassifier protocol
Acceptance: ≥98% page-type accuracy on dev set; misclassifications listed by type.
```

#### P3.1 — page_types.yaml

```text
## Task
P3.1 — Populate config/page_types.yaml with descriptions (application form, continuation, verification, ID types, bank statement, memo, checklist, brochure/terms, photo/signature, blank, other).

## Skills
S4

## In-scope
config/page_types.yaml only (plus loader tests if needed).

## Out-of-scope
Hard-coding type names in Python conditionals beyond reading config.
```

#### P3.2 — Classifier strategies

```text
## Task
P3.2 — Two strategies behind PageClassifier: (a) zero-shot VLM using descriptions, (b) small trained image+text classifier. Config selects; both may vote.

## Skills
S4, SETUP.md for models

## In-scope
src/torus/classify/

## Stop and ask
New model dependency / extras before adding.
```

#### P3.3 — Pack splitting

```text
## Task
P3.3 — Group consecutive pages into logical documents.

## Skills
S4

## Depends on
P3.1, P3.2
```

#### P3.4 — Multi-person detection

```text
## Task
P3.4 — Assign pages to borrower / co-applicant / guarantor when multiple people appear.

## Skills
—

## Depends on
P3.3

## Hard rule
No customer names in code; use structural/role cues + config.
```

#### P3.5 — Classification eval

```text
## Task
P3.5 — Per-type precision/recall + confusion matrix in eval report.

## Skills
S4, S6

## Depends on
P1.x eval harness, P3.2
```

---

### Phase 4 — Layout and field location

```text
## Phase ownership (all P4.*)
In-scope: src/torus/layout/, config/families/ (generated hints only)
Skills: S8
HARD RULE: no fixed page coordinates in code. Prove with random-layout synth seeds.
Acceptance: ≥97% expected fields located on dev set.
```

#### P4.1 — Layout detection

```text
## Task
P4.1 — Detect text lines, blocks, tables, figures, stamps, signatures, photos → Region objects. Layout model + classic line detector.

## Skills
S8

## Depends on
P0.3 Region, P2 pages

## Stop and ask
New layout model dependency.
```

#### P4.2 — Label anchoring

```text
## Task
P4.2 — Read printed labels; match field specs via synonyms + embeddings; search value relative to label (right/below/same row). Multi-language labels OK via config synonyms.

## Skills
S1 (field spec synonyms), S8

## Depends on
P4.1, P0.10 field specs, a printed-text reader or mock
```

#### P4.3 — Comb boxes

```text
## Task
P4.3 — Detect comb/character boxes; inpaint lines without cutting strokes; split per-character cells; handle handwriting crossing lines.

## Skills
S8

## Depends on
P4.1, P0.7 comb generators
```

#### P4.4 — Checkboxes

```text
## Task
P4.4 — Detect checkboxes/radios; fill ratio (ticked/empty/crossed); map to adjacent option labels.

## Skills
S8

## Depends on
P4.1, P0.7
```

#### P4.5 — Tables

```text
## Task
P4.5 — Table cells/headers/rows; tables split across pages.

## Skills
S8

## Depends on
P4.1
```

#### P4.6 — Stamps and signatures

```text
## Task
P4.6 — Detect/mask stamps & signatures before reading underneath; record regions (e.g. signed yes/no columns).

## Skills
S8

## Depends on
P4.1
```

#### P4.7 — Form families

```text
## Task
P4.7 — Auto-save repeated label layouts to config/families/<auto_id>.yaml (generated, human-reviewed). Hints only — never required for correctness.

## Skills
—

## Depends on
P4.2
```

#### P4.8 — Location evaluation

```text
## Task
P4.8 — Measure whether the right region was found per column before reading; wire into eval.

## Skills
S6

## Depends on
P1.x, labelled location ground truth format, P4.2+
```

---

### Phase 5 — Readers

```text
## Phase ownership (all P5.*)
In-scope: src/torus/readers/, config/readers.yaml, optional extras
Skills: S2, SETUP.md
HARD: lazy heavy imports; never invent confidence; enabled:false until bake-off; missing model = skip not crash.
```

#### P5.1 — Reader framework

```text
## Task
P5.1 — Registry, lazy load, timeouts, device from doctor, batching, cache by crop hash + reader version.

## Skills
S2

## Depends on
P0.3 Reader protocol, P0.6 doctor
```

#### P5.2 — Classic OCR

```text
## Task
P5.2 — PaddleOCR (PP-OCRv5), docTR, Tesseract. Word boxes + per-char conf where available.

## Skills
S2, SETUP.md

## Depends on
P5.1

## Stop and ask
Confirm extras/licences before pinning new packages.
```

#### P5.3 — Document VLMs

```text
## Task
P5.3 — Document-VLM family (PaddleOCR-VL, GLM-OCR, MinerU, dots.ocr, olmOCR, DeepSeek-OCR, Chandra) via local server where useful.

## Skills
S2, SETUP.md

## Depends on
P5.1

## Note
Implement as plugins; not all must ship in first PR — one solid path + registry stubs OK if brief allows incremental enablement; prefer completing smoke-tested adapters one model at a time if scope explodes (report split).
```

#### P5.4 — General VLM (Qwen3-VL)

```text
## Task
P5.4 — Qwen3-VL field-crop reading; ReadHint-constrained prompts; token log-probs as confidence.

## Skills
S2, SETUP.md

## Depends on
P5.1
```

#### P5.5 — Character-cell reader

```text
## Task
P5.5 — Small classifier per handwritten comb cell; train on public handwriting data first.

## Skills
S2, S8

## Depends on
P5.1, P4.3 cell crops
```

#### P5.6 — Handwriting line readers

```text
## Task
P5.6 — TrOCR and PARSeq as independent voters on handwritten crops.

## Skills
S2, SETUP.md

## Depends on
P5.1
```

#### P5.7 — Script / language routing

```text
## Task
P5.7 — Detect script/language per region; route to supporting readers (e.g. Tamil, Hindi, Arabic).

## Skills
S2

## Depends on
P5.1+
```

#### P5.8 — Reader bake-off

```text
## Task
P5.8 — Bake-off command: per reader × field type accuracy/speed on dev set; propose default set in readers.yaml (human approves defaults).

## Skills
S2, S6

## Depends on
P1.x, enabled readers
```

---

### Phase 6 — Consensus and confidence

```text
## Phase ownership (all P6.*)
In-scope: src/torus/consensus/, pipeline thresholds in config
Skills: S6
Acceptance: zero silent errors on dev AND hidden at confirmed; calibration plot in report.
```

#### P6.1 — Character-level voting

```text
## Task
P6.1 — ROVER-style alignment; reader weights per field type from dev set.

## Skills
—

## Depends on
P0.3 Reading/Candidate, P5 readings
```

#### P6.2 — Agreement rules

```text
## Task
P6.2 — Config rules: critical → two different reader families; non-critical → one strong reading + validator pass.

## Skills
—

## Depends on
P0.4, P0.10 critical flags, P6.1
```

#### P6.3 — Calibration

```text
## Task
P6.3 — Calibrate agreement features → P(correct) on dev set (isotonic/logistic). Re-fit when readers change. Never invent ad-hoc confidence formulas.

## Skills
S6

## Depends on
P6.1, P1.x, page quality (P2.6), validators (may stub until P7)
```

#### P6.4 — Status thresholds

```text
## Task
P6.4 — Assign confirmed|likely|needs_review|not_found; thresholds in config so confirmed has zero silent errors on dev.

## Skills
S6

## Depends on
P6.3
```

#### P6.5 — Escalation

```text
## Task
P6.5 — On disagreement, escalate to stronger/slower readers (bigger VLM, more crops, alt preprocess) before review.

## Skills
S2

## Depends on
P5.1, P6.2
```

---

### Phase 7 — Validation

```text
## Phase ownership (all P7.*)
In-scope: src/torus/validate/, resources/lookups/ (public data only)
Skills: S3
Acceptance: unit tests every validator; failing validator ⇒ never confirmed.
```

#### P7.1 — Format validators

```text
## Task
P7.1 — Indian mobile, landline+STD, email, PAN, Aadhaar (Verhoeff), voter EPIC, DL, passport, IFSC, account lengths, dates. Selected per field spec.

## Skills
S3

## Tests
Made-up valid/invalid examples only — never real IDs from customer packs.
```

#### P7.2 — Lookup validators

```text
## Task
P7.2 — PIN exists; PIN agrees with city/state; India Post master in resources/lookups/; closed state list.

## Skills
S3

## Depends on
Public lookup files only in git.
```

#### P7.3 — Cross-page agreement

```text
## Task
P7.3 — Same fact across pages must agree; agreement raises confidence; disagreement → review with both values.

## Skills
S3

## Depends on
P6 status model, multi-page candidates
```

#### P7.4 — Plausibility

```text
## Task
P7.4 — Age range, date order, phone ≠ account number, rules from field specs.

## Skills
S3
```

#### P7.5 — Normalisation

```text
## Task
P7.5 — Per-column normalisation (dates, name casing, address lines) from template.

## Skills
S1, S3

## Depends on
P0.10 / template types
```

---

### Phase 8 — Mapping and Excel

```text
## Phase ownership (all P8.*)
In-scope: src/torus/mapping/, src/torus/output/
Skills: S1
Acceptance: hidden set zero silent errors; every filled cell has evidence; synth packs → exact Excel.
Read: schemas/deb/excel_columns.json until client Excel arrives.
```

#### P8.1 — Template import

```text
## Task
P8.1 — Read client Excel headers only → config/templates/<id>.yaml. Mark non-engine columns passthrough. Devs enrich descriptions/types/validators (S1).

## Skills
S1

## Note
Until client file shared, keep/refresh deb_v1 from schemas.
```

#### P8.2 — Rows and people

```text
## Task
P8.2 — Cardinality: rows per person/address/pack; assign values to correct person/row.

## Skills
S1

## Depends on
P3.4, template cardinality
```

#### P8.3 — LLM field mapper

```text
## Task
P8.3 — Local LLM (Qwen3 family) maps specs + validated candidates → JSON. Must cite region ids; uncited rejected; not_found allowed; temperature 0. No cloud APIs.

## Skills
S1, SETUP.md

## Depends on
P0.3 FieldMapper, P6–P7 candidates
```

#### P8.4 — Deterministic mapping first

```text
## Task
P8.4 — Directly anchored+validated columns fill without LLM. LLM only for judgement cases.

## Skills
S1

## Depends on
P4 anchors, P8.3
```

#### P8.5 — Excel writer

```text
## Task
P8.5 — Write copy of client template; second sheet: status, confidence, evidence (page+box); colour by status. Fill empties policy as configured — never invent overwrites of protected cells unless config says so.

## Skills
—

## Depends on
P8.2–P8.4
```

#### P8.6 — CLI torus run

```text
## Task
P8.6 — torus run <file|folder> --template <id> --out <xlsx>

## Skills
—

## Depends on
P8.5 end-to-end wiring
```

---

### Phase 9 — Review

```text
## Phase ownership (all P9.*)
In-scope: src/torus/review/ (+ optional review extra)
Skills: —
Depends on: P8 FieldValue statuses + evidence
Corrections ONLY under TORUS_DATA_DIR.
```

#### P9.1 — Review UI

```text
## Task
P9.1 — Local web app queue for needs_review/likely: crop, page context, all readings, validators, suggestion. Keyboard-first.

## Skills
—

## Extra
review optional install if needed
```

#### P9.2 — Correction store

```text
## Task
P9.2 — Persist decisions (crop ref, value, reviewer, time) under TORUS_DATA_DIR.

## Skills
—
```

#### P9.3 — Re-export

```text
## Task
P9.3 — Regenerate Excel with corrected values marked reviewed.

## Skills
—

## Depends on
P8.5, P9.2
```

#### P9.4 — Promote to test set

```text
## Task
P9.4 — One-click promote reviewed packs into test set format (P1.1).

## Skills
S6

## Depends on
P1.1, P9.2
```

#### P9.5 — Review metrics

```text
## Task
P9.5 — Fields/pack needing review, time/field, most-corrected columns in eval report.

## Skills
S6
```

---

### Phase 10 — Learning loop

```text
## Phase ownership (all P10.*)
In-scope: src/torus/learn/, config/train/, model registry
Skills: S7
Acceptance: one full cycle with measured automatic-rate increase; promote only if hidden improves and silent errors stay 0.
```

#### P10.1 — Dataset builder

```text
## Task
P10.1 — Corrections → training sets per reader type; split by document. Output under TORUS_DATA_DIR/training/ — never git.

## Skills
S7
```

#### P10.2 — Fine-tune recipes

```text
## Task
P10.2 — config/train/ recipes: char-cell, TrOCR/PARSeq, LoRA for doc VLM and Qwen3-VL.

## Skills
S7, SETUP.md
```

#### P10.3 — Model registry

```text
## Task
P10.3 — Versioned weights: data hash, base model, metrics, date. readers.yaml points at registry versions.

## Skills
S7
```

#### P10.4 — Promotion rule

```text
## Task
P10.4 — Promote only if hidden-set improves and silent errors stay zero; re-run P6.3 calibration after promotion.

## Skills
S6, S7
```

#### P10.5 — Active learning

```text
## Task
P10.5 — Prioritise review where model is least sure.

## Skills
S7

## Depends on
P9 queue, P6 confidence
```

---

### Phase 11 — Service / production

```text
## Phase ownership (all P11.*)
In-scope: src/torus/service/, packaging, ops docs
HARD: metadata-only audit; no outbound network from workers; auth on API.
```

#### P11.1 — HTTP API

```text
## Task
P11.1 — FastAPI: submit file/folder → job id → Excel/status. Batch folder watcher.

## Skills
—
```

#### P11.2 — Queue and workers

```text
## Task
P11.2 — Job queue/workers; model servers as separate processes; retries/timeouts per stage.

## Skills
—
```

#### P11.3 — Audit log

```text
## Task
P11.3 — Per job: models, config versions, timings, statuses, reviewer actions. Metadata only.

## Skills
—
```

#### P11.4 — Monitoring

```text
## Task
P11.4 — Throughput, queue length, review rate, silent-error spot checks, model health.

## Skills
—
```

#### P11.5 — Security

```text
## Task
P11.5 — API auth; encryption at rest for TORUS_DATA_DIR; retention/deletion; no outbound network from workers.

## Skills
—
```

#### P11.6 — Packaging

```text
## Task
P11.6 — Docker CPU/GPU images; config bundle; upgrade/rollback procedure.

## Skills
—
```

#### P11.7 — Load test

```text
## Task
P11.7 — Load test realistic packs; document throughput per hardware in docs/SETUP.md.

## Skills
SETUP.md
```

---

### Phase 12 — Scale-out

```text
## Phase ownership (all P12.*)
Prefer config-only changes. Code only for genuinely new field types/validators (ask first).
Skills: S1, S4, S6, S8
Acceptance: one new template onboarded with no pipeline code changes.
```

#### P12.1 — New template

```text
## Task
P12.1 — Import Excel, field specs, label 20 packs, evaluate. Code only if new type/validator required (ask first).

## Skills
S1, S6
```

#### P12.2 — New language/script

```text
## Task
P12.2 — Enable script readers (P5.7); synth fixtures in that script; evaluate.

## Skills
S2, S8, S6
```

#### P12.3 — New document family

```text
## Task
P12.3 — New family (e.g. invoices, KYC-only): page types (S4), field specs, test set.

## Skills
S4, S1, S6
```

#### P12.4 — Onboarding metric

```text
## Task
P12.4 — Track time-to-onboard-new-template; goal = days not weeks.

## Skills
S6
```

---

## Quick copy: minimal one-task message

When you are in a hurry, send this (still paste Shared preamble above it, or `@AGENTS.md`):

```text
@AGENTS.md @docs/IMPLEMENTATION_PLAN.md @SKILLS.md

Implement ONLY <TASK_ID> from docs/IMPLEMENTATION_PLAN.md.
Follow docs/AGENT_PROMPTS.md brief for <TASK_ID>.
Base branch: <BASE_BRANCH>.
First reply: plan (files + tests), then implement.
Stop and ask for new deps, protocol changes, or anything marked HARD STOP / owner approval.
Do not commit unless I ask.
```

---

## Checklist for the human dispatcher

- [ ] Task ID still open / not already on another branch
- [ ] Upstream tasks merged or interfaces available
- [ ] Shared preamble + task brief pasted
- [ ] `@` files attached
- [ ] Human gates called out (P0.8 history, labelling, bake-off defaults, client Excel, licences)
- [ ] Agent told: no commit / no push to main unless asked
