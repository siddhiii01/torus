# Torus: phase-by-phase implementation plan

Version 1, 18 November 2026. Branch `dev-sid`.

## What we are building, in one paragraph

A local engine that takes **any PDF or image** (scanned, photographed, handwritten, any layout, any language) and **fills the columns of the client's Excel template**. Nothing more. Each column is described once in configuration, and the engine finds, reads, checks and fills it. Every value carries its source (page and box) and a status. A value is `confirmed` only when several independent signals agree. Everything else goes to a short human review queue. Corrections feed back into training, so the share that needs review keeps falling.

**Quality target:** zero silent errors (a `confirmed` value that is wrong). The automatic rate, meaning the share of fields that need no human, starts low and rises with every cycle.

**Who does what:** AI coding agents write the code and tests, run them, and report evidence (`AGENTS.md`, `SKILLS.md`). Developers check the results, give feedback, and approve decisions.

Hardware is not part of this plan. Each developer's machine runs whatever models it can. Model requirements are in `docs/SETUP.md` and are checked automatically by `torus doctor`.

---

## Where we start

The repo already has useful pieces and some that must be replaced.

**Keep and port into the new structure**

- `ocr/engine.py`: file → pages → words with boxes, confidence, status. This becomes `ingest/` plus the first readers.
- The review HTML idea, which becomes the base of `review/`.
- `schemas/deb/excel_columns.json`, `field_groups.json`, `row_key.md`: the column list, the scored fields and row matching. These become the first template config and the eval rules.
- `scripts/deb_score.py`, which becomes `eval/`.
- PIN master lookup and the "fill empties only, never overwrite" policy.
- The idea in `ocr/digit_crop.py` (find the label, crop the comb, read digits only), made general.

**Remove (hard-coded or unsafe)**

- Name and street fixes tied to one customer in `scripts/deb_extract.py`.
- Fixed page coordinates and `page == 0` logic throughout the DEB scripts.
- The invented confidence formula in `ocr/crop_vlm.py`.
- Real customer PDFs and answer keys committed under `data/deb/cases/`. They move out of git (this needs owner approval, Phase 0).

Old scripts move to `legacy/` and are deleted as each part is replaced.

---

## Phase overview

| Phase | Name | Result |
|---|---|---|
| 0 | Foundation | Clean repo, package structure, tooling, data models, config, CI, synthetic fixtures |
| 1 | Evaluation first | Scoring harness and test-set format, so every later phase is measured |
| 2 | Ingest and page clean-up | Any PDF or image becomes upright, clean, classified-ready pages |
| 3 | Page classification and pack splitting | Every page has a type. Packs split into documents |
| 4 | Layout and field location | Label anchors, comb boxes, checkboxes, tables, stamps, without fixed positions |
| 5 | Reader framework and readers | Plug-in readers (classic OCR, document VLMs, general VLMs) behind one interface |
| 6 | Consensus and confidence | Character-level voting and calibrated confidence |
| 7 | Validation and cross-checks | Format, checksum, lookup and cross-page agreement for every value |
| 8 | Field mapping and Excel output | LLM fills template columns with evidence. Excel written with status |
| 9 | Review and correction | Review screen, correction store, feedback into eval |
| 10 | Learning loop | Fine-tuning readers on corrections, model registry |
| 11 | Service and production hardening | API, batch runs, audit log, monitoring, packaging |
| 12 | Scale-out | New templates, new languages, new document families, by config only |

Phases 0–1 come first, in order. After that, phases 2–7 can be worked on in parallel by different developers, because they meet only through the `torus.core` interfaces. Phase 8 needs 5–7. Phases 9–10 need 8.

Each phase lists **tasks** (agent-sized, one PR each), **acceptance criteria** (what tests and evaluation must show) and a **human check** (what a developer looks at).

---

## Phase 0: Foundation

**Goal:** a codebase where agents can work safely and every change is tested.

| ID | Task |
|---|---|
| P0.1 | Create the `src/torus/` package and the folder layout from `AGENTS.md` §3.1. Add `pyproject.toml` (Python 3.12, base deps pinned, optional extras `paddle`, `doctr`, `vlm`, `train`, `review`). |
| P0.2 | Tooling: ruff, mypy, pytest with markers (`model`, `slow`), and pre-commit. Add a GitHub Actions workflow that runs lint, types and the non-model tests on every push and PR. |
| P0.3 | `torus.core`: pydantic data models (`Document`, `Page`, `Region`, `Reading`, `Candidate`, `FieldValue`, `ExtractionResult`), protocols (`Reader`, `PageClassifier`, `Validator`, `FieldMapper`) and typed errors. Unit tests. |
| P0.4 | `torus.config`: typed loader for `config/*.yaml` with schema validation and a config hash recorded in every result. Starter files: `pipeline.yaml`, `readers.yaml`, `page_types.yaml`, `templates/`. |
| P0.5 | `torus.logging`: metadata-only logger with redaction. A test proves that document text cannot reach log output. |
| P0.6 | `torus doctor`: detects OS, CPU, RAM, GPU and memory, installed extras and model weights, then prints which readers can run on this machine. |
| P0.7 | Synthetic fixture framework (`tests/fixtures/synth/`): seeded generators for printed forms, comb rows, checkboxes, handwriting-like text from public handwriting fonts, rotation, fax-like degradation and stamps. |
| P0.8 | Data safety: move `data/deb/cases/*` out of the repo to `TORUS_DATA_DIR`, tighten `.gitignore`, and add a pre-commit check that blocks PDFs, images, xlsx and json under data paths. **Removing the files from git history needs owner approval before it is done.** |
| P0.9 | Move the existing `ocr/` and `scripts/` into `legacy/`, marked read-only. Note which parts each later phase will port. |
| P0.10 | Convert `schemas/deb/excel_columns.json` + `field_groups.json` into `config/templates/deb_v1.yaml` (field specs with types, validators, `look_in`, cardinality, critical flag). It is replaced by the client's Excel once it is shared (P8.1). |

**Acceptance:** `pytest -m "not model"` passes in CI. `torus doctor` runs on Windows and Linux. The config loads and validates. The logger redaction test passes. No customer data is tracked in the working tree.

**Human check:** the repo layout matches `AGENTS.md`, CI is green, and the data-move plan is approved.

---

## Phase 1: Evaluation first

**Goal:** before improving anything, be able to measure it.

| ID | Task |
|---|---|
| P1.1 | Test-set format: each case is a folder in `TORUS_DATA_DIR/testsets/<set>/<case>/` with `source.pdf`, `expected.json` (rows of column → value) and `meta.json` (split `dev` or `hidden`, tags such as year, scan type, handwriting level). |
| P1.2 | `torus.eval`: port `deb_score.py`. Add row alignment (from `row_key.md`, made configurable per template), per-column match rules (exact, normalised, fuzzy) from the template, and metrics: accuracy, **silent errors**, status rates, automatic rate. |
| P1.3 | Report writer: Markdown and HTML summary per run, with per-column and per-page-type tables, timing, and model and config versions. No document text. |
| P1.4 | `torus.eval diff`: compare two runs. Fail when silent errors rise or a column drops beyond the noise margin. |
| P1.5 | Labelling tool (local, simple): open a PDF, fill the template columns by hand, save `expected.json`. This is used to grow the test set without sending documents anywhere. |
| P1.6 | Baseline: run the legacy pipeline through the new harness on the existing cases and record the starting numbers. |

**Acceptance:** the harness scores a synthetic test set end-to-end in CI. A baseline report exists for the real dev set (stored locally, summary in the PR).

**Human check:** the metrics make sense, and developers start labelling: **at least 20 packs, 5 of them marked `hidden`**, covering different years, forms, scan quality and handwriting.

---

## Phase 2: Ingest and page clean-up

**Goal:** any input file becomes clean, upright page images.

| ID | Task |
|---|---|
| P2.1 | `ingest/`: PDF (PyMuPDF), images (jpg, png, tiff, multi-page tiff), and phone photos (EXIF rotation). Render at the dpi in config. Use the embedded text layer only when a quality check says it is usable (in reading order, not garbled). |
| P2.2 | Orientation detection (0/90/180/270), including handwriting-only pages: a classic detector plus a small classifier, and the two must agree, otherwise both orientations are kept for reading. |
| P2.3 | Deskew, perspective correction for photos, and border/shadow cropping. |
| P2.4 | Quality repair for fax-like and faded scans: stroke thickening, denoising, and contrast normalisation for grey scans. Each step is switchable in config and measured. |
| P2.5 | Blank and near-blank page detection (ink density plus OCR check). Blank pages are recorded, not silently dropped. |
| P2.6 | Page quality score (blur, contrast, resolution), stored on the `Page` for later confidence and routing. |

**Acceptance:** component tests on synthetic rotated, skewed, degraded and blank pages pass. On the dev set, a human-checked sample of 100 pages is 100% upright, and blank pages are flagged correctly.

**Human check:** look at a contact sheet of before/after pages from the dev set.

---

## Phase 3: Page classification and pack splitting

**Goal:** know what each page is, so each column is searched only where it can be.

| ID | Task |
|---|---|
| P3.1 | Page-type list in `config/page_types.yaml` with descriptions. Start from what the packs contain: application form, form continuation, verification report, ID card copies (voter ID, Aadhaar, PAN, driving licence, ration card, passport), bank statement, approval memo, checklist, brochure/terms, photo/signature page, blank, other. |
| P3.2 | Classifier, two strategies behind one interface: (a) a zero-shot VLM prompt using the descriptions, and (b) a small trained image+text classifier once labels exist. Config chooses which; both can vote. |
| P3.3 | Pack splitting: group consecutive pages into logical documents (one application form spanning 2 pages, one ID copy, and so on). |
| P3.4 | Multi-person detection: identify which pages belong to borrower, co-applicant or guarantor where the pack contains several people. |
| P3.5 | Classification evaluation: per-type precision/recall and a confusion matrix, added to the eval report. |

**Acceptance:** at least 98% page-type accuracy on the dev set. Every misclassification is listed by type in the report.

**Human check:** the confusion matrix, plus a spot check of the "other" bucket.

---

## Phase 4: Layout and field location

**Goal:** find every region that can hold a template column, with no fixed coordinates.

| ID | Task |
|---|---|
| P4.1 | Layout detection: text lines, blocks, tables, figures, stamps, signatures and photos, using a layout model plus a classic line detector. Output: `Region` objects. |
| P4.2 | Label anchoring: printed labels are read with a printed-text reader, then matched to field specs by meaning, using synonyms in the field spec plus embeddings. That covers "Mobile", "Mob. No.", "Cell" and other-language labels. The value region is searched relative to the label (right, below, in the same table row). |
| P4.3 | Comb and character-box handling: detect box grids, remove the lines without cutting strokes (inpainting), and split into per-character cells. Handles boxes with handwriting crossing the lines. |
| P4.4 | Checkboxes and radio buttons: detect them and measure the fill ratio (ticked, empty, crossed out). Map them to the option labels next to them. |
| P4.5 | Tables: detect cells and headers and map rows. Handles tables split across pages. |
| P4.6 | Stamps and signatures: detect and mask them before reading text underneath, and record them as regions (some columns need "signed: yes/no"). |
| P4.7 | Form families: when a printed form is seen repeatedly, save its label layout as `config/families/<auto_id>.yaml` (generated, then human-reviewed). Used as a *hint* that speeds up and stabilises location. It is never required. |
| P4.8 | Location evaluation: for labelled cases, measure whether the right region was found for each column (before reading). |

**Acceptance:** on the dev set, at least 97% of expected fields have a located region. All tests run on synthetic forms with random layouts generated per seed, which proves there are no fixed positions.

**Human check:** the debug overlay (boxes on pages) for a sample of unseen packs.

---

## Phase 5: Reader framework and readers

**Goal:** many independent readers behind one interface, any of which can be swapped by config.

| ID | Task |
|---|---|
| P5.1 | `readers/` framework: registry, lazy loading, timeouts, a GPU/CPU device choice from `torus doctor`, batching, and caching of readings by crop hash plus reader version. |
| P5.2 | Classic OCR family: PaddleOCR (PP-OCRv5 detection + recognition), docTR, Tesseract. Word boxes and per-character confidences where available. |
| P5.3 | Document-VLM family: PaddleOCR-VL, GLM-OCR, MinerU, dots.ocr, olmOCR, DeepSeek-OCR, Chandra. Region- and page-level reading with layout output. Served through a local model server where useful (vLLM or Ollama). |
| P5.4 | General-VLM family: Qwen3-VL (sizes chosen by config). Field-crop reading with constrained prompts built from the `ReadHint` (charset, length, script), returning token log-probabilities as confidence. |
| P5.5 | Character-cell reader: a small classifier for one handwritten character per comb cell (digits and letters), trained first on public handwriting datasets and later on our corrections. |
| P5.6 | Handwriting line readers: TrOCR and PARSeq, as additional independent voters on handwritten crops. |
| P5.7 | Script and language detection per region, which routes to readers that support that script (for example Tamil, Hindi, Arabic). |
| P5.8 | Reader bake-off command: per reader, per field type, accuracy and speed on the dev set. The results decide the default reader set in `config/readers.yaml`. |

**Acceptance:** every reader passes its smoke test where installed. The pipeline runs with any subset enabled. The bake-off report exists and the default set is chosen from it.

**Human check:** the bake-off table, and agreement on the default reader set.

---

## Phase 6: Consensus and confidence

**Goal:** turn several readings into one value with an honest confidence.

| ID | Task |
|---|---|
| P6.1 | Character-level alignment of readings (ROVER-style), with reader weights learned per field type from the dev set. |
| P6.2 | Agreement rules from config: critical fields need agreement from **two different reader families**. Non-critical fields need one strong reading plus a validator pass. |
| P6.3 | Confidence calibration: map raw agreement features (reader scores, number of agreeing families, validator results, page quality) to a probability of being right, fitted on the dev set (isotonic or logistic). Re-fit whenever readers change. |
| P6.4 | Status assignment: `confirmed` / `likely` / `needs_review` / `not_found`, with thresholds chosen so that the dev set has **zero silent errors** at `confirmed`. The thresholds live in config. |
| P6.5 | Escalation: when readers disagree, automatically ask stronger or slower readers (bigger VLM, more crops, different preprocessing) before giving up to review. |

**Acceptance:** zero silent errors on the dev set **and** on the hidden set. A calibration plot is in the report.

**Human check:** a sample of `confirmed` values from unseen packs, checked by eye.

---

## Phase 7: Validation and cross-checks

**Goal:** catch wrong readings with rules a reader cannot fool.

| ID | Task |
|---|---|
| P7.1 | Format validators: Indian mobile, landline with STD code, email, PAN, Aadhaar (Verhoeff checksum), voter ID (EPIC), driving licence, passport, IFSC, account number lengths, dates. Each is selected per column in the field spec. |
| P7.2 | Lookup validators: PIN code exists, and PIN agrees with the city and state (India Post master in `resources/lookups/`). The state comes from a closed list. |
| P7.3 | Cross-page agreement: the same fact (name, DOB, address, phone, ID number) found on several pages must agree. Agreement raises confidence. Disagreement sends the field to review and shows both values. |
| P7.4 | Plausibility: age range, date order (application after DOB), phone not equal to account number, and similar rules from the field specs. |
| P7.5 | Normalisation per column type (date format, name casing, address line splitting), driven by the template. |

**Acceptance:** unit tests for every validator. On the dev set, every value has validator results recorded, and no value that fails a validator is ever `confirmed`.

**Human check:** the list of validator failures on the dev set. Are they real misreads or validator bugs?

---

## Phase 8: Field mapping and Excel output

**Goal:** fill exactly the client's Excel columns, with evidence.

| ID | Task |
|---|---|
| P8.1 | Template import: read the client's Excel (headers only) and generate or update `config/templates/<id>.yaml`. Developers then add descriptions, types and validators per column (`SKILLS.md` S1). Columns the engine should not fill (internal checker columns, for example) are marked `passthrough`. |
| P8.2 | Rows and people: decide how many rows a pack produces (per person, per address, per pack) from the template's cardinality rules. Assign each value to the right person and row. |
| P8.3 | LLM field mapper: a local LLM (Qwen3 family, size from config) gets the field specs plus the validated candidates with their region ids, and returns JSON constrained to the template schema. Every value **must cite region ids**. Uncited values are rejected. "Not found" is allowed. Temperature 0. |
| P8.4 | Deterministic mapping first: columns that are directly anchored (label found, value read, validated) are filled without the LLM. The LLM handles only what needs judgement: which address is which type, free-text fields, multi-person packs. |
| P8.5 | Excel writer: write into a copy of the client template (formatting kept). Add a second sheet with status, confidence and evidence (page and box) per cell. Colour cells by status. |
| P8.6 | End-to-end CLI: `torus run <file or folder> --template <id> --out <xlsx>`. |

**Acceptance:** on the hidden set, zero silent errors, and every filled cell has evidence. The integration test on synthetic packs produces the expected Excel exactly.

**Human check:** open the produced Excel for unseen packs and compare it against the PDFs.

---

## Phase 9: Review and correction

**Goal:** a person resolves uncertain fields quickly, and every correction is captured.

| ID | Task |
|---|---|
| P9.1 | Review UI (local web app): a queue of `needs_review` and `likely` fields. Each shows the crop, the page in context, every reader's reading, the validator messages and the suggested value. Keyboard-first, so confirming or correcting takes seconds. |
| P9.2 | Correction store: each decision is saved with the crop reference, the final value, the reviewer and the time, under `TORUS_DATA_DIR`. |
| P9.3 | Re-export: after review, the Excel is regenerated with the corrected values marked `reviewed`. |
| P9.4 | Feedback into eval: reviewed packs can be promoted into the test set with one click. |
| P9.5 | Review metrics: fields per pack needing review, time per field, and the most-corrected columns, all shown in the eval report. |

**Acceptance:** a reviewer can finish a typical pack's queue without opening the PDF separately. Corrections round-trip into the Excel and the store.

**Human check:** reviewers try it on real packs and report friction.

---

## Phase 10: Learning loop

**Goal:** get better on *our* documents with every batch of corrections.

| ID | Task |
|---|---|
| P10.1 | Dataset builder: turn corrections into training sets per reader type (character cells, handwritten line crops, field crops for VLM fine-tuning), split by document. |
| P10.2 | Fine-tune recipes in `config/train/`: character-cell model, TrOCR/PARSeq, and LoRA fine-tuning for the document VLM and Qwen3-VL. |
| P10.3 | Model registry: versioned weights with data hash, base model, metrics and date. `config/readers.yaml` points to registry versions. |
| P10.4 | Promotion rule: new weights replace old ones only when the hidden-set evaluation improves and silent errors stay at zero. Re-run calibration (P6.3) after every promotion. |
| P10.5 | Active learning: prioritise review of the field types and crops where the model is least sure, so each correction teaches the most. |

**Acceptance:** one full cycle (corrections → fine-tune → evaluation → promotion) is done, with a measured increase in the automatic rate.

**Human check:** before/after table per column.

---

## Phase 11: Service and production hardening

**Goal:** run it reliably for daily volumes.

| ID | Task |
|---|---|
| P11.1 | HTTP API (FastAPI): submit a file or folder, get a job id, fetch the Excel and status. Batch folder watcher for bulk runs. |
| P11.2 | Job queue and workers. Model servers run as separate processes so they can scale per GPU. Retries and timeouts per stage. |
| P11.3 | Audit log: per job, which models and config versions ran, timings, statuses, and who reviewed what. Metadata only. |
| P11.4 | Monitoring: throughput, queue length, review rate, silent-error spot checks, model health. |
| P11.5 | Security: authentication on the API, encryption at rest for `TORUS_DATA_DIR`, retention and deletion policy, and no outbound network from workers. |
| P11.6 | Packaging: Docker images (CPU and GPU variants), a single config bundle per deployment, and an upgrade/rollback procedure. |
| P11.7 | Load test with realistic pack sizes, and document the throughput per hardware profile in `docs/SETUP.md`. |

**Acceptance:** a batch of unseen packs runs end-to-end through the service with the same results as the CLI, and the audit log is complete.

**Human check:** an operations walkthrough of deploy, run, review, roll back.

---

## Phase 12: Scale-out

**Goal:** new clients, templates, languages and document families **without new code**.

| ID | Task |
|---|---|
| P12.1 | New template: import the Excel, write the field specs, label 20 packs, evaluate. Code changes only if a new field type or validator is genuinely needed. |
| P12.2 | New language or script: enable readers that support it (P5.7), add synthetic fixtures in that script, and evaluate. |
| P12.3 | New document family (for example invoices or KYC-only packs): add page types (S4), field specs, a test set. |
| P12.4 | Track a "time to onboard a new template" metric. The goal is days, not weeks. |

**Acceptance:** one new template onboarded with no pipeline code changes.

---

## Definition of done (every task)

1. Tests written and passing (unit, component, integration as relevant). Model tests pass where the model is installed.
2. Lint, format and type checks are clean.
3. Evaluation run if output can change. The report table is in the PR. No silent-error increase.
4. No hard-coded document knowledge, no real data, and no document text anywhere in the diff.
5. Docs updated if behaviour or config changed.

## Open items

- **Client Excel template:** to be shared. Until then, `config/templates/deb_v1.yaml` is generated from the existing `schemas/deb/excel_columns.json` (73 columns, sheet "56 accounts").
- **Removing customer data from git history** (Phase 0, P0.8): needs owner approval.
- **Test set:** developers to label at least 20 packs (5 hidden) during Phase 1.
- **Reviewers:** who reviews, and the expected daily volume (needed by Phase 9).
- **Licences:** checked per model before production use (recorded in `docs/SETUP.md`).
