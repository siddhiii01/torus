# OCR Engine: Plan & Progress Tracker

Owner: Celeris Ventures (AI team) · Started: 2026-10-13
This is the living document for the project. Update it at the end of every working session.

---

## 1. Why we are building this

We use Google Document AI for OCR. Two problems:

1. **Cost.** Per-page pricing hurts at scale.
2. **Testing.** We cannot test freely because every run costs money.

Goal: an OCR engine that runs **locally on our own servers**, so cost is fixed (hardware) and testing is unlimited.

Customer documents contain PII (Aadhaar, PAN, passports, bank documents). Running locally also means **no document ever leaves our infrastructure**.

---

## 2. Expected end result

A self-hosted service that takes **any document** (PDF or image) and returns **structured, reliable text with positions and confidence**, ready for the downstream LLM extraction step.

### 2.1 What it must do
- Accept: PDF (digital and scanned), JPG/PNG/TIFF, multi-page files.
- Return, per page: full text, lines and words with **bounding boxes**, **confidence scores**, reading order, and detected tables.
- Handle **documents we have never seen** without a human looking at them (see 2.3).
- Be a drop-in replacement for Document AI at the call site: same kind of request and a compatible response shape, so the main project can switch with a config change.
- Support new languages and new document domains by **adding a plugin or model**, not rewriting the system.

### 2.2 What it must not do
- Send any document, text or image to an external service.
- Use Chinese-origin models (company constraint). This includes PaddleOCR, Qwen-family models, MinerU, DeepSeek-OCR, and fine-tunes built on them. Check the base model of anything fine-tuned.
- Log document content. Logs hold metadata only (page count, timings, engine used, confidence stats).

### 2.3 Unknown documents: the central design problem
We will process documents nobody on the team sees. The engine therefore cannot rely on per-document tuning. It must:

1. **Route automatically.** Detect digital vs scanned, orientation, language and quality per page, and pick the engine.
2. **Measure its own confidence.** Every page gets a quality score (OCR confidence, text sanity checks, language-model plausibility).
3. **Escalate when unsure.** Low-confidence pages go to a stronger fallback (vision-language model). If still unsure, the result is flagged `needs_review` instead of silently returning garbage.
4. **Stay honest.** The response always says which engine ran and how sure it is. Downstream code can decide what to do with low-confidence output.
5. **Improve from feedback.** Collect flagged pages (with consent and under the PII policy) to grow the test set.

### 2.4 Success criteria (to be given numbers after the benchmark)
| Metric | Target |
|---|---|
| Character accuracy vs reference on printed docs | TBD (benchmark vs Document AI) |
| Field-level accuracy after LLM step (vendor, amounts, dates, IDs) | TBD |
| Silent failures (bad output returned with high confidence) | as close to zero as possible |
| Throughput | TBD per GPU |
| Cost per page | far below Document AI |

---

## 3. Document domains

| Priority | Domain | Notes |
|---|---|---|
| 1 | Invoices, receipts, bills | Vendor, amounts, dates. Mostly digital PDFs, some scans and photos. Sample set exists. |
| 2 | KYC / identity (Aadhaar, PAN, passports) | Photos and scans. Needs rotation handling, passport MRZ. High PII sensitivity. |
| 3 | Banking and insurance (cheques, forms, claims) | Handwriting, stamps, checkboxes, tables. |
| Later | More domains and languages | Engine must stay extensible. |

---

## 4. Planned architecture

```
Input file
  -> Intake: type sniffing, page split, size limits
  -> Per-page analysis: digital text layer? scanned? orientation, language, quality
  -> Router
       |- digital page    -> direct text-layer extraction (PyMuPDF), exact and ~free
       |- scanned page    -> preprocess (deskew, denoise) -> detect text -> recognize text
       |- hard / low conf -> vision-language model fallback (GPU)
  -> Layout and tables (reading order, table structure)
  -> Quality gate (confidence, sanity checks) -> ok | escalate | needs_review
  -> Normalized output (JSON): text, words, boxes, confidence, tables, provenance
  -> Downstream LLM extraction (separate step, not part of this engine)
```

Design principles:
- **Pluggable engines** behind one interface. Swapping a model must not touch the rest.
- **Stateless service** (FastAPI), so it scales horizontally when volume grows.
- **Evidence over opinion.** Every engine choice is backed by a benchmark on our data.
- **Privacy by default.** Local only, no content in logs, test data stays on our machines.

---

## 5. Candidate components (allowed under our constraints)

| Role | Candidates | Origin | Status |
|---|---|---|---|
| Digital PDF text | PyMuPDF | Artifex | planned |
| Classical OCR baseline | Tesseract | Google | to benchmark |
| Detect + recognize | docTR | Mindee (France) | to benchmark, likely main |
| Layout / tables | Docling, Granite-Docling | IBM | to benchmark |
| Detect + recognize + layout | Surya | Datalab (US) | to benchmark, **license check needed** |
| Handwriting | TrOCR, Florence-2 | Microsoft | later |
| Fallback VLM | Gemma 3, Mistral Small vision, Llama 3.2 Vision | Google / Mistral / Meta | later, **license check needed** |

Rejected: PaddleOCR, Qwen-VL, MinerU, DeepSeek-OCR, olmOCR and Nanonets-OCR (Qwen-based). Reason: no Chinese-origin models.

---

## 6. Hardware

| Machine | GPU | Use |
|---|---|---|
| Dev laptop | RTX 2050, 4 GB VRAM | Build and test the classical pipeline (small models or CPU) |
| PC | 16 GB VRAM | Vision-language model experiments (quantized 7-12B) and heavier benchmarks |
| Production server | GPU available (spec TBD) | Size after benchmarks give throughput numbers |

Rule of thumb: text-layer extraction needs CPU only; detect and recognize models are small; vision-language models are the real GPU consumers.

---

## 7. Data

- **Sample set:** `E:\final-recon-sets` (about 200 files: vendor invoices, GRVs, cargo documents, emails). Procurement reconciliation scenarios.
- **Ground truth:** none yet. Document AI outputs are not saved. We must create reference values ourselves (see Phase 1).
- **PII policy:** real customer data stays on company machines. It is never uploaded to external services, never committed to a repository and never pasted into logs. Samples are read from `E:\` and are not copied into this project folder without a decision.

---

## 8. Roadmap (step by step)

### Phase 0: Foundation  (done)
- [x] Define goal, constraints and end result (this document)
- [x] Set up the project: Python 3.12 venv, git (`main`), `.gitignore` that blocks documents and reports
- [ ] Decide the dependency and license review process

### Phase 1: Understand our data  `<- we are here`
- [x] Inventory script (`scripts/inventory.py`): type, size, pages, text-layer presence. Metadata only.
- [x] Triage: classify each PDF as digital, scanned or mixed. See Findings below.
- [x] Dedupe and propose gold-set candidates (`scripts/gold_candidates.py`).
- [x] Labelling scope decided: **option A**, one set of key fields per file (doc type, vendor, number, date, currency, total, notes).
- [x] Labelling tool built: `scripts/label.py` (resumable, writes git-ignored `reports/labels.csv`).
- [x] **Label the 40 gold files** (done 2026-10-19 by Claude at the owner's request, saved to git-ignored `reports/labels.csv`). A human spot-check of about 10 files is still advisable. `scripts/label.py` stays available for adding labels later.
- [ ] Collect samples for the other domains (KYC, cheques, forms, handwriting).
- [x] Record findings here (section 8a).

### Phase 1b: Engine setup and smoke test (done 2026-10-19)
- [x] Installed CPU PyTorch 2.14.1, docTR 1.1.0, pytesseract 0.3.13 (Tesseract 5.3.3 was already on the laptop, English only). Versions pinned in `requirements.txt`.
- [x] Smoke test (`scripts/smoke_test.py`): see Findings, Goal 4.

### Phase 2: Baseline benchmark
- [x] Benchmark harness: `scripts/run_ocr.py` (runs engines, caches output) and `scripts/score.py` (field presence scoring). Field-level only so far. Character-level accuracy is not measured yet.
- [x] First run: text layer, Tesseract, docTR on all 105 pages of the 40 gold files (results in Findings, Goal 5).
- [ ] Still to run: Docling, Surya (license check first), and a vision-language model on the 16 GB PC. Compare on accuracy, speed and VRAM.
- [ ] Optional: run Document AI once on the gold set (small cost) as the reference to beat.
- [ ] Decide the primary engine from the numbers.

### Phase 3: Core engine
- [ ] Engine interface and plugin registry
- [ ] Page analysis and router
- [ ] Preprocessing (deskew, rotation, denoise)
- [ ] Normalized output schema (Document AI-compatible adapter)
- [ ] Quality gate and confidence scoring

### Phase 4: Hard cases and robustness
- [ ] Vision-language fallback on the 16 GB PC
- [ ] Tables, handwriting, stamps, low-quality photos
- [ ] Stress test with deliberately bad inputs (blur, rotation, shadows, folded paper)
- [ ] Failure modes: corrupt PDFs, huge files, password-protected files

### Phase 5: Service
- [ ] FastAPI service, async queue for large jobs
- [ ] Docker image, GPU deployment
- [ ] Metadata-only logging and metrics
- [ ] Load test, capacity sizing

### Phase 6: New domains and languages
- [ ] KYC documents (Aadhaar, PAN, passport MRZ)
- [ ] Banking and insurance forms
- [ ] Additional languages as requirements appear

### Phase 7: Cutover
- [ ] Shadow mode: run next to Document AI in the main project, compare results
- [ ] Switch over per document type when accuracy is proven
- [ ] Keep Document AI as an optional fallback during the transition

---

## 8a. Findings

### Goal 1: inventory of `E:\final-recon-sets` (2026-10-14)
- 202 files: 115 PDF, 31 zip, 23 md, 22 eml, 5 xlsx, 4 jpg, 1 xls, 1 csv. No unreadable files.
- 253 PDF pages: **205 digital (81%)**, **48 scanned or image-only (19%)**.
- By file: 95 fully digital, 19 fully scanned, 1 mixed.
- Most PDFs are short: 59 have 1 page, 26 have 2 pages, the longest has 12.
- Scanned PDFs appear in 10 of the 16 case folders, so scans are spread across vendors and not one outlier.
- Several files appear to be duplicates (identical size and page count in different folders). Dedupe before building the gold set.
- Caveat: "digital" only means a text layer exists (at least 25 characters per page). Its quality is not yet checked. An old OCR layer on a scan would count as digital.

Implication: roughly four of five pages need no OCR at all, which supports the router design (text layer first). The remaining 19% are the pages where engine quality matters.

### Goal 2: dedupe and gold-set candidates (2026-10-14)
- Script: `scripts/gold_candidates.py`. Outputs `reports/duplicates.csv` and `reports/gold_candidates.csv` (names and hashes only, git-ignored).
- 119 PDFs and images, **96 unique by content**, 23 duplicate copies. Unique mix: 77 digital, 14 scanned, 1 mixed, 4 images.
- Proposed gold set: **40 files, about 105 pages** across 19 folders (21 digital, 14 scanned, 1 mixed, 4 images). Every unique hard case (scan, mixed, image) is included, and digital files fill the rest round-robin across vendors.
- 105 pages is more than the 30-50 pages planned for hand-labelling. Options: label only the key fields on page 1 of each file (fast), or trim the long multi-page files.
- **Coverage gap:** the whole sample set is procurement paperwork (invoices, GRVs, cargo documents). It has no KYC documents, cheques, forms or handwriting. Priority 2 and 3 domains cannot be benchmarked until we get those samples. The engine must not be declared ready on this data alone.

### Goal 3: gold-set labels (2026-10-19)
- Documents were read by Claude (text layers read locally; scans and photos rendered to images and viewed). **This means their content passed through the model provider.** The owner authorized it for this one-off labelling of procurement paperwork. It is not a precedent for KYC or banking documents.
- Labelling rules: PO vendor = supplier. Customs declaration vendor = declarant (importer/exporter field). Delivery advice and invoice vendor = issuer on the letterhead. Date is ISO (YYYY-MM-DD). Amounts have no thousands separators. `-` means the field is absent.
- Mix of the 40 files: 10 invoices, 6 customs declarations, 6 delivery advices, 5 transport chits (photographed handwritten invoices), 5 purchase orders, 4 photos with no text fields, and one each of shipping advice, ownership-transfer letter, blank page and a non-business document.
- Surprises that matter for the engine:
  - **Most documents are not invoices.** An "invoice extractor" would miss over half of this set. Document-type detection is a core need.
  - **Handwriting and photos:** 5 transport chits are phone photos of handwritten invoices. Hardest class here.
  - **One "scanned copy" PDF is completely blank.** The quality gate must detect empty pages and flag them, not return empty text as success.
  - **Scrambled text layers:** several digital PDFs return text in a jumbled order (side-by-side tables, text overlays), so a text layer is not automatically usable. The engine needs layout-aware reading order even for digital PDFs.
  - **Mixed number formats:** `59.700,00` (European) vs `59,700.00`. Normalization must handle both.
  - **Bundles:** a single PDF may hold several documents (delivery advice + customs form + HS code form + invoice). Page-level classification and splitting are needed.
  - **Some totals live on a later page** (TP-Link total is on page 2 of 7), so page-1-only reading is not enough.
  - Two files are synthetic test documents and one is a project plan, not a customer document.
- Label caveats: a few values were inferred (Lenovo AE4915167790 total from line items, Mitsumi invoice total from the item table, one handwritten reference with an unclear first digit). They are marked in the notes column.

### Goal 4: engine smoke test, page 1 of three gold files, laptop CPU (2026-10-19)
| File | Engine | Sec | Words | Conf | Label doc number found |
|---|---|---|---|---|---|
| #3 digital invoice | text layer | 0.0 | 401 | n/a | yes |
| #3 digital invoice | Tesseract | 3.7 | 416 | 0.85 | yes |
| #3 digital invoice | docTR | 20.3 | 432 | 0.89 | yes |
| #34 clean scan | text layer | 0.0 | 0 | n/a | no (no text layer) |
| #34 clean scan | Tesseract | 1.7 | 322 | 0.83 | yes |
| #34 clean scan | docTR | 10.9 | 318 | 0.91 | yes |
| #6 handwritten photo | text layer | 0.0 | 0 | n/a | no (no text layer) |
| #6 handwritten photo | Tesseract | 1.6 | 79 | 0.55 | **no** |
| #6 handwritten photo | docTR | 2.3 | 98 | 0.88 | yes (see caveat) |

- All three engines run end to end. docTR default models: detection `fast_base` and recognition `crnn_vgg16_bn`, 123 MB in total, one-time download (load took 70 s including the download).
- The text layer is instant and exact where it exists, and absent for scans and photos, which confirms the router design.
- Tesseract was fast on clean scans but weak on the handwritten photo (confidence 0.55). docTR coped better there.
- docTR on CPU took 2 to 20 s per page. The first page includes warm-up, so these are not final timings. A GPU should cut this a lot.
- **Caveats:** "found" is a substring match of the labelled number in the whole output, so a short number like 6111 could match by chance. Confidence values come from different engines and are not comparable. Three files is not a benchmark. Goal 5 needs real scoring (field-level and character-level).
- Disk: D: has 3.3 GB free after installs.

### Goal 5: first benchmark, 40 gold files, all 105 pages, laptop CPU (2026-10-19)
Metric: does the labelled value appear anywhere in the engine's output (values normalized: dates, number formats, amounts). This measures **text recall for the downstream LLM**, not whether the right value was picked. "Router" = text layer when a page has at least 25 characters, otherwise docTR.

| Field | Text layer | Tesseract | docTR | Router |
|---|---|---|---|---|
| doc_number | 61% | 82% | 100% | 100% |
| doc_date | 59% | 84% | 89% | 89% |
| total_amount | 59% | 75% | 94% | 94% |
| vendor | 62% | 91% | 100% | 100% |
| **All fields** | 60% | 83% | **96%** (130/136) | **96%** |

| Document group | Text layer | Tesseract | docTR | Router |
|---|---|---|---|---|
| Digital PDF | 100% | 97% | 100% | 100% |
| Scanned PDF | 9% | 91% | 100% | 100% |
| Handwritten photo (transport chits) | 0% | 15% | **70%** | 70% |
| Plain photo | 0% | 50% | 100% | 100% |

Speed (laptop CPU): Tesseract 1.3 s/page, docTR 5.6 s/page. The router skips OCR on digital pages, so it is much faster than docTR alone at the same accuracy.

Findings:
- **docTR is the leading primary OCR engine.** Clear lead over Tesseract on scans and handwriting. Router matches docTR's accuracy at lower cost.
- **All remaining misses are handwriting:** the 4 handwritten transport chits miss dates (all four) and totals (two). Handwritten dates and amounts are the engine's weak spot, and the first target for a vision-language fallback.
- A scoring bug was found and fixed during the run (matcher accepted at most 2 decimals; one document prints 995224.546875). Lesson: inspect every miss before trusting the number.
- **Caveats:** (1) Presence anywhere in a document is lenient. A short value like a total of 150 could match by chance. (2) The test is order-independent, so it does **not** measure reading order, and scrambled text layers still score 100%. (3) 40 files, 6 handwritten fields: small sample, wide uncertainty. (4) Only English text, procurement documents only. KYC, cheques and non-English are untested. (5) Labels were made by Claude and not yet spot-checked by a human.

### Goal 6a: docTR tuning on the 5 handwritten chits (2026-10-19)
Script: `scripts/tune_doctr.py`. 20 labelled fields in total. Laptop CPU.

| Config | Fields found | Sec/page | Still missing |
|---|---|---|---|
| Baseline (200 dpi, fast_base + crnn_vgg16_bn) | 14/20 | 2.7 | dates in 4 files, totals in 2 |
| 300 dpi render | 14/20 | 2.6 | same as baseline |
| Contrast (CLAHE) | 15/20 | 2.5 | dates in 4, total in 1 |
| Tilt correction | 16/20 | 4.6 | dates in 3, total in 1 |
| Detector db_resnet50 | 14/20 | 3.4 | worse on #26 |
| Recognizer **parseq** | **17/20** | 4.9 | dates in 3 (#11, #15, #26) |
| Recognizer master | 15/20 | 26.7 | too slow, not better |

- parseq is the best single change (+3 fields), at about 2x the time. Tilt correction and contrast each help a little. Resolution and the heavier detector do not help.
- **These differences are 1 to 3 fields out of 20.** That is within what chance could produce on 5 files. Treat as hints, not proof. Handwritten dates are still the main failure in every config.
- parseq lost one field that the baseline had (date on #26), so it is not a pure win.
- Not yet tested: combinations (tilt + parseq), regression on the scanned and printed documents, speed on GPU.

## 9. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-19 | docTR is the leading primary OCR engine, used behind a text-layer-first router | Benchmark: 96% field recall vs Tesseract 83%. Provisional until Docling, Surya and a VLM are compared. |
| 2026-10-19 | Claude labelled the gold set instead of the owner | Owner request. One-off exception to "content never leaves our machines". Not for KYC. |
| 2026-10-13 | Run fully local, no external OCR services | Cost, free testing, PII |
| 2026-10-13 | No Chinese-origin models | Company constraint |
| 2026-10-13 | Router design: text layer first, OCR only when needed | Digital PDFs need no OCR |
| 2026-10-13 | LLM extraction stays a separate step on top of the engine | User decision. Engine produces text, positions and confidence. |

## 10. Open questions

- Which Document AI processor is used today (generic OCR, Invoice Parser, other)? Needed to know which fields the engine must reproduce.
- Which languages first, and which later?
- Production server GPU spec and expected pages per day.
- Accuracy targets: what does "good enough" mean for the business?
- Licenses: confirm commercial-use terms for Surya and each candidate vision-language model before depending on them.
- Is there a retention rule for storing customer documents used as test data?

## 11. Progress log

| Date | Session summary |
|---|---|
| 2026-10-14 | Goal 1 done: git initialized, venv and PyMuPDF 1.28.2 installed, inventory script written and run on the sample folder. Findings in 8a. |
| 2026-10-19 | Goal 5 done: first benchmark. Router 96% field recall, weak spot is handwriting (70%). Findings in 8a. |
| 2026-10-19 | Goal 3 done: all 40 gold files labelled by Claude, findings in 8a. Next: Goal 4, install and smoke-test OCR engines (owner installs, Claude guides). |
| 2026-10-13 | Discussed OCR fundamentals. Reviewed constraints (local, no Chinese models, real PII). Found sample set at `E:\final-recon-sets` (about 200 files, 115 PDFs). Created this plan. |
