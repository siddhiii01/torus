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
- [ ] Decide the labelling scope, then write reference values by hand for key fields and sample text regions.
- [ ] Collect samples for the other domains (KYC, cheques, forms, handwriting).
- [x] Record findings here (section 8a).

### Phase 2: Baseline benchmark
- [ ] Build a benchmark harness: run engine X on the gold set, score character accuracy and field accuracy.
- [ ] Run PyMuPDF, Tesseract, docTR, Docling, Surya. Compare on accuracy, speed and VRAM.
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

## 9. Decision log

| Date | Decision | Reason |
|---|---|---|
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
| 2026-10-13 | Discussed OCR fundamentals. Reviewed constraints (local, no Chinese models, real PII). Found sample set at `E:\final-recon-sets` (about 200 files, 115 PDFs). Created this plan. |
