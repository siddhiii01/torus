# OCR Engine: Production Plan

Owner: Celeris Ventures (AI team) · Started: 2026-10-13 · This revision: 2026-10-28

This is the living document. The production plan below replaces the earlier phase list as the way we work. Benchmark findings stay in section 8a so we do not lose them.

---

## Starting line

**Today, on this laptop, before any new model.**

The machine is a 12th-gen Intel i5-12450H (8 cores / 12 threads), 16 GB RAM, NVIDIA GeForce RTX 2050 with 4 GB VRAM, plus Intel UHD graphics. Python 3.12.10 and the project venv are installed. D: has about 2.9 GB free. C: has about 27 GB free. E: has about 14 GB free.

The repo is a benchmark harness, not an engine. `scripts/` can inventory a folder, pick a gold set, cache words from the PDF text layer, Tesseract, and docTR, and score whether a labelled field appears somewhere in that text. Bounding boxes are read and then thrown away. There is no service, no shared output schema, and no review screen.

What we already know from the 40-file gold set (section 8a): docTR behind a text-layer-first path finds 96% of labelled fields somewhere in the text. That number does not mean the page was read correctly. It ignores reading order, so a scrambled text layer still scores perfectly. Handwritten transport chits are the weak class (about 70%). One scanned PDF is blank. Some digital PDFs return text in a jumbled order. A single PDF can hold several documents.

The sample folder `E:\final-recon-sets` was re-counted on 2026-10-28:

| | Count |
|---|---|
| Files | 214, across 29 case folders |
| PDFs | 125 (105 fully digital, 19 fully scanned, 1 mixed) |
| PDF pages | 280 (232 with a text layer, 48 scanned or image-only) |
| Images | 4 jpg |
| Other | 33 zip, 22 eml, 23 md, 5 xlsx, 1 xls, 1 csv |

Most PDFs are short: 64 have 1 page, 27 have 2, the longest has 12. Since the 14 October inventory the folder grew by digital pages. The scanned-page count is still 48, so the existing gold set still covers the hard cases. Zips, email, markdown, and spreadsheets are out of scope. The engine reads PDFs and images.

The old 96% figure stays as a hint about which reader to start with. It is not the acceptance test.

## Ending line

A local OCR engine that can take Document AI’s place for **English procurement documents of the kinds in this folder**: digital PDFs, scans, phone photos, handwriting, poor scans, multi-page files, and blank pages.

A file goes in. For every page the engine returns:

- the words on that page, in reading order
- lines and words with bounding boxes and confidence
- which reader ran
- a status of `ok` or `needs_review`

No coordinates are hardcoded for these vendors. A new procurement PDF of the same kinds must go through the same code with no new rules.

**Done, for this document family,** means all of the following:

1. A fixed review set (the current 40 gold files, about 105 pages, plus any new scan or photo that shows up later) has been checked by eye: page image on one side, engine output on the other.
2. On a passing page, the text a person can read on the page is in the output, in an order a person can follow, with boxes on the right words. Blank pages are flagged, not returned as a successful empty read.
3. Pages that still fail stay on a short list with the failure class (handwriting, unreadable scan, scrambled order, no text at all). They are not silently shipped.
4. A procurement PDF that was not in the review set runs on the same path and survives the same kind of check.
5. Only then do we wrap it as a service and talk about switching production off Document AI for this family.

Other families (KYC, cheques, forms, other languages) are a later ending line. They use the same engine. They do not change this one.

The engine reads the page. It does not fill an invoice form. Vendor, amount, and date extraction stays downstream, on top of this output.

## How we build it

Work is a loop, not a model bake-off against Document AI. There will be no Document AI score to chase. We open a real file, compare, name the reason, and change the engine.

### 1. Engine contract, on this laptop

One pipeline, one JSON schema:

`file -> pages -> reader -> words, boxes, confidence, reading order, status`

Readers plug in behind that schema. The first two readers are the ones already installed:

- **Text layer** (PyMuPDF), only when the page actually has usable embedded text. A scrambled layer is a failure, not a success. Usable means the words come back in an order that follows the page, not merely that 25 characters exist.
- **docTR** (Mindee, France; now maintained in Germany; Apache 2.0) for scans, photos, and any page whose text layer is missing or scrambled. Keep the boxes. Default models first: `fast_base` + `crnn_vgg16_bn`. Run it on the GPU when it fits.

Tesseract stays installed as a second reader we can call on a page docTR reads badly. It is not the main path. It was weaker on handwriting (15% field recall there).

This step also adds a local review view: the page image, and the words and boxes the engine produced. That view is the test. It writes no document text to git or to logs. Logs stay metadata: page count, reader name, timings, confidence summary, status.

### 2. Review the pages we have

Run that pipeline on the gold set. Review by class, not by staring at all 280 pages on day one:

| Class | Why it is in the first review |
|---|---|
| Digital PDF with a clean text layer | Must be exact and cheap |
| Digital PDF with scrambled order | Known failure. Text layer is not automatically trusted |
| Clean scan | docTR already looks strong here. Confirm full-page reading and boxes, not just one field |
| Handwritten photo (transport chits) | Known weak class |
| Plain photo, blank page, mixed PDF, multi-page bundle | Each has already bitten us once |

For each miss, record the class and the cause. Causes we already expect: reading order, missed handwriting, a blank page treated as success, a box on the wrong line, a word split or merged. Fix one class, re-run those pages, review again.

### 3. Change the reader only when a class is still failing

Order on this 4 GB GPU. Stop at the first reader that fixes the class. Do not download the next one speculatively.

| Order | Reader | Origin | When |
|---|---|---|---|
| 1 | PDF text layer, layout-aware | PyMuPDF | Digital pages |
| 2 | docTR default | France / Germany, Apache 2.0 | Scans and photos |
| 3 | docTR `parseq`, then tilt or contrast if a review page shows it | same | A class docTR default still misses. Earlier tuning moved 14/20 to 17/20 handwritten fields. That is a hint, re-check on the full page |
| 4 | Tesseract `eng` | Google / US, already on the machine | Second opinion on a page docTR mangles |
| 5 | TrOCR or Florence-2 | Microsoft, US | Handwriting still failing, and the model loads in 4 GB |
| 6 | Docling | IBM, US | Tables and layout still wrong, and it fits in 16 GB RAM |
| 7 | Surya OCR 2 (650M) | Datalab, US | Only after a license check. Code is Apache 2.0. Weights are a modified OpenRAIL-M: free under a $5M revenue and funding line, and not for a product that competes with Datalab. Do not download until that is confirmed |

Indian-origin models are allowed. For English page OCR on this laptop, none of the readers above is Indian. When a later language needs one, we add it the same way: it has to load here, and it has to fix a class we can see.

### 4. Flag, then stop

The flag is a written note in this file and in the session, with the failure class, what was tried, and what the machine did. It is the list to take to seniors. We do not keep stacking models past the point this laptop can run.

**Flag now, before any new download:** D: has about 2.9 GB free. New weights go on C: (or E:), not on D:.

**Flag and stop when any of these happens:**

- The next allowed model does not load in 4 GB VRAM or 16 GB RAM.
- A page takes so long that reviewing the gold set is impractical.
- Handwriting or poor scans still fail after every reader in the table above that actually loaded.
- A page produces no usable text from every allowed reader. That page, and only that page, may try a Chinese-origin model (PaddleOCR, Qwen-family, MinerU, DeepSeek-OCR, and anything whose base model is one of those, including olmOCR while it is Qwen-based). One such page is already a reason to ask for a bigger machine rather than to adopt that model as the default.

The bigger machine is for a larger fallback reader and for speed. The server comes after a reader is chosen, so we host something we have already seen work. Hosting does not fix a reader we have not reviewed.

### 5. Service, after the review set passes

FastAPI on this machine first, so the call shape is real. Metadata-only logs. Then the same image on the server the seniors provide. Production switches off Document AI for this document family only after step 2’s ending-line checks pass on an unseen procurement file.

## What we will not do

- Hardcode a vendor’s layout, coordinates, or field positions.
- Treat “the labelled invoice number appeared somewhere” as proof the page was read.
- Send a document, image, or OCR text to an external API.
- Put document content in git, in logs, or in this plan beyond the failure classes already recorded.
- Start with KYC, cheques, or other languages.
- Use a Chinese-origin model as a normal reader.

## 16-hour plan (week 1 and week 2)

Budget is **16 hours total**. Week 1 and week 2 get **8 hours each**, in two **4-hour** blocks. Same shape both weeks: run, look, fix, write the result down. No block is “explore”.

Clock already spent on 29 Oct: about 2 hours. The pipeline keeps boxes, and `reports/review/index.html` exists for gold #03, #34, #06, and #37. The blank page returns `needs_review`. Those 2 hours are Block 1a. **14 hours remain.**

**Done at hour 16** means all of this is true:

1. All 40 gold files have been through `ocr/engine.py`.
2. Every failed page is on a class list in this file (reading order, handwriting, empty, bad boxes, or no text). No document text in the list.
3. The top class has been fixed and those pages re-checked, or it is a hardware flag because the next reader does not fit in 4 GB.
4. A local service on this machine accepts a PDF or image and returns the same JSON: words, boxes, confidence, reader, `ok` or `needs_review`. Logs are metadata only.
5. One procurement PDF that is not in the gold set has been run through that service and checked by eye.

Server hosting, KYC, other languages, Surya, and any Chinese-origin model are outside these 16 hours. A page that still has no usable text at hour 16 is the flag for seniors, not a reason to start a new model track inside this budget.

### Week 1 — 8 hours — 29 and 30 Oct

| Block | Hours | Date | Work | Stop when |
|---|---|---|---|---|
| 1a | 2 | Thu 29 | Engine, boxes, review of #03 #34 #06 #37. **Done.** | Review file exists |
| 1b | 2 | Thu 29 | Review by us. Reading order fixed. Field/value JSON added. Intern parallel. **Done.** | Four-file verdicts written |
| 2 | 4 | Thu 29 | All 40 gold files through the engine. **Done.** Class list below. | Class list written |

### Week 2 — 8 hours — 2 and 3 Nov

| Block | Hours | Date | Work | Stop when |
|---|---|---|---|---|
| 3 | 4 | Mon 2 | One change aimed at the top class only. If it is handwriting: docTR `parseq` on the failed chits. If it is order: layout-aware text layer. Re-run only the failed pages. Keep the change only when the page looks better. | Failed pages re-checked, or a hardware flag if the model will not load |
| 4 | 4 | Tue 3 | Hours 1–2: local HTTP service, same JSON, metadata log. Hour 3: one unseen procurement file through the service, checked by eye. Hour 4: write the hour-16 result in this file. Passes and the remaining flags. Then stop. | Ending checklist above is filled in |

Weights stay in `~\.cache` on C:. docTR stays on the CPU build that is already installed. A download that does not fit on C:, or a model that does not load in 4 GB, ends that block early and becomes the flag.

### Day 1 check (29 Oct, our pass, no document text)

Reviewed the 10 pages in `reports/review` with boxes drawn on the page images. Intern instructions are in `REVIEW_FOR_INTERN.md`. Their notes are a second pass. The fix does not wait on them.

| Page | Boxes | Order | Words | Next action |
|---|---|---|---|---|
| #03 digital invoice | On the words | Fixed. Visual order, across the row. | Printed text is present | Keep. Field/table JSON added on top |
| #34 pages 1–4, scan invoice and packing lists | On the words | Rows run left to right. Letterhead grouping is slightly odd | Printed lines, quantities, and totals are readable | No model change from these pages |
| #34 page 5, delivery advice | On the labels | Usable | Ticked checkboxes are not recorded as ticked. Stamp text is partial | Note. Not today’s fix |
| #34 page 6, ownership and acceptance | On the printed lines | Usable | Signature not read. Stamp text partial | Note |
| #34 page 7, HS code list | On the words | Rows run left to right, totals included | Printed table readable. Stamp partial | No model change |
| #06 handwritten chit | On printed and handwritten words | Top to bottom | Printed form is fine. Handwriting: year digit wrong, a short name truncated, a reference prefix wrong, one garbage token. Arabic and a truck number not read | Week 2 block 3. English handwriting only. Arabic stays a note |
| #37 blank | None, correctly | Empty | Empty, status `needs_review` | Pass |

### Gold run, all 40 files (29 Oct)

`scripts/run_gold.py` on `E:\final-recon-sets`. 105 pages. JSON in git-ignored `reports/gold_run/`.

| | Count |
|---|---|
| Files | 40 |
| Pages | 105 |
| Words | 22,287 |
| Field pairs | 757 |
| Table rows | 737 |
| Status `ok` | 38 files |
| Status `needs_review` | 2 files: #17 photo (`low_confidence`), #37 blank (`empty`) |
| Readers | 21 text layer only, 18 docTR only, 1 mixed |

Class list for Week 2 (no document text):

| Class | Evidence | Next |
|---|---|---|
| Handwriting | Transport chits #06, #11, #15, #18, #26 return words but known digit/name errors from Day 1 | Block 3: docTR `parseq` on these five only |
| Empty page | #37 | Already flagged. Keep |
| Low confidence photo | #17 | Inspect in Block 3. May stay `needs_review` |
| Checkbox / stamp | Seen on #34 | Note. Not the top fix |
| Field/table pairing | Geometry-based. Better after order fix. Still imperfect | Spot-check during Block 3; no vendor templates |

## Hardware flag to take upstairs

This laptop can build and review the engine. It cannot host a 7B–12B vision model, and D: cannot hold more weights.

Ask when we hit the stop conditions above. The ask is a GPU machine with enough VRAM for a larger fallback reader (the 16 GB machine already mentioned in the old plan is the right class), plus disk for weights. The server is the step after that reader earns a place in the pipeline.

---

## 8a. Findings (history)

These notes are from 14–19 October 2026. The production plan above is the operating plan. The 96% score is field presence, not page-reading quality.

### Goal 1: inventory of `E:\final-recon-sets` (2026-10-14)

- 202 files: 115 PDF, 31 zip, 23 md, 22 eml, 5 xlsx, 4 jpg, 1 xls, 1 csv. No unreadable files.
- 253 PDF pages: **205 digital (81%)**, **48 scanned or image-only (19%)**.
- By file: 95 fully digital, 19 fully scanned, 1 mixed.
- Most PDFs are short: 59 have 1 page, 26 have 2 pages, the longest has 12.
- Scanned PDFs appear in 10 of the 16 case folders, so scans are spread across vendors and not one outlier.
- Several files appear to be duplicates. Dedupe before building the gold set.
- "Digital" only means a text layer exists (at least 25 characters per page). An old OCR layer on a scan would count as digital.

Refreshed 2026-10-28: 214 files, 125 PDFs, 280 pages, 232 digital, 48 scanned. Scanned count unchanged. 29 case folders. See the starting line.

### Goal 2: dedupe and gold-set candidates (2026-10-14)

- Script: `scripts/gold_candidates.py`. Outputs `reports/duplicates.csv` and `reports/gold_candidates.csv` (names and hashes only, git-ignored).
- 119 PDFs and images, **96 unique by content**, 23 duplicate copies. Unique mix: 77 digital, 14 scanned, 1 mixed, 4 images.
- Proposed gold set: **40 files, about 105 pages** across 19 folders (21 digital, 14 scanned, 1 mixed, 4 images). Every unique hard case is included.
- The sample set is procurement paperwork. KYC, cheques, and forms are not in it. The engine is not finished for those families on this data.

### Goal 3: gold-set labels (2026-10-19)

- Documents were read by Claude. **Content passed through the model provider.** The owner authorized it for this one-off labelling of procurement paperwork. It is not a precedent for KYC or banking documents. Further review is local, by eye.
- Mix of the 40 files: 10 invoices, 6 customs declarations, 6 delivery advices, 5 transport chits (photographed handwritten invoices), 5 purchase orders, 4 photos with no text fields, and one each of shipping advice, ownership-transfer letter, blank page, and a non-business document.
- Most documents are not invoices. Handwritten chits are the hardest class. One scanned PDF is blank. Several digital text layers are scrambled. Number formats mix `59.700,00` and `59,700.00`. A single PDF may bundle several documents. Some totals sit on a later page.

### Goal 4: smoke test, page 1 of three gold files, laptop CPU (2026-10-19)

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

docTR default models: detection `fast_base`, recognition `crnn_vgg16_bn`, 123 MB. "Found" is a substring match. Three files is not a page-reading test.

### Goal 5: field-presence run, 40 gold files, 105 pages, laptop CPU (2026-10-19)

Metric: does the labelled value appear anywhere in the output. Router = text layer when a page has at least 25 characters, otherwise docTR.

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
| Handwritten photo | 0% | 15% | **70%** | 70% |
| Plain photo | 0% | 50% | 100% | 100% |

Speed on CPU: Tesseract 1.3 s/page, docTR 5.6 s/page. Every remaining router miss was handwriting (dates, and some totals). The test does not measure reading order. Labels were not spot-checked by a person.

### Goal 6a: docTR tuning on the 5 handwritten chits (2026-10-19)

| Config | Fields found | Sec/page | Still missing |
|---|---|---|---|
| Baseline (200 dpi, fast_base + crnn_vgg16_bn) | 14/20 | 2.7 | dates in 4 files, totals in 2 |
| 300 dpi render | 14/20 | 2.6 | same as baseline |
| Contrast (CLAHE) | 15/20 | 2.5 | dates in 4, total in 1 |
| Tilt correction | 16/20 | 4.6 | dates in 3, total in 1 |
| Detector db_resnet50 | 14/20 | 3.4 | worse on #26 |
| Recognizer parseq | 17/20 | 4.9 | dates in 3 (#11, #15, #26) |
| Recognizer master | 15/20 | 26.7 | too slow, not better |

Treat 1–3 fields out of 20 as a hint. Handwritten dates failed in every config. Combinations and a full-page re-review are not done.

## 9. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-28 | Success is a manual page-vs-output review, not a Document AI score | No Document AI dump will be provided. The page is the reference. |
| 2026-10-28 | No per-vendor geometry | These files are the first set, not the only layouts the engine will see. |
| 2026-10-28 | Chinese-origin models are a last resort on a page that no allowed reader can read | Allowed readers are US, Indian, French, and other non-Chinese models. A page that needs a Chinese model is a hardware flag. |
| 2026-10-28 | Build and review on this laptop. Stop and ask for a bigger GPU when the next reader does not fit or a class still fails | RTX 2050 4 GB, 16 GB RAM. Server hosting comes after the reader is chosen. |
| 2026-10-28 | Scope now is English procurement PDFs and images in `E:\final-recon-sets` | Other document families later, same engine. |
| 2026-10-19 | docTR is the first OCR reader, behind the text layer | 96% field presence vs Tesseract 83%. Provisional until page-level review. |
| 2026-10-19 | Claude labelled the gold set | Owner request. One-off. Not for KYC. |
| 2026-10-13 | Run fully local, no external OCR services | Cost, free testing, PII. |
| 2026-10-13 | Text layer first, OCR only when the layer is missing or unusable | Most pages in this folder are digital. |
| 2026-10-13 | Field extraction stays downstream | The engine returns text, positions, and confidence. |

## 10. Open questions

- Surya OCR 2 weights: is Celeris under the $5M line, and is internal replacement of Document AI allowed under that license? Do not download until this is answered.
- Retention rule for customer files used as the review set. They stay on this machine either way.
- Production pages per day, once we size the server. Not needed to start.

## 11. Progress log

| Date | Session summary |
|---|---|
| 2026-10-29 | Blocks 1a, 1b, and 2 done same day. All 40 gold files run. 38 ok, 2 needs_review. Next: handwriting (`parseq`) then the service. |
| 2026-10-29 | 16-hour plan set: 8h this week, 8h next week. Block 1a done (engine and four-file review). 14 hours left. |
| 2026-10-29 | Week 1 started: engine keeps boxes, review view for gold #03, #34, #06, #37. |
| 2026-10-28 | Production plan rewritten around manual review, this laptop, and an explicit stop-and-ask for a bigger machine. Inventory refreshed: 214 files, 280 PDF pages, 48 still scanned. |
| 2026-10-19 | Field-presence run: router 96%, handwriting 70%. Gold set labelled. Smoke test passed. |
| 2026-10-14 | Inventory, dedupe, gold candidates. |
| 2026-10-13 | Constraints and this plan started. Sample folder found at `E:\final-recon-sets`. |
