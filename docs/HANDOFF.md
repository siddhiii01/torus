# Handoff: where Torus stands (20 November 2026)

For the engineer picking this up on a stronger machine. Read this first, then `docs/TESTING.md`
for exact commands and `docs/SETUP.md` for models.

## One-line status

All 12 phases of `docs/IMPLEMENTATION_PLAN.md` are **built and unit-tested on synthetic data**.
Almost nothing has been **measured on real documents**. Treat the code as ready for evaluation, not
as finished.

## What exists

- `src/torus/`: core models, config, ingest, preprocess, classify, layout, readers, consensus,
  validate, mapping, output (Excel), review, learn, service, eval. One package per plan phase.
- `config/`: pipeline, readers, page types, service, train recipes, and templates
  (`deb_v1`, `invoice_v1`, `starter_v1`).
- `legacy/`: the old `ocr/` and `scripts/`, read-only, with a port map in `legacy/README.md`.
- Tests: `python -m pytest -m "not model"` passes (about 1280 tests, synthetic data only).
  Lint (`ruff`) and types (`mypy`) are clean.
- The client workbook `DEB File 60 Cases.xlsx`, sheet "56 accounts" (header row 2), matches
  `config/templates/deb_v1.yaml` exactly: 73 columns, nothing added, removed or moved.

## What is NOT verified

| Area | State |
|---|---|
| Accuracy on real documents | **Never measured.** Only 3 labelled cases exist (plan needs 20, 5 hidden). No hidden set. |
| Readers | Only Tesseract has been run, and only on a few pages. Paddle, docTR, TrOCR, PARSeq and all VLM readers have **never run on real weights**. The 13 `model` tests are skipped. |
| Reader bake-off, calibration | Not run. All readers are `enabled: false` in `config/readers.yaml`; no calibrator exists, so **nothing can be `confirmed`**. |
| First real result | A full 26-page pack with Tesseract only: **all 41 fillable fields came out `not_found`**, nothing written. Cause not investigated: could be weak reader output, label matching, or a wiring problem. Priority one for the next person. |
| Speed | Fixed for the pack tested; see below. Other readers are untimed. |
| Review loop | **Gap:** nothing builds a review bundle from a real run. `torus review` and `torus reexport` need a bundle JSON (`torus.review.bundle.ReviewBundle`: the result plus every reading plus the source path), but `torus run` never writes one. The UI is tested only on hand-made bundles. |
| Learning loop (P10) | Tooling exists. `corrections/` is empty and no fine-tune has ever run. |
| Service (P11) | Tested on synthetic packs with workers. Docker images were never built or run. |
| Throughput | One row in `docs/SETUP.md` section 7 (no readers enabled). |

## Speed findings

Two full-resolution `cv2.HoughCircles` searches dominated run time and were fixed:

- Radio-button detection (`layout/checkbox.py`) took 77% of one page (177 of 230 s). It now runs on a
  downscaled copy (`hough_max_side`, default 1000 px).
- Stamp detection (`layout/classic.py`) took 696 s of a 729 s page. It now runs on a downscaled mask
  (`stamp_hough_max_side`, default 800 px). That page went from 712 s to 17 s.
- Regression tests cover both (`tests/component/test_layout_checkbox.py`, `test_layout_stamp.py`).
  Outputs were identical before and after on the pages checked.

Result on this laptop (i5-1235U, no GPU, Tesseract only): one 26-page pack takes about 144 s of
per-page time (1-17 s a page) plus preprocessing, a few minutes in all. Before the fixes, 3 pages
took over 12 minutes. `torus run` prints per-page progress to stderr.

Tesseract itself is cheap (about 0.1 s per read). Other readers, especially the VLMs, will be far
slower; time them on the target machine.

## Decisions still open

1. **Git history purge.** Customer PDFs and answer keys for 3 cases are deleted from the working tree but
   remain in git history. Owner approval needed. Until then keep the repo **private**.
2. **Passthrough columns.** None are marked. Likely candidates (needs confirming): checker columns
   55-59 and system columns 2-4.
3. **Second sheet "4 accounts"** (71 columns) is not integrated. Is it a variant layout that needs its own
   template?
4. **Model licences.** `docs/SETUP.md` lists several as "check model card". Only models cleared for the
   client's production use should be enabled.
5. **Reviewers** (who, how many packs a day). Not decided; nothing is blocked.
6. The workbook contains about 60 filled cases. If the matching source PDFs exist, they would give answer
   keys and speed up labelling.

## Suggested first days

1. Set `TORUS_DATA_DIR` and `TORUS_MODEL_DIR` (outside the repo). Copy over the test cases and the client
   workbook from `torus-data/` by file transfer, not git.
2. `python -m torus doctor`. Install the extras your machine can use (`paddle`, `doctr`, `vlm`, `review`).
3. Enable readers in `config/local.yaml` (git-ignored), then run one real page with `torus run`.
4. Find out why every field comes back `not_found` on a full pack (see the first-result row above).
5. Label 20 packs (5 hidden) with `torus label`.
6. Run the reader bake-off, fit calibration, set thresholds so the dev set has zero silent errors.
7. Build the review-bundle writer (see the gap above), then try the review loop with real reviewers.

## Data safety

Real documents, answer keys and workbooks live only under `TORUS_DATA_DIR`. Never commit them. A
pre-commit hook blocks PDFs, images, xlsx and json under data paths. Logs and reports carry metadata
only, never document text.
