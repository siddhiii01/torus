# Legacy

Read-only holding area for the pre-Torus `ocr/` and `scripts/` (moved here by P0.9).
Do not extend code here. Port useful parts into `src/torus/` with tests, then delete the legacy file.

These scripts were written to run from the repo root. They are kept for reference and for
the P1.6 baseline; paths such as `schemas/` and `reports/` and the `ocr` import resolve
against the old layout, so run them with `PYTHONPATH=legacy` and expect to adjust `ROOT`.

## Port map

| Legacy file | Replaced by | Phase |
|---|---|---|
| `ocr/engine.py` | `torus.ingest`, `torus.readers` | 2, 5 |
| `ocr/preprocess.py` | `torus.preprocess` | 2 |
| `ocr/comb_merge.py`, `ocr/digit_crop.py` | `torus.layout.comb`, `torus.readers.char_cell` | 4, 5 |
| `ocr/crop_vlm.py` | `torus.readers.general_vlm` (invented confidence formula is dropped; see `torus.consensus`) | 5, 6 |
| `ocr/fields.py` | `torus.layout.anchor` + template field specs | 4 |
| `ocr/pin_master.py` | `torus.validate.lookup` | 7 |
| `scripts/deb_score.py`, `score.py` | `torus.eval` | 1 |
| `scripts/label.py`, `deb_app.py` | `torus.eval.label`, `torus.review` | 1, 9 |
| `scripts/review.py`, `deb_review.py` | `torus.review` | 9 |
| `scripts/deb_extract.py` | `torus run` (customer-specific fixes removed, not ported) | 8 |
| `scripts/deb_bakeoff_*`, `tune_doctr.py` | `torus.eval.readers_bakeoff` | 5 |
| `scripts/run_gold.py`, `gold_candidates.py`, `inventory.py` | `torus.eval.promote_legacy` / one-off, delete | 1 |
| `scripts/check_engine.py`, `smoke_test.py`, `run_ocr.py` | reader smoke tests in `tests/readers/` | 5 |
