# SKILLS.md: recipes for recurring tasks

Each skill is a fixed procedure an AI agent follows for a common type of task. Pick the matching skill, follow its steps in order, and finish with its **Done when** checks. If no skill fits, follow the general workflow in `AGENTS.md` §6, and propose a new skill here once the task has come up twice.

Every skill assumes the rules in `AGENTS.md`: no hard-coding, config over code, synthetic tests only, no document text in logs or git.

---

## S1. Add or change an Excel column (field spec)

Use when the client template gains a column, or a column's meaning changes.

1. Open `config/templates/<template_id>.yaml`. Add or edit the field spec:
   ```yaml
   - key: mobile_no
     excel_header: "Mobile No"          # exact header text in the client's Excel
     description: "Primary 10-digit mobile number of this person"
     type: phone_in                     # one of the types in torus.core.types
     charset: digits
     length: 10
     validators: [indian_mobile]
     look_in: [application_form, verification_report]   # page types
     cardinality: per_person            # per_pack | per_person | per_address
     critical: true                     # critical fields need 2 reader families to confirm
     synonyms: ["Mobile", "Mob. No.", "Cell"]   # printed label variants (any language)
   ```
2. If the type or validator does not exist, use S3 first.
3. Regenerate the template from the client Excel if headers changed: `python -m torus.output.template sync <client.xlsx> <template_id>`. This only reads headers, never data.
4. Add a synthetic integration case that contains the field (`tests/fixtures/synth/cases/`).
5. Run `pytest tests/integration -k <template_id>` and the evaluation.

**Done when:** the config validates, the synthetic case fills the column correctly, and the evaluation shows the column's accuracy and review rate.

## S2. Add a new reader (OCR or vision model)

1. Read `docs/SETUP.md` for the model's hardware needs and licence. Record the licence in the PR.
2. Create `src/torus/readers/<family>/<name>.py` implementing the `Reader` protocol. Import heavy libraries inside the class, not at module top.
3. Use the `ReadHint` (charset, script, length) to constrain the model: a prompt for VLMs, or decoding constraints for CTC/attention readers.
4. Return real confidences if the model gives them (token log-probs, character scores). If it gives none, return `None`. **Never invent a confidence.**
5. Register it in `config/readers.yaml` with `family`, `capabilities`, `min_vram_gb`, `min_ram_gb`, `device`, `enabled: false`.
6. Add `tests/readers/test_<name>.py` with `@pytest.mark.model`: it loads, reads a synthetic printed crop and a synthetic handwritten crop, and returns the right type.
7. Add an optional extra in `pyproject.toml` if it needs new packages.
8. Run the reader bake-off: `python -m torus.eval readers --set dev --reader <name>`. Attach the per-field-type accuracy table.

**Done when:** the smoke test passes on a machine that meets the requirements, the pipeline still runs with the reader disabled, and the bake-off table is in the PR.

## S3. Add a validator or field type

1. Add the validator to `src/torus/validate/` as a pure function behind the `Validator` protocol. Examples: Verhoeff checksum for Aadhaar, the PAN pattern, the EPIC (voter ID) pattern, a PIN that must exist in the lookup, a believable date of birth.
2. Patterns and tables that describe *public standards* (ID formats, PIN lists) are allowed in code or in `resources/lookups/`. Anything about one client or one document is not.
3. Unit tests: valid examples, invalid examples, edge cases (spaces, OCR confusions such as O/0 and I/1). Use made-up values that satisfy the format, never real IDs.
4. Register it by name so field specs can reference it.

**Done when:** unit tests cover pass, fail and normalisation, and S1 can reference it.

## S4. Add a new page type or document family

1. Add the page type to `config/page_types.yaml` with a description and the visual and text cues a human would use.
2. Add 3+ synthetic examples (generator) and, outside the repo, label real examples in the test set.
3. Retrain or re-prompt the classifier: `python -m torus.classify train` (small model) or update the zero-shot prompt (VLM).
4. Update any field spec `look_in` lists that should search this page type.
5. Run the classification evaluation: per-type precision and recall, plus the confusion matrix.

**Done when:** classification metrics for the new type are reported and no existing type gets worse.

## S5. Investigate an extraction failure

Use when a human reports a wrong or missing value.

1. Reproduce on the real file locally: `python -m torus.debug trace <file> --field <key>`. This writes a local HTML trace (git-ignored) showing the page, located region, every reader's output, the vote, the validators and the mapper decision.
2. Name the **failure class**: wrong page type, region not found, region found but misread, readers disagree, validator wrong, mapper chose the wrong candidate, or the answer key is wrong.
3. Build a **synthetic** test that reproduces the class (not the document).
4. Fix at the right layer (see the table below). Never patch the output string.
5. Re-run the evaluation and confirm that no other column regressed.

| Failure class | Fix layer |
|---|---|
| Wrong page type | classifier data or prompt (S4) |
| Region not found | `layout/` anchor logic or family hints in config |
| Misread | add or replace a reader (S2), voting weights, or fine-tuning (S7) |
| Readers disagree, wrong winner | `consensus/` |
| Validator too strict or loose | `validate/` (S3) |
| Wrong column or wrong person | `mapping/` prompt or field spec description (S1) |

**Done when:** a synthetic regression test exists, the evaluation shows the fix, and the failure-class tally is updated in the eval report.

## S6. Run an evaluation and report it

1. `python -m torus.eval run --set <dev|hidden> --pred-dir <dir> --report reports/eval/<date>.md`
2. The report shows per column: accuracy, silent errors, `confirmed` / `likely` / `needs_review` / `not_found` rates; per page type: precision/recall/F1, confusion matrix, and misclassifications by gold type (when `pages.json` + page preds exist); field location located-rate / mean IoU (when `locations.json` + location preds exist); plus timing. For classification-only: `python -m torus.eval classify --set <dev|hidden> --pred-dir <dir> --report reports/eval/<date>_classify.md`. For location-only (layout before reading): `python -m torus.eval locate --set <dev|hidden> --pred-dir <dir> --report reports/eval/<date>_locate.md`.
3. Compare with the last report on `main`:
   `python -m torus.eval diff reports/eval/baseline.json reports/eval/latest.json`
   (exit 1 if silent errors rise or a column drops beyond `noise_margin` in
   `config/pipeline.yaml`).
4. Paste the summary table (no document text) into the PR.

**Done when:** the table is in the PR and every regression is explained or fixed.

## S7. Fine-tune a reader on corrections

1. Export the review corrections: `python -m torus.learn export --since <date>`. Crops and labels go to `TORUS_DATA_DIR/training/`, never to git.
2. Split by document (not by crop), so no document is in both train and test.
3. Run the fine-tune config `config/train/<reader>.yaml`. Record the base model, data size and hyperparameters.
4. Evaluate the new weights on the hidden set against the current weights.
5. Register the new weights in the model registry with a version. Switch `config/readers.yaml` only if accuracy goes up and silent errors do not.

**Done when:** before/after numbers are in the PR and the registry has the new version with its data hash.

## S8. Add a synthetic fixture generator

1. Add a generator under `tests/fixtures/synth/` that draws the pattern: a comb row, a checkbox, a stamp over text, a rotated page, a faded fax-like scan, mixed scripts.
2. Make it deterministic from a seed. It returns the image plus the ground truth (regions and text).
3. Use only public fonts and datasets (record their licences in `tests/fixtures/synth/SOURCES.md`).

**Done when:** the generator has a unit test and is used by at least one component test.

## S9. Prepare a PR for human review

1. Format, lint, type-check and run all non-model tests. Run model tests if available.
2. Run the evaluation if the output could change.
3. Write the PR description: task id, what changed, why, test results, the evaluation table (before → after), risks, and what the human should check.
4. Confirm that there is no real document data, no document text and no secrets in the diff (`git diff --stat`, plus a scan for `.pdf`, `.png`, `.xlsx`).

**Done when:** a reviewer can judge the change from the PR alone.
