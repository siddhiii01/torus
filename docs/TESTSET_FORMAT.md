# Evaluation test-set format (P1.1)

Real labelled packs live **outside the repo** under `TORUS_DATA_DIR`. Synthetic
cases used in CI are built at test time by generators under
`tests/fixtures/synth/` — never commit customer PDFs, page images, or answer keys.

## Layout

```text
$TORUS_DATA_DIR/
  testsets/
    <set>/                    # e.g. deb_dev, smoke
      <case>/                 # folder name == meta.case_id
        source.pdf            # input pack (required for real sets)
        expected.json         # gold column → value rows
        meta.json             # split + tags (metadata only)
        pages.json            # optional gold page types (P3.5)
        locations.json        # optional gold value-region boxes (P4.8)
```

Path helpers: `torus.eval.resolve_data_dir`, `get_testsets_root`, `set_dir`,
`case_dir`. Validate with `torus.eval.validate_case_dir` / `validate_testset`.

## `meta.json`

| Field | Required | Description |
|---|---|---|
| `case_id` | yes | Must equal the case folder name |
| `split` | yes | `dev` or `hidden` |
| `template_id` | yes | Template config id (e.g. `starter_v1`, `deb_v1`) |
| `tags` | no | Object; see below |
| `page_count_expected` | no | Non-negative int |
| `notes` | no | Operator notes about quality/split — **not** extracted field values |

### `tags`

| Field | Values |
|---|---|
| `year` | Integer 1900–2100 |
| `scan_type` | `digital` \| `scan` \| `photo` \| `fax` \| `mixed` |
| `handwriting_level` | `none` \| `light` \| `medium` \| `heavy` |

Extra keys under `tags` are rejected (`extra = forbid`). Add new tag fields via
schema change when needed — do not put free-form document text here.

Example:

```json
{
  "case_id": "pack_001",
  "split": "dev",
  "template_id": "starter_v1",
  "tags": {
    "year": 2024,
    "scan_type": "scan",
    "handwriting_level": "light"
  },
  "page_count_expected": 12,
  "notes": ""
}
```

## `expected.json`

| Field | Required | Description |
|---|---|---|
| `template_id` | yes | Must match `meta.template_id` |
| `case_id` | no | When set, must match the folder / `meta.case_id` |
| `version` | no | Schema version for this file (default `1`) |
| `rows` | yes | List of objects: template column key → scalar value |

Row values are JSON scalars only: `string`, `number`, `boolean`, or `null`.
Keys starting with `_` are ignored (reserved for harness bookkeeping). Nested
objects/arrays are invalid.

When validating with `template_keys=...` (from the template config), every
column key in any row must be a known field key. Unknown keys fail the case.

Omitted keys mean “no gold for this column on this row” (scoring in P1.2).
Use `null` only when you need an explicit empty gold cell.

Example (synthetic / starter template):

```json
{
  "template_id": "starter_v1",
  "case_id": "pack_001",
  "version": 1,
  "rows": [
    {
      "person_name": "Synthetic Person A",
      "mobile_no": "9876543210",
      "address_pincode": "560001",
      "notes": null
    }
  ]
}
```

## `pages.json` (optional, P3.5)

Gold page-type labels for classification evaluation. Type ids only — never OCR
text or crops. When present, `python -m torus.eval run` and
`python -m torus.eval classify` score predictions against it.

| Field | Required | Description |
|---|---|---|
| `case_id` | no | When set, must match the folder / `meta.case_id` |
| `version` | no | Schema version (default `1`) |
| `pages` | yes | List of `{index, page_type}` (optional `confidence` on preds) |

`index` is the 0-based page index. `page_type` must be a catalogue id from
`config/page_types.yaml` for meaningful metrics (unknown ids are still scored).

Example:

```json
{
  "case_id": "pack_001",
  "version": 1,
  "pages": [
    {"index": 0, "page_type": "application_form"},
    {"index": 1, "page_type": "id_card"},
    {"index": 2, "page_type": "blank"}
  ]
}
```

Predicted page types for scoring live under the prediction directory (not in
the case folder), searched in this order:

1. `{case_id}_{variant}.pages.json` (when `--pred-variant` is set)
2. `{case_id}.pages.json`
3. `{case_id}/pages.json`
4. `{case_id}_pages.json`

Classification reports include overall accuracy, per-type precision/recall/F1,
a confusion matrix (rows = gold, columns = predicted), and misclassifications
listed by gold type. No document text appears in the report.

## `locations.json` (optional, P4.8)

Gold value-region boxes for **location evaluation** (did layout find the right
region for each column *before* reading?). Geometry and column keys only —
never OCR text, crops, or field values. When present,
`python -m torus.eval run` and `python -m torus.eval locate` score predictions
against it using IoU ≥ `pipeline.layout.locate_iou` (default `0.3`).

| Field | Required | Description |
|---|---|---|
| `case_id` | no | When set, must match the folder / `meta.case_id` |
| `version` | no | Schema version (default `1`) |
| `fields` | yes | List of field locations (see below) |

Each entry in `fields`:

| Field | Required | Description |
|---|---|---|
| `column_key` | yes | Template field key (e.g. `mobile_no`) |
| `page_index` | yes | 0-based page index |
| `bbox` | yes | Normalised box `{x0,y0,x1,y1}` in `[0,1]`, top-left origin |
| `row_index` | no | Index into `expected.json` rows (default `0`) |
| `kind` | no | Optional `RegionKind` string; matching does not require kind equality |

Example:

```json
{
  "case_id": "pack_001",
  "version": 1,
  "fields": [
    {
      "column_key": "mobile_no",
      "row_index": 0,
      "page_index": 0,
      "bbox": {"x0": 0.42, "y0": 0.18, "x1": 0.68, "y1": 0.22},
      "kind": "field_value"
    },
    {
      "column_key": "address_pincode",
      "page_index": 0,
      "bbox": {"x0": 0.42, "y0": 0.31, "x1": 0.55, "y1": 0.35}
    }
  ]
}
```

Predicted locations for scoring live under the prediction directory (not in
the case folder), searched in this order:

1. `{case_id}_{variant}.locations.json` (when `--pred-variant` is set)
2. `{case_id}.locations.json`
3. `{case_id}/locations.json`
4. `{case_id}_locations.json`

Location reports include overall **located rate** (`located / expected`),
precision, mean IoU of matched pairs, and a per-column table. Phase 4
acceptance targets ≥97% located rate on the dev set. No document text
appears in the report.

## Reader bake-off (P5.8)

```bash
python -m torus.eval readers --set deb_dev --report reports/eval/readers.md \
  [--reader NAME ...] [--min-samples 10]
```

Each reader reads the **gold** boxes in `locations.json` (so layout errors are
excluded) and is scored against `expected.json` with the field's own match
rule. Cases or fields without a gold box or gold value are skipped. Readers
that cannot run here (hardware, weights, extra, server) are listed as skipped,
and `enabled: false` readers are still measured. The report has accuracy,
errors, empty outputs and median/p95 latency per reader x field type, plus a
*proposed* default set (best reader and best reader of another family per field
type). It never edits `config/readers.yaml`; a human approves the defaults. No
document text appears in it.

## `source.pdf`

Required and non-empty for production sets (`require_source=True`, the default).
Unit tests may pass `require_source=False` when only JSON shape is under test.
Generators that build a synthetic mini-set write a tiny real PDF so full
validation can run in CI without customer data.

## Validation rules (summary)

1. Case folder contains `meta.json`, `expected.json`, and (by default) `source.pdf`.
2. Both JSON files parse as objects and satisfy the pydantic schemas in
   `torus.eval.schema`. Optional `pages.json` uses `PageLabelsDocument`;
   optional `locations.json` uses `LocationsDocument`.
3. `meta.case_id` equals the folder name; optional `expected.case_id` /
   `pages.case_id` / `locations.case_id` match too.
4. `meta.template_id` equals `expected.template_id`.
5. Optional: all expected column keys ⊆ template field keys.
6. No document body text in logs or error messages — paths, keys, and schema
   messages only.

## Labelling (P1.5)

Grow the test set locally — the PDF never leaves the machine.

**CLI** (opens the PDF in the system viewer, then prompts for columns):

```bash
# TORUS_DATA_DIR must be set (see docs/SETUP.md)
torus label \
  --pdf /path/to/pack.pdf \
  --set deb_dev \
  --case pack_001 \
  --template starter_v1 \
  --split dev \
  --year 2024 \
  --scan-type scan \
  --handwriting light

# Non-interactive (local rows JSON only):
torus label --pdf pack.pdf --set deb_dev --case pack_001 \
  --template starter_v1 --rows-json rows.json --no-open
```

Per cell: type a value, Enter to omit the column, or `-` for an explicit null.
Writes `$TORUS_DATA_DIR/testsets/<set>/<case>/{source.pdf,meta.json,expected.json}`
and validates with the P1.1 schemas. Use `--force` to overwrite.

**Optional UI** (requires `pip install -e ".[review]"`):

```bash
streamlit run src/torus/review/label_app.py
```

Core API: `torus.eval.label.write_labelled_case` / `LabelCaseRequest`.

## Legacy bake-off cases

Older DEB cases under `$TORUS_DATA_DIR/legacy/deb/cases/<slug>/` use `gold.json`
instead of `expected.json`. Keep them for historical scoring; new labelling and
Phase 1+ eval use this test-set layout. A converter can map `gold.json` →
`expected.json` when promoting a pack (P9.4).

## Synthetic mini-set (tests)

```python
from tests.fixtures.synth.testset import build_mini_testset

root = build_mini_testset(tmp_path / "data")
# → <root>/testsets/mini/case_dev_001/
# → <root>/testsets/mini/case_hidden_001/
```

Uses template id `starter_v1` and placeholder names only (no real customers).
