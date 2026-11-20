# Testing Torus

Commands below were checked against each tool's `--help`. Run from the repo root. Set
`TORUS_DATA_DIR` first (see `docs/SETUP.md`).

## 1. Automated tests (no models, no data)

```
python -m pytest -m "not model" -q
python -m ruff check .
python -m mypy src
```

The suite uses synthetic documents only. Tests marked `model` need installed weights and are skipped
otherwise (`python -m pytest -m model`).

## 2. What can run on this machine

```
python -m torus doctor
```

Lists extras, hardware and which readers can run. All readers are disabled by default. Enable one per
machine in `config/local.yaml` (git-ignored):

```yaml
readers:
  tesseract_printed: true
```

## 3. Run the pipeline on a file or folder

```
python -m torus run <pdf-or-folder> --template deb_v1 --out out.xlsx
python -m torus run <pdf> --template deb_v1 --template-xlsx "<client.xlsx>" --out out.xlsx
```

The second form fills a copy of the client's workbook and keeps its formatting. The workbook gets a
second sheet `torus_evidence` with status, confidence and evidence per cell. Progress prints to stderr.
Without a calibrator (`--calibrator` or `TORUS_MODEL_DIR/calibration/calibrator.json`) nothing is
`confirmed`.

Time one page first. Extract it into a temporary folder, not into the repo.

## 4. Template import

```
python -m torus.output.template sync "<client.xlsx>" <template_id> --sheet "<sheet>" --header-row 2 --dry-run
```

Drop `--dry-run` to write `config/templates/<template_id>.yaml`. Only headers are read.

## 5. Labelling and review

```
pip install -e ".[review]"
python -m torus label --pdf <pdf> --set <set> --case <case> --template deb_v1 --split dev
python -m torus review <bundle.json> [--port N] [--reviewer NAME]
python -m torus reexport <bundle.json> --template deb_v1 --out out.xlsx
python -m torus promote-review <bundle.json> ...
```

Note: `torus run` does **not** yet write a review bundle (see `docs/HANDOFF.md`), so `review`,
`reexport` and `promote-review` cannot be fed from a real run today.

## 6. Evaluation

```
python -m torus.eval run --set <set> --pred-dir <dir> --report reports/eval/<date>.md [--split dev|hidden]
python -m torus.eval diff <baseline.json> <candidate.json>
python -m torus.eval classify --set <set> --pred-dir <dir> --report <file>
python -m torus.eval locate   --set <set> --pred-dir <dir> --report <file>
python -m torus.eval readers  --set <set> --report <file> [--reader NAME]
```

`eval readers` is the reader bake-off (needs readers installed). `eval run` scores predictions you
already produced; there is no one-command "run and score" yet.

## 7. Service

```
python -m torus.service serve | worker | loadtest | purge | verify-audit | bundle
```

`python -m torus.service loadtest` writes `reports/load/latest.md`. Docker files are in `deploy/` and
have not been built.

## 8. Learning loop

```
python -m torus.learn export | register | registry | promote
```

Needs reviewed corrections (none exist yet) and the `train` extra.
