# Setup and model requirements

This is the only place where hardware matters. The pipeline itself runs on any machine. Each machine simply enables the models it can run. Run `torus doctor` (available from Phase 0) to see what your machine can run.

## 1. Base setup (every developer)

```bash
git clone https://github.com/siddhiii01/torus.git
cd torus
git checkout dev-sid
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Until optional model extras are needed, the editable install above is enough. The legacy `requirements.txt` remains for the pre-`src/torus` scripts only.

Local checks (same as CI):

```bash
ruff format src tests && ruff check src tests
mypy src
pytest -m "not model"
```

GitHub Actions runs those three steps on every push and pull request (Python 3.12). Tests marked `@pytest.mark.model` are skipped in that job.

Environment variables (put them in `.env`, which is git-ignored):

| Variable | Purpose |
|---|---|
| `TORUS_DATA_DIR` | Real documents, test sets, corrections and training data. **Outside the repo.** Encrypted disk recommended. |
| `TORUS_MODEL_DIR` | Model weights cache. Outside the repo, on a disk with space (see sizes below). |
| `HF_HOME`, `PADDLE_HOME`, `DOCTR_CACHE_DIR` | Point these inside `TORUS_MODEL_DIR`. |
| `TORUS_LLM_URL` | Local model server for VLMs/LLMs (vLLM or Ollama), e.g. `http://localhost:11434`. Must be local or internal. |

### `TORUS_DATA_DIR` layout

Create the folder **outside** the git checkout (example: `%USERPROFILE%\torus-data` on Windows, `~/torus-data` on Linux/macOS), then:

```bash
# Windows PowerShell
setx TORUS_DATA_DIR "$env:USERPROFILE\torus-data"
# Linux / macOS
export TORUS_DATA_DIR="$HOME/torus-data"
```

Expected tree (create empty dirs as needed; do not put any of this in git):

```text
$TORUS_DATA_DIR/
  README.md                 # optional local note
  testsets/<set>/<case>/    # labelled eval cases (Phase 1+)
    source.pdf
    expected.json           # column → value rows (no other doc text dumps)
    meta.json               # split (dev|hidden), tags — metadata only
  corrections/              # review decisions (Phase 9+)
  training/                 # crops + labels for fine-tune jobs (Phase 10+)
  legacy/deb/               # bake-off cases moved out of the repo in P0.8
    cases/<slug>/source.pdf
    cases/<slug>/gold.json
    cases/<slug>/meta.json
    raw/                    # client spreadsheets, never committed
```

Full case schema, validation rules and synthetic mini-set notes:
[`docs/TESTSET_FORMAT.md`](TESTSET_FORMAT.md). Validate with
`torus.eval.validate_case_dir` / `validate_testset`.

Label new packs locally with `torus label` (or the optional Streamlit UI under
the `review` extra). See the Labelling section in `docs/TESTSET_FORMAT.md`.

Rules:

- The repo `data/` directory is a stub (`data/README.md` only). Pre-commit blocks PDFs, images, spreadsheets and JSON under `data/`, `samples/`, and `**/fixtures/real/`.
- Never commit real documents, page images, OCR text, crops or answer keys.
- **Purging past customer files from git history needs explicit owner approval.** Working-tree removal and `git rm --cached` are enough for day-to-day safety; history rewrite is a separate, approved step.

Tesseract is a system install (Windows: UB Mannheim build. Linux: `apt install tesseract-ocr`). Add `tesseract-ocr-tam` and other language packs as needed.

## 2. Optional model stacks

Install only what your machine can run:

```bash
pip install -e ".[paddle]"     # PaddleOCR, PaddleOCR-VL
pip install -e ".[doctr]"      # docTR
pip install -e ".[vlm]"        # transformers / vLLM client for VLM readers
pip install -e ".[train]"      # fine-tuning (LoRA, datasets)
pip install -e ".[review]"     # review UI
```

For GPUs: install the CUDA build of torch (and paddle) that matches your driver **before** the extras.

## 3. Model requirements

The figures are approximate and come from public guides. Confirm them with `torus doctor` and the reader smoke tests, then correct this table. "4-bit" means quantised weights. Licences must be checked before production use.

### Classic OCR

| Model | Use | Min GPU memory | CPU-only | Disk | Licence |
|---|---|---|---|---|---|
| Tesseract 5 | printed text, second opinion | none | yes | <100 MB | Apache 2.0 |
| docTR (fast_base + crnn / parseq) | printed text and layout | none (GPU optional) | yes, slower | ~150 MB | Apache 2.0 |
| PaddleOCR PP-OCRv5 | printed + handwriting, many scripts incl. Tamil | none (GPU optional) | yes | ~200 MB | Apache 2.0 |

### Small handwriting and character readers

| Model | Use | Min GPU memory | CPU-only | Disk | Licence |
|---|---|---|---|---|---|
| TrOCR base (handwritten/printed) | handwritten line and field crops | 2 GB | yes, slow | ~1.3 GB | MIT |
| PARSeq | short text crops | 1 GB | yes | ~100 MB | Apache 2.0 |
| Character-cell model (ours) | one character per comb box | none | yes | <50 MB | ours |

Handwriting line readers (P5.6, family `handwriting`, `enabled: false` until bake-off):

- `trocr_handwritten`: `pip install -e ".[vlm]"`. Weights load offline (`local_files_only`). Put `microsoft/trocr-base-handwritten` in the HF cache under `HF_HOME`, or set `model:` in `config/local.yaml` to a folder under `TORUS_MODEL_DIR`. Decoding is constrained by the field's charset and length.
- `parseq_handwritten`: `pip install -e ".[doctr]"`. docTR's `parseq` recogniser, cached under `DOCTR_CACHE_DIR`. docTR cannot constrain decoding, so the charset hint is not applied.

Reader name: `char_cell` (family `char_cell`, `enabled: false` until bake-off).
Inference is NumPy-only (no torch). Weights path in config: `char_cell/v1.npz`
under `TORUS_MODEL_DIR`. Train first on synthetic glyphs, then on public EMNIST:

```bash
# Synthetic only (no download; good for smoke / CI-like machines)
python -m torus.readers.char_cell train --source synth --out char_cell/v1.npz

# Public EMNIST (needs: pip install -e ".[train]")
python -m torus.readers.char_cell train --source emnist --out char_cell/v1.npz
# or both:
python -m torus.readers.char_cell train --source both --out char_cell/v1.npz
```

Relative `--out` resolves under `TORUS_MODEL_DIR`. EMNIST cache goes under
`TORUS_MODEL_DIR/cache/emnist` (or `TORUS_DATA_DIR` / `HF_HOME`), never into git.
Later phases fine-tune on review corrections (P10); do not point training at
customer documents from this command.

### Document VLMs (OCR-specialised)

Served through the local model server in `TORUS_LLM_URL` (Ollama `/api/chat` or
OpenAI-compatible `/v1/chat/completions`; override path with `TORUS_DOC_VLM_PATH`).
Readers are registered in `config/readers.yaml` under family `document_vlm`,
`enabled: false` until bake-off. No pip extra is required for the HTTP path
(stdlib client). Set `enabled: true` in `config/local.yaml` and point
`TORUS_LLM_URL` at a server that has the matching model tag loaded.

| Model | Reader name | Size | Min GPU memory | Disk | Licence |
|---|---|---|---|---|---|
| PaddleOCR-VL-1.6 | `paddleocr_vl` | ~0.9B | ~4 GB (needs flash-attention, otherwise much more) | ~2 GB | Apache 2.0 |
| GLM-OCR | `glm_ocr` | ~0.9B | ~2–4 GB | ~2 GB | check model card |
| MinerU2.5-Pro | `mineru` | ~1.2B | ~4 GB | ~3 GB | check model card |
| dots.ocr | `dots_ocr` | ~3B | ~6 GB | ~6 GB | check model card |
| olmOCR-2 (7B) | `olmocr` | 7B | ~16 GB (8–10 GB at 4-bit) | ~15 GB | Apache 2.0 |
| DeepSeek-OCR-2 | `deepseek_ocr` | ~3B | ~8 GB | ~7 GB | check model card |
| Chandra OCR 2 | `chandra` | ~5B | ~12 GB | ~11 GB | modified OpenRAIL-M (revenue limits, needs review) |

### General VLMs (field-crop handwriting reading)

| Model | Min GPU memory | Disk |
|---|---|---|
| Qwen3-VL-2B | ~3 GB (4-bit ~2 GB) | ~5 GB |
| Qwen3-VL-4B | ~4 GB at 4-bit, ~10 GB at full precision | ~9 GB |
| Qwen3-VL-8B | ~6–8 GB at 4-bit, ~18 GB at full precision | ~17 GB |
| Qwen3-VL-32B / 30B-A3B | ~20–24 GB at 4-bit | ~65 GB |
| Qwen3-VL-235B-A22B | multi-GPU server (4–8 × 80 GB) | ~470 GB |

Readers `qwen3_vl_2b`, `qwen3_vl_4b` and `qwen3_vl_8b` (family `general_vlm`, `enabled: false`
until bake-off) read field crops through the same local server (`TORUS_LLM_URL`, OpenAI-compatible
`/v1/chat/completions`, so vLLM or Ollama's `/v1`). Default server tags are `qwen3-vl:<size>`; load
that tag, or the server's own name for the model, in the server. The prompt is built from the
`ReadHint`. `raw_score` is the model's sequence probability (`exp` of the summed token log-probs)
when the server returns `logprobs`; if the server rejects the flag the reader retries without it and
`raw_score` is `None`. No pip extra is needed.

### Field-mapping LLMs

| Model | Min GPU memory | Disk |
|---|---|---|
| Qwen3-4B | ~4 GB at 4-bit (CPU possible, slow) | ~3 GB |
| Qwen3-8B | ~6 GB at 4-bit | ~5 GB |
| Qwen3-14B | ~10 GB at 4-bit | ~9 GB |
| Qwen3-32B | ~20 GB at 4-bit | ~20 GB |

The mapper (P8.3) is configured in `config/pipeline.yaml` under `mapping` and is off by default. To
enable it: load a Qwen3 tag on the local server, set `mapping.model` to that tag (or in
`config/local.yaml`), set `mapping.enabled: true`, and point `TORUS_LLM_URL` at the server. Only
local/internal hosts are accepted; temperature is fixed at 0. No pip extra is needed.

## 4. Page classifier (Phase 3)

Configured in `config/pipeline.yaml` under `classify`:

| Strategy | What it needs | Notes |
|---|---|---|
| `trained` | Optional weights JSON under `TORUS_MODEL_DIR` | Image + text-cue features; NumPy only. Without weights, a cue/visual prior runs. Train with `python -m torus.classify train --manifest … --out page_clf.json` (images under `TORUS_DATA_DIR`). |
| `vlm` | Local server in `TORUS_LLM_URL` + `classify.vlm.model` | Zero-shot prompt from `config/page_types.yaml`. Uses the existing `vlm` extra only if you later swap in an in-process transformers client; the default client is stdlib HTTP to localhost/internal. |
| `vote` (default) | Either or both of the above | Agreement boosts confidence; disagreement applies a penalty. Missing models are skipped, never a crash. |

No new pip extras are required for the default classify path.

## 5. Choosing what to enable

1. Run `torus doctor`. It lists every reader in `config/readers.yaml` as `ready`, `can run (weights missing)` or `not supported here`.
2. Download weights only for readers your machine supports (`torus models pull <name>`, from Phase 5).
3. Set `enabled: true` for them in your **local** override file `config/local.yaml` (git-ignored). The shared `config/readers.yaml` defines defaults for the team.
4. Fewer readers is fine for development. Confidence will be lower and more fields go to review, but nothing breaks. Official evaluation numbers are produced on the team's evaluation machine with the full default set.

## 6. Where the weights go

Never inside the repo. Use `TORUS_MODEL_DIR` on a disk with enough space (the full reader set needs roughly 100–150 GB, excluding the 235B model). On a laptop with a small system drive, point it to a secondary drive.

## 7. Throughput per hardware profile (service load test)

Measured with `python -m torus.service loadtest` (synthetic packs, real worker
processes, shared queue). Add a row per machine you test; the report written to
`reports/load/latest.md` has the full detail. Numbers apply only to the reader set
listed.

| Profile | Readers enabled | Workers | Pack sizes (pages) | Packs/h | Pages/h | Service p50 / p95 |
|---|---|---|---|---|---|---|
| i5-1235U (12 threads), 15.7 GB RAM, no GPU, Windows 11 | none (pipeline overhead only) | 2 | 4, 8 (6 packs) | 61 | 367 | 99 s / 153 s |

Read this carefully: with no reader enabled this is the cost of ingest,
preprocessing, layout and classification alone, on a laptop CPU. Throughput with
the real reader set (and a GPU) must be measured on the target hardware before
capacity planning; do not extrapolate from this row. The `deb_v1` template was
used. Run to completion (6 of 6 packs succeeded, none timed out).
