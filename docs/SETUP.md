# Setup and model requirements

This is the only place where hardware matters. The pipeline itself runs on any machine. Each machine simply enables the models it can run. Run `torus doctor` (available from Phase 0) to see what your machine can run.

## 1. Base setup (every developer)

```bash
git clone https://github.com/siddhiii01/torus.git
cd torus
git checkout dev-sid
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"            # from Phase 0; until then: pip install -r requirements.txt
pre-commit install
```

Environment variables (put them in `.env`, which is git-ignored):

| Variable | Purpose |
|---|---|
| `TORUS_DATA_DIR` | Real documents, test sets, corrections and training data. **Outside the repo.** Encrypted disk recommended. |
| `TORUS_MODEL_DIR` | Model weights cache. Outside the repo, on a disk with space (see sizes below). |
| `HF_HOME`, `PADDLE_HOME`, `DOCTR_CACHE_DIR` | Point these inside `TORUS_MODEL_DIR`. |
| `TORUS_LLM_URL` | Local model server for VLMs/LLMs (vLLM or Ollama), e.g. `http://localhost:11434`. Must be local or internal. |

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

### Document VLMs (OCR-specialised)

| Model | Size | Min GPU memory | Disk | Licence |
|---|---|---|---|---|
| PaddleOCR-VL-1.6 | ~0.9B | ~4 GB (needs flash-attention, otherwise much more) | ~2 GB | Apache 2.0 |
| GLM-OCR | ~0.9B | ~2–4 GB | ~2 GB | check model card |
| MinerU2.5-Pro | ~1.2B | ~4 GB | ~3 GB | check model card |
| dots.ocr | ~3B | ~6 GB | ~6 GB | check model card |
| olmOCR-2 (7B) | 7B | ~16 GB (8–10 GB at 4-bit) | ~15 GB | Apache 2.0 |
| DeepSeek-OCR-2 | ~3B | ~8 GB | ~7 GB | check model card |
| Chandra OCR 2 | ~5B | ~12 GB | ~11 GB | modified OpenRAIL-M (revenue limits, needs review) |

### General VLMs (field-crop handwriting reading)

| Model | Min GPU memory | Disk |
|---|---|---|
| Qwen3-VL-2B | ~3 GB (4-bit ~2 GB) | ~5 GB |
| Qwen3-VL-4B | ~4 GB at 4-bit, ~10 GB at full precision | ~9 GB |
| Qwen3-VL-8B | ~6–8 GB at 4-bit, ~18 GB at full precision | ~17 GB |
| Qwen3-VL-32B / 30B-A3B | ~20–24 GB at 4-bit | ~65 GB |
| Qwen3-VL-235B-A22B | multi-GPU server (4–8 × 80 GB) | ~470 GB |

### Field-mapping LLMs

| Model | Min GPU memory | Disk |
|---|---|---|
| Qwen3-4B | ~4 GB at 4-bit (CPU possible, slow) | ~3 GB |
| Qwen3-8B | ~6 GB at 4-bit | ~5 GB |
| Qwen3-14B | ~10 GB at 4-bit | ~9 GB |
| Qwen3-32B | ~20 GB at 4-bit | ~20 GB |

## 4. Choosing what to enable

1. Run `torus doctor`. It lists every reader in `config/readers.yaml` as `ready`, `can run (weights missing)` or `not supported here`.
2. Download weights only for readers your machine supports (`torus models pull <name>`, from Phase 5).
3. Set `enabled: true` for them in your **local** override file `config/local.yaml` (git-ignored). The shared `config/readers.yaml` defines defaults for the team.
4. Fewer readers is fine for development. Confidence will be lower and more fields go to review, but nothing breaks. Official evaluation numbers are produced on the team's evaluation machine with the full default set.

## 5. Where the weights go

Never inside the repo. Use `TORUS_MODEL_DIR` on a disk with enough space (the full reader set needs roughly 100–150 GB, excluding the 235B model). On a laptop with a small system drive, point it to a secondary drive.
