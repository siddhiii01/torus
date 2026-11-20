"""Dataset builder (P10.1): review corrections → training sets per reader type.

Crops are re-rendered from each bundle's source file (the store keeps boxes, not
pixels). Output holds document content, so it lives under
``TORUS_DATA_DIR/training/`` and never in git. Reports and errors carry counts
and ids only, never label text.

Rules that keep labels trustworthy:

* only human ``accept`` / ``correct`` decisions with a non-empty value count;
* a field with several evidence regions is skipped (label is not tied to one crop);
* the split is by *document* (hash of document id + seed), so no document spans
  two splits;
* a char-cell sample set needs one detected cell per label character, else the
  field is skipped (no guessing which cell holds which character).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from torus.core.errors import TorusError
from torus.core.models import BBox
from torus.review.bundle import load_bundle
from torus.review.queue import ReviewError
from torus.review.store import CorrectionRecord, CorrectionStore

TRAINING_SUBDIR = "training"
MANIFEST_NAME = "manifest.json"


class LearnError(TorusError):
    """Bad dataset request or unusable training input."""


class SampleKind(StrEnum):
    CHAR_CELL = "char_cell"  # one glyph image + one character label
    LINE = "line"  # one crop + its text (TrOCR / PARSeq)
    FIELD_VLM = "field_vlm"  # one crop + target value (VLM LoRA)


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    document_id: str
    split: Split
    column_key: str
    page_index: int = Field(ge=0)
    region_id: str
    cell_index: int | None = None
    image: str  # path relative to the dataset folder
    label: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    kind: SampleKind
    seed: int
    val_fraction: float
    test_fraction: float
    since: datetime | None
    created_at: datetime
    data_hash: str
    counts: dict[str, int]
    skipped: dict[str, int]
    samples: list[Sample]


# Page image in, normalised box in, PIL crop out; lets tests avoid real files.
CropFn = Callable[[Any, BBox], Any]
# (page image, box) → one image per detected character cell, left to right.
CellSplitter = Callable[[Any, BBox], Sequence[Any]]
# document id → (source path, per-page image getter)
PageProvider = Callable[[str], Callable[[int], Any]]


def assign_split(
    document_id: str,
    *,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Split:
    """Deterministic document-level split from a hash (stable across runs)."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise LearnError("val/test fractions must be >= 0 and sum to < 1")
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < test_fraction:
        return Split.TEST
    if u < test_fraction + val_fraction:
        return Split.VAL
    return Split.TRAIN


def effective_label(rec: CorrectionRecord, *, include_accepted: bool) -> str | None:
    """Human-verified text for a record, or ``None`` when it is not usable."""
    if rec.action == "correct":
        text = rec.value
    elif rec.action == "accept" and include_accepted:
        text = rec.value if rec.value is not None else rec.suggestion
    else:
        return None
    if text is None or not text.strip():
        return None
    return text


def crop_box(image: Any, box: BBox, *, pad: float = 0.0) -> Any:
    """Pixel crop of a normalised box (optionally padded)."""
    x0 = max(0.0, box.x0 - pad)
    y0 = max(0.0, box.y0 - pad)
    x1 = min(1.0, box.x1 + pad)
    y1 = min(1.0, box.y1 + pad)
    left, top = int(x0 * image.width), int(y0 * image.height)
    right = max(left + 1, round(x1 * image.width))
    bottom = max(top + 1, round(y1 * image.height))
    return image.crop((left, top, min(right, image.width), min(bottom, image.height)))


def default_cell_splitter(image: Any, box: BBox) -> list[Any]:
    """Split a comb crop into cell images using the layout stage (lazy import)."""
    from torus.core.models import Region
    from torus.core.types import RegionKind
    from torus.layout.comb import split_comb_cells

    comb = Region(
        id="comb", page_index=0, bbox=box, kind=RegionKind.COMB, label_anchor=None
    )
    return [crop_box(image, c.bbox) for c in split_comb_cells(image, comb)]


def _hash_samples(samples: Iterable[tuple[Sample, bytes]]) -> str:
    h = hashlib.sha256()
    for sample, png in sorted(samples, key=lambda s: s[0].sample_id):
        h.update(sample.sample_id.encode())
        h.update(b"\0")
        h.update(sample.label.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(png).digest())
    return h.hexdigest()


def _png_bytes(image: Any) -> bytes:
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _sample_id(document_id: str, item_id: str, cell: int | None) -> str:
    raw = f"{document_id}|{item_id}|{cell}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def build_dataset(
    records_by_document: Mapping[str, Mapping[str, CorrectionRecord]],
    pages: PageProvider,
    out_dir: Path,
    *,
    kind: SampleKind,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    since: datetime | None = None,
    include_accepted: bool = True,
    exclude_documents: Iterable[str] = (),
    cell_splitter: CellSplitter = default_cell_splitter,
    crop: CropFn = crop_box,
) -> Manifest:
    """Write crops + ``manifest.json`` under ``out_dir`` and return the manifest.

    ``records_by_document`` maps document id → latest record per item id (what
    ``CorrectionStore.latest`` returns). ``exclude_documents`` keeps documents
    (for example the hidden evaluation set) out of every split.
    """
    excluded = set(exclude_documents)
    skipped: Counter[str] = Counter()
    kept: list[tuple[Sample, bytes]] = []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for document_id in sorted(records_by_document):
        if document_id in excluded:
            skipped["excluded_document"] += 1
            continue
        split = assign_split(
            document_id,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        get_page: Callable[[int], Any] | None = None
        for item_id, rec in sorted(records_by_document[document_id].items()):
            if since is not None and rec.decided_at < since:
                skipped["before_since"] += 1
                continue
            label = effective_label(rec, include_accepted=include_accepted)
            if label is None:
                skipped["no_usable_label"] += 1
                continue
            if len(rec.crop_refs) != 1:
                skipped["not_single_evidence"] += 1
                continue
            ref = rec.crop_refs[0]
            if get_page is None:
                try:
                    get_page = pages(document_id)
                except (LearnError, ReviewError):
                    skipped["document_unavailable"] += 1
                    continue
            try:
                page_image = get_page(ref.page_index)
            except (IndexError, LearnError, ReviewError):
                skipped["page_unavailable"] += 1
                continue

            if kind is SampleKind.CHAR_CELL:
                chars = [c for c in label if not c.isspace()]
                cells = list(cell_splitter(page_image, ref.bbox))
                if len(cells) != len(chars):
                    skipped["cell_count_mismatch"] += 1
                    continue
                items: list[tuple[int | None, Any, str]] = [
                    (i, cell, ch)
                    for i, (cell, ch) in enumerate(zip(cells, chars, strict=True))
                ]
            else:
                items = [(None, crop(page_image, ref.bbox), label)]

            for cell_index, image, text in items:
                sid = _sample_id(document_id, item_id, cell_index)
                rel = f"images/{sid}.png"
                png = _png_bytes(image)
                (out_dir / rel).write_bytes(png)
                kept.append(
                    (
                        Sample(
                            sample_id=sid,
                            document_id=document_id,
                            split=split,
                            column_key=rec.column_key,
                            page_index=ref.page_index,
                            region_id=ref.region_id,
                            cell_index=cell_index,
                            image=rel,
                            label=text,
                        ),
                        png,
                    )
                )

    samples = sorted((s for s, _ in kept), key=lambda s: s.sample_id)
    manifest = Manifest(
        kind=kind,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        since=since,
        created_at=datetime.now(UTC),
        data_hash=_hash_samples(kept),
        counts={sp.value: sum(1 for s in samples if s.split is sp) for sp in Split},
        skipped=dict(skipped),
        samples=samples,
    )
    (out_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), "utf-8")
    _write_split_files(out_dir, kind, samples)
    return manifest


def _write_split_files(out_dir: Path, kind: SampleKind, samples: list[Sample]) -> None:
    """One JSONL per split in the layout each trainer reads."""
    for split in Split:
        lines: list[str] = []
        for s in samples:
            if s.split is not split:
                continue
            if kind is SampleKind.FIELD_VLM:
                row: dict[str, Any] = {
                    "image": s.image,
                    "field": s.column_key,
                    "target": s.label,
                }
            else:
                row = {"image": s.image, "text": s.label}
            lines.append(json.dumps(row, ensure_ascii=False))
        (out_dir / f"{split.value}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def load_records(
    store: CorrectionStore, document_ids: Iterable[str]
) -> dict[str, dict[str, CorrectionRecord]]:
    return {d: store.latest(d) for d in document_ids}


def bundle_sources(bundles_dir: Path) -> dict[str, Path]:
    """Map document id → source path from every ``*.json`` bundle in a folder."""
    out: dict[str, Path] = {}
    for path in sorted(bundles_dir.glob("*.json")):
        bundle = load_bundle(path)
        out[bundle.result.document_id] = Path(bundle.source_path)
    return out


def ingest_pages(sources: Mapping[str, Path]) -> PageProvider:
    """Page provider that re-renders a document on first use (lazy ingest)."""

    def provider(document_id: str) -> Callable[[int], Any]:
        source = sources.get(document_id)
        if source is None:
            raise LearnError("no bundle found for document")
        from torus.ingest import ingest

        document = ingest(source).document

        def get(index: int) -> Any:
            if not 0 <= index < len(document.pages):
                raise IndexError("page out of range")
            return document.pages[index].image

        return get

    return provider
