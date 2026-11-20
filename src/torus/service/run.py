"""End-to-end run (P8.6): files → pages → fields → rows → Excel.

This is glue. Each stage is called through its public function; no stage's
internals are used, and the heavy parts (readers, LLM) are injected so the run
degrades instead of crashing: no reader means every column is ``not_found``, no
calibrator means nothing can be ``confirmed``, no LLM means judgement columns go
to review.

Logs carry metadata only (hashes, counts, status codes), never document text.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from torus.classify.apply import classify_document
from torus.classify.factory import build_page_classifier
from torus.classify.persons import assign_document_persons
from torus.config.schemas import FieldSpec, ReaderSpec, TemplateConfig, TorusConfig
from torus.consensus.agreement import check_agreement
from torus.consensus.calibrate import Calibrator, extract_features, reader_fingerprint
from torus.consensus.status import assign_status
from torus.consensus.vote import vote
from torus.core.errors import IngestError
from torus.core.models import (
    Document,
    ExtractionResult,
    FieldValue,
    Page,
    ReadHint,
    Reading,
    Region,
    ValidationContext,
)
from torus.core.protocols import PageClassifier
from torus.ingest.formats import SUPPORTED_SUFFIXES
from torus.ingest.loader import ingest
from torus.layout.anchor import anchor_fields
from torus.layout.detect import detect_layout
from torus.logging import get_logger
from torus.mapping.llm_client import JsonLlmClient, build_default_llm_client
from torus.mapping.llm_mapper import LlmFieldMapper
from torus.mapping.rows import FieldObservation, PageOwner, plan_rows
from torus.preprocess import (
    apply_blank_detect,
    apply_orientation,
    apply_page_quality,
    apply_quality_repair,
    correct_geometry,
    detect_orientation,
)
from torus.readers.runner import ReaderRunner
from torus.validate.registry import run_validators

log = get_logger(__name__)

UNCALIBRATED = "uncalibrated"
CALIBRATOR_STALE = "calibrator_stale"


class RegionReaders(Protocol):
    """What the run needs from the reader framework (``ReaderRunner`` adapter)."""

    def names(self) -> list[str]: ...

    def family(self, name: str) -> str: ...

    def version(self, name: str) -> str: ...

    def prints_text(self, name: str) -> bool: ...

    def read(
        self, name: str, image: object, region: Region, hint: ReadHint
    ) -> Reading: ...


class RunnerReaders:
    """``RegionReaders`` over the configured reader registry (skip, never crash)."""

    def __init__(self, runner: ReaderRunner) -> None:
        self._runner = runner

    def names(self) -> list[str]:
        return self._runner.registry.available_names()

    def _spec(self, name: str) -> ReaderSpec:
        spec = self._runner.registry.config.get(name)
        if spec is None:
            raise KeyError(name)
        return spec

    def family(self, name: str) -> str:
        return self._spec(name).family

    def version(self, name: str) -> str:
        return self._spec(name).version or "0"

    def prints_text(self, name: str) -> bool:
        return self._spec(name).capabilities.printed

    def read(self, name: str, image: object, region: Region, hint: ReadHint) -> Reading:
        return self._runner.read(name, image, region, hint)


@dataclass
class RunContext:
    """Everything a run needs, built once and shared across files."""

    config: TorusConfig
    template: TemplateConfig
    readers: RegionReaders
    classifier: PageClassifier | None = None
    llm_client: JsonLlmClient | None = None
    calibrator: Calibrator | None = None


@dataclass(frozen=True)
class RunReport:
    """Counts only — never any cell content."""

    results: list[ExtractionResult]
    files_ok: int
    files_failed: int
    failed_files: list[str]


def collect_inputs(source: Path) -> list[Path]:
    """One file, or every supported file directly inside a folder (sorted)."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise IngestError(f"input not found: {source.name}")


def _embedded_texts(embedded: Sequence[object]) -> dict[int, str]:
    out: dict[int, str] = {}
    for page in embedded:
        words = getattr(page, "words", ())
        if getattr(page, "usable", False) and words:
            out[page.page_index] = " ".join(w.text for w in words)  # type: ignore[attr-defined]
    return out


def _preprocess(document: Document, config: TorusConfig) -> None:
    pre = config.pipeline.preprocess
    for page in document.pages:
        if not page.has_image:
            continue
        apply_orientation(page, detect_orientation(page.image, pre.orientation))
        correct_geometry(page, pre)
        apply_quality_repair(page, pre.quality_repair)
        apply_page_quality(page, pre.page_quality)
        apply_blank_detect(page, pre.blank_detect)


def _fields_for_page(template: TemplateConfig, page: Page) -> list[FieldSpec]:
    return [
        f
        for f in template.scored_fields()
        if f.duplicate_of is None
        and (
            not f.look_in
            or (page.page_type is not None and page.page_type in f.look_in)
        )
    ]


def _hint(spec: FieldSpec) -> ReadHint:
    return ReadHint(
        charset=spec.charset,
        expected_length=spec.length,
        min_length=spec.min_length,
        max_length=spec.max_length,
    )


def _observe_page(
    page: Page,
    ctx: RunContext,
    document_id: str,
    reasons: dict[tuple[str, str], str],
) -> list[FieldObservation]:
    """Locate, read, vote, validate and grade every column present on a page."""
    cfg = ctx.config.pipeline
    fields = _fields_for_page(ctx.template, page)
    names = ctx.readers.names()
    if not fields or not page.has_image or not names:
        return []

    layout = detect_layout(page, cfg.layout)
    label_kinds = frozenset(cfg.layout.anchor.label_kinds)
    label_reader = next((n for n in names if ctx.readers.prints_text(n)), names[0])
    label_texts: dict[str, str] = {}
    for region in layout.regions:
        if region.kind.value not in label_kinds:
            continue
        reading = ctx.readers.read(label_reader, page.image, region, ReadHint())
        if reading.error is None and reading.text.strip():
            label_texts[region.id] = reading.text.strip()

    anchored = anchor_fields(
        layout.regions, fields, label_texts, config=cfg.layout.anchor
    )
    families = {n: ctx.readers.family(n) for n in names}
    fingerprint = reader_fingerprint((n, ctx.readers.version(n)) for n in names)
    by_key = {f.key: f for f in fields}
    out: list[FieldObservation] = []
    for hit in anchored:
        spec = by_key[hit.field_key]
        region = hit.value_region
        hint = _hint(spec)
        readings = [ctx.readers.read(n, page.image, region, hint) for n in names]
        result = vote(
            region.id,
            readings,
            field_type=spec.type.value,
            families=families,
        )
        text = result.candidate.text
        if not text:
            continue  # nothing readable: the column stays not_found
        validators = run_validators(
            spec.validators,
            text,
            ValidationContext(
                column_key=spec.key,
                document_id=document_id,
                page_types=(page.page_type,) if page.page_type else (),
            ),
        )
        agreement = check_agreement(
            result,
            readings,
            critical=spec.critical,
            validator_results=validators,
            rules=cfg.voting,
        )
        extra: list[str] = []
        if ctx.calibrator is None:
            probability, extra = 0.0, [UNCALIBRATED]
        elif ctx.calibrator.is_stale(fingerprint):
            probability, extra = 0.0, [CALIBRATOR_STALE]
        else:
            probability = ctx.calibrator.predict(
                extract_features(result, readings, validators, page.quality_score),
                spec.type.value,
                current_fingerprint=fingerprint,
            )
        decision = assign_status(
            located=True,
            vote_result=result,
            agreement=agreement,
            probability=probability,
            thresholds=cfg.confidence,
            has_evidence=True,
        )
        if decision.reasons or extra:
            reasons[(spec.key, region.id)] = ",".join([*decision.reasons, *extra])
        out.append(
            FieldObservation(
                column_key=spec.key,
                region_id=region.id,
                page_index=page.index,
                bbox=region.bbox,
                text=text,
                confidence=decision.confidence,
                status=decision.status,
                anchored=True,
                validator_results=validators,
            )
        )
    return out


def _attach_reasons(
    row: dict[str, FieldValue], reasons: Mapping[tuple[str, str], str]
) -> dict[str, FieldValue]:
    """Carry consensus reasons onto values the mapper left without one."""
    fixed: dict[str, FieldValue] = {}
    for key, fv in row.items():
        if fv.reason is None and fv.evidence and fv.status.value != "confirmed":
            why = reasons.get((key, fv.evidence[0].region_id))
            if why:
                fv = fv.model_copy(update={"reason": why})
        fixed[key] = fv
    return fixed


def process_file(path: Path, ctx: RunContext) -> ExtractionResult:
    """Run one file through the whole pipeline."""
    started = time.perf_counter()
    cfg = ctx.config
    ing = ingest(path, pipeline=cfg.pipeline)
    document = ing.document
    _preprocess(document, cfg)

    texts = _embedded_texts(ing.embedded_text)
    classifier = ctx.classifier or build_page_classifier(
        cfg.page_types, cfg.pipeline.classify, page_text_lookup=texts
    )
    classify_document(classifier, document)
    persons = assign_document_persons(document, cfg.page_types, page_text_lookup=texts)
    owners = [
        PageOwner(
            page_index=p.page_index,
            person_index=p.person_index,
            person_role=p.person_role,
        )
        for p in persons.pages
    ]

    reasons: dict[tuple[str, str], str] = {}
    observations: list[FieldObservation] = []
    for page in document.pages:
        observations.extend(_observe_page(page, ctx, document.id, reasons))

    plan = plan_rows(owners, observations, ctx.template)
    rows = LlmFieldMapper(ctx.llm_client).map_plan(plan, ctx.template)
    rows = [_attach_reasons(r, reasons) for r in rows]

    versions = {n: ctx.readers.version(n) for n in ctx.readers.names()}
    if ctx.llm_client is not None:
        versions["mapper_llm"] = ctx.llm_client.model_id
    log.info(
        "run_file_done",
        document_id=document.id,
        page_count=len(document.pages),
        row_count=len(rows),
        stage="run",
        status="ok",
    )
    return ExtractionResult(
        document_id=document.id,
        template_id=ctx.template.id,
        rows=rows,
        timings_ms={"total": (time.perf_counter() - started) * 1000.0},
        model_versions=versions,
        config_hash=cfg.config_hash,
    )


def run_files(paths: Sequence[Path], ctx: RunContext) -> RunReport:
    """Process each file; one bad file is reported and does not stop the rest."""
    results: list[ExtractionResult] = []
    failed: list[str] = []
    for path in paths:
        try:
            results.append(process_file(path, ctx))
        except IngestError as exc:
            log.warning("run_file_failed", stage="run", status="ingest_error")
            failed.append(f"{path.name}: {exc}")
    return RunReport(
        results=results,
        files_ok=len(results),
        files_failed=len(failed),
        failed_files=failed,
    )


def default_llm_client(config: TorusConfig) -> JsonLlmClient | None:
    return build_default_llm_client(config.pipeline.mapping)
