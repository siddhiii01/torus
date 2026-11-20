"""Public ingest entry: path → Document + embedded-text assessment."""

from __future__ import annotations

from pathlib import Path

from torus.config.schemas import IngestConfig, PipelineConfig
from torus.core.errors import IngestError
from torus.core.models import Document
from torus.ingest.formats import SourceFormat, detect_format
from torus.ingest.hashing import file_sha256
from torus.ingest.images import load_image_pages
from torus.ingest.pdf import load_pdf_pages
from torus.ingest.result import IngestResult
from torus.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_DPI = 300


def ingest(
    path: Path | str,
    *,
    dpi: int | None = None,
    pipeline: PipelineConfig | None = None,
    ingest_config: IngestConfig | None = None,
) -> IngestResult:
    """Load a PDF or image into a :class:`~torus.core.models.Document`.

    Parameters
    ----------
    path:
        Source file (pdf / jpg / png / tiff).
    dpi:
        Render resolution for PDFs. Overrides ``pipeline.dpi`` when set.
    pipeline:
        Optional pipeline config (dpi + ingest thresholds).
    ingest_config:
        Optional embedded-text thresholds; overrides ``pipeline.ingest``.
    """
    source = Path(path)
    if not source.is_file():
        raise IngestError(f"file not found: {source.name}")

    resolved_dpi = _resolve_dpi(dpi=dpi, pipeline=pipeline)
    thresholds = _resolve_ingest_config(ingest_config=ingest_config, pipeline=pipeline)
    source_format = detect_format(source)
    file_hash = file_sha256(source)

    if source_format == SourceFormat.PDF:
        pages, embedded = load_pdf_pages(
            source, dpi=resolved_dpi, ingest_config=thresholds
        )
    else:
        pages = load_image_pages(source, source_format=source_format)
        embedded = []

    usable_count = sum(1 for item in embedded if item.usable)
    document = Document(
        id=file_hash,
        source_path=str(source.resolve()),
        pages=pages,
        metadata={
            "format": source_format.value,
            "dpi": str(resolved_dpi),
            "page_count": str(len(pages)),
            "embedded_text_usable_pages": str(usable_count),
        },
    )

    _log.info(
        "ingest_done",
        file_hash=file_hash,
        page_count=len(pages),
        stage="ingest",
        status="ok",
    )

    return IngestResult(
        document=document,
        embedded_text=tuple(embedded),
        dpi=resolved_dpi,
    )


def _resolve_dpi(*, dpi: int | None, pipeline: PipelineConfig | None) -> int:
    if dpi is not None:
        if not 72 <= dpi <= 1200:
            raise IngestError(f"dpi must be in 72..1200, got {dpi}")
        return dpi
    if pipeline is not None:
        return pipeline.dpi
    return _DEFAULT_DPI


def _resolve_ingest_config(
    *,
    ingest_config: IngestConfig | None,
    pipeline: PipelineConfig | None,
) -> IngestConfig:
    if ingest_config is not None:
        return ingest_config
    if pipeline is not None:
        return pipeline.ingest
    return IngestConfig()
