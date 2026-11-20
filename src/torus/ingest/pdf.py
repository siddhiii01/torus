"""PDF ingest via PyMuPDF: render pages at config dpi, assess text layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torus.config.schemas import IngestConfig
from torus.core.errors import IngestError
from torus.core.models import Page
from torus.ingest.embedded_text import PageEmbeddedText, assess_pdf_page_text


def load_pdf_pages(
    path: Path,
    *,
    dpi: int,
    ingest_config: IngestConfig,
) -> tuple[list[Page], list[PageEmbeddedText]]:
    """Render every PDF page and assess embedded text usability.

    Blank pages are kept. Password-protected PDFs raise :class:`IngestError`.
    """
    import pymupdf

    # pymupdf ships without stubs; treat the handle as Any.
    try:
        doc: Any = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 — surface as IngestError
        raise IngestError(f"cannot open PDF: {path.name}") from exc

    try:
        if getattr(doc, "needs_pass", False):
            raise IngestError(f"password-protected PDF: {path.name}")

        pages: list[Page] = []
        embedded: list[PageEmbeddedText] = []
        for index in range(int(doc.page_count)):
            pdf_page = doc[index]
            image = _render_page(pdf_page, dpi=dpi, pymupdf=pymupdf)
            width, height = image.size
            page = Page(index=index, width=width, height=height)
            page.bind_image(image)
            pages.append(page)
            embedded.append(assess_pdf_page_text(pdf_page, index, ingest_config))
        return pages, embedded
    finally:
        doc.close()


def _render_page(pdf_page: Any, *, dpi: int, pymupdf: Any) -> Any:
    """Render one PDF page to an RGB PIL image at ``dpi``."""
    from PIL import Image

    try:
        pix = pdf_page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    except Exception as exc:  # noqa: BLE001
        raise IngestError("failed to render PDF page") from exc
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
