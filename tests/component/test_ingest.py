"""Component tests: ingest PDF/images with EXIF and embedded text (P2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.synth import (
    write_synth_jpeg_with_exif_orientation,
    write_synth_multipage_tiff,
    write_synth_pdf_image_only,
    write_synth_pdf_multipage,
    write_synth_pdf_with_text,
    write_synth_png,
)

from torus.config import IngestConfig, load_config
from torus.core import IngestError
from torus.ingest import ingest


def test_ingest_png_binds_image(tmp_path: Path) -> None:
    path = write_synth_png(tmp_path / "page.png", label="PNG")
    result = ingest(path, dpi=150)
    doc = result.document
    assert len(doc.pages) == 1
    assert doc.id  # content hash
    assert doc.metadata["format"] == "png"
    page = doc.pages[0]
    assert page.has_image is True
    assert page.image.size == (page.width, page.height)
    assert result.embedded_text == ()


def test_ingest_jpeg_applies_exif_orientation(tmp_path: Path) -> None:
    path = write_synth_jpeg_with_exif_orientation(tmp_path / "phone.jpg", orientation=6)
    result = ingest(path)
    page = result.document.pages[0]
    # Orientation 6 rotates 90° CW: 120x60 → 60x120.
    assert page.width == 60
    assert page.height == 120
    assert page.image.size == (60, 120)


def test_ingest_multipage_tiff(tmp_path: Path) -> None:
    path = write_synth_multipage_tiff(tmp_path / "pack.tiff", page_count=3)
    result = ingest(path)
    assert len(result.document.pages) == 3
    assert [p.index for p in result.document.pages] == [0, 1, 2]
    for page in result.document.pages:
        assert page.has_image is True


def test_ingest_pdf_renders_at_dpi_and_uses_text(tmp_path: Path) -> None:
    path = write_synth_pdf_with_text(tmp_path / "digital.pdf")
    result = ingest(path, dpi=100)
    assert result.dpi == 100
    assert len(result.document.pages) == 1
    page = result.document.pages[0]
    # A4 at 100 dpi ≈ 827 x 1169.
    assert page.width == pytest.approx(827, abs=5)
    assert page.height == pytest.approx(1169, abs=5)
    assert len(result.embedded_text) == 1
    assert result.embedded_text[0].usable is True
    assert result.embedded_text[0].word_count >= 3
    assert result.document.metadata["embedded_text_usable_pages"] == "1"


def test_ingest_pdf_image_only_text_not_usable(tmp_path: Path) -> None:
    path = write_synth_pdf_image_only(tmp_path / "scan.pdf")
    result = ingest(path, dpi=72, ingest_config=IngestConfig())
    assert len(result.document.pages) == 1
    assert result.document.pages[0].has_image is True
    assert len(result.embedded_text) == 1
    assert result.embedded_text[0].usable is False
    assert result.document.metadata["embedded_text_usable_pages"] == "0"


def test_ingest_multipage_pdf_keeps_all_pages(tmp_path: Path) -> None:
    path = write_synth_pdf_multipage(tmp_path / "multi.pdf", page_count=2)
    result = ingest(path, dpi=72)
    assert len(result.document.pages) == 2
    assert len(result.embedded_text) == 2
    assert all(item.usable for item in result.embedded_text)


def test_ingest_uses_pipeline_dpi(tmp_path: Path) -> None:
    # Load real repo config; dpi is 300 in config/pipeline.yaml.
    config = load_config()
    path = write_synth_png(tmp_path / "x.png")
    result = ingest(path, pipeline=config.pipeline)
    assert result.dpi == config.pipeline.dpi


def test_ingest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        ingest(tmp_path / "missing.pdf")


def test_ingest_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported"):
        ingest(path)


def test_ingest_same_bytes_same_id(tmp_path: Path) -> None:
    a = write_synth_png(tmp_path / "a.png", label="SAME")
    b = write_synth_png(tmp_path / "b.png", label="SAME")
    assert ingest(a).document.id == ingest(b).document.id
