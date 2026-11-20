"""Unit tests for ingest format detection and hashing (P2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from torus.core import IngestError
from torus.ingest import SUPPORTED_SUFFIXES, SourceFormat, detect_format
from torus.ingest.hashing import file_sha256


def test_detect_format_known_suffixes() -> None:
    assert detect_format(Path("a.PDF")) == SourceFormat.PDF
    assert detect_format(Path("b.jpeg")) == SourceFormat.JPEG
    assert detect_format(Path("c.JPG")) == SourceFormat.JPEG
    assert detect_format(Path("d.png")) == SourceFormat.PNG
    assert detect_format(Path("e.tif")) == SourceFormat.TIFF
    assert detect_format(Path("f.tiff")) == SourceFormat.TIFF


def test_detect_format_rejects_unknown() -> None:
    with pytest.raises(IngestError, match="unsupported"):
        detect_format(Path("notes.docx"))


def test_supported_suffixes_cover_task_formats() -> None:
    for suffix in (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        assert suffix in SUPPORTED_SUFFIXES


def test_file_sha256_stable(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"torus-synth-bytes")
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64
