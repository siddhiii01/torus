"""Local OCR pipeline. Text layer first, docTR when the page has no usable text."""

from ocr.engine import read_document

__all__ = ["read_document"]
