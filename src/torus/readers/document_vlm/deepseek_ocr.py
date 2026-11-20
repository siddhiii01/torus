"""DeepSeek-OCR via local model server. Licence: check model card before production."""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "DeepSeek-OCR-2"
create_reader = make_create_reader(DEFAULT_MODEL)
