"""dots.ocr via local model server. Licence: check model card before production."""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "dots.ocr"
create_reader = make_create_reader(DEFAULT_MODEL)
