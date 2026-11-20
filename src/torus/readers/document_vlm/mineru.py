"""MinerU document VLM via local model server. Licence: check model card."""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "MinerU2.5-Pro"
create_reader = make_create_reader(DEFAULT_MODEL)
