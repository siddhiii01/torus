"""olmOCR via local model server. Licence: Apache 2.0 (see docs/SETUP.md)."""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "olmOCR-2"
create_reader = make_create_reader(DEFAULT_MODEL)
