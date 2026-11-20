"""Chandra OCR via local model server.

Licence: modified OpenRAIL-M (revenue limits) — needs human review before
production enablement (see docs/SETUP.md).
"""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "Chandra-OCR-2"
create_reader = make_create_reader(DEFAULT_MODEL)
