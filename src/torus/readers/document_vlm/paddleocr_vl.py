"""PaddleOCR-VL via local server (in-process paddle path later if bake-off needs it).

Licence: Apache 2.0 (see docs/SETUP.md). Default server tag: PaddleOCR-VL.
"""

from __future__ import annotations

from torus.readers.document_vlm._factory import make_create_reader

DEFAULT_MODEL = "PaddleOCR-VL"
create_reader = make_create_reader(DEFAULT_MODEL)
