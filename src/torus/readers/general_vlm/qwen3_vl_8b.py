"""Qwen3-VL 8B via the local model server."""

from __future__ import annotations

from torus.readers.general_vlm.qwen3_vl import make_create_reader

DEFAULT_MODEL = "qwen3-vl:8b"
create_reader = make_create_reader(DEFAULT_MODEL)
