"""Plug-in readers behind the Reader protocol.

Framework (P5.1): registry, lazy load, device from doctor, timeouts, batching,
and crop-hash + reader-version caching. Document VLMs (P5.3) and general VLMs
(P5.4) live under their packages. Character-cell MLP (P5.5) is
``torus.readers.char_cell``. All stay ``enabled: false`` until bake-off.
"""

from torus.readers.cache import ReadingCache
from torus.readers.crop_hash import cache_key, crop_hash, hash_crop_bytes
from torus.readers.device import resolve_device
from torus.readers.registry import ReaderRegistry, RegisteredReader
from torus.readers.runner import ReaderRunner
from torus.readers.timeout import run_with_timeout

__all__ = [
    "ReaderRegistry",
    "ReaderRunner",
    "ReadingCache",
    "RegisteredReader",
    "cache_key",
    "crop_hash",
    "hash_crop_bytes",
    "resolve_device",
    "run_with_timeout",
]
