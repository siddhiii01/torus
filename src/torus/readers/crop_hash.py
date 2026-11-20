"""Hash region crops for the reading cache. Never logs pixel content."""

from __future__ import annotations

import hashlib
from typing import Any

from torus.core.models import BBox, ReadHint, Region


def pixel_box(size: tuple[int, int], bbox: BBox) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) for a normalised bbox; at least 1 px wide/high."""
    width, height = int(size[0]), int(size[1])
    x0 = max(0, min(width, int(bbox.x0 * width)))
    y0 = max(0, min(height, int(bbox.y0 * height)))
    x1 = max(x0 + 1, min(width, int(bbox.x1 * width)))
    y1 = max(y0 + 1, min(height, int(bbox.y1 * height)))
    return x0, y0, x1, y1


def crop_image(image: Any, bbox: BBox) -> Any:
    """Return a PIL crop of ``image`` for a normalised bbox (0–1, top-left).

    Non-PIL inputs are returned unchanged (caller hashes ``repr`` fallback).
    """
    size = getattr(image, "size", None)
    crop = getattr(image, "crop", None)
    if size is None or crop is None or not callable(crop):
        return image
    if int(size[0]) <= 0 or int(size[1]) <= 0:
        return image
    return crop(pixel_box((int(size[0]), int(size[1])), bbox))


def hash_crop_bytes(image: Any) -> str:
    """SHA-256 of RGB pixel bytes (or raw bytes / stable fallback)."""
    digest = hashlib.sha256()
    tobytes = getattr(image, "tobytes", None)
    mode = getattr(image, "mode", None)
    size = getattr(image, "size", None)
    if callable(tobytes) and mode is not None and size is not None:
        # Include mode/size so identical buffers with different layouts differ.
        digest.update(str(mode).encode("ascii"))
        digest.update(f"{int(size[0])}x{int(size[1])}".encode("ascii"))
        convert = getattr(image, "convert", None)
        rgb = convert("RGB") if callable(convert) and mode != "RGB" else image
        digest.update(rgb.tobytes())
        return digest.hexdigest()

    to_bytes = getattr(image, "to_bytes", None)
    if callable(to_bytes):
        digest.update(to_bytes())
        return digest.hexdigest()

    if isinstance(image, (bytes, bytearray, memoryview)):
        digest.update(bytes(image))
        return digest.hexdigest()

    # Last resort for synthetic test doubles: identity of the object type+id
    # is unstable across runs, so use a stable empty marker instead.
    digest.update(b"unhashable-crop")
    return digest.hexdigest()


def crop_hash(image: Any, region: Region) -> str:
    """Hash the cropped pixels for ``region`` on ``image``."""
    cropped = crop_image(image, region.bbox)
    return hash_crop_bytes(cropped)


def hint_digest(hint: ReadHint) -> str:
    """Stable short digest of decoding constraints (part of the cache key)."""
    payload = (
        f"{hint.charset.value}|{hint.script or ''}|{hint.language or ''}|"
        f"{hint.expected_length}|{hint.min_length}|{hint.max_length}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(
    *,
    crop_digest: str,
    reader_name: str,
    reader_version: str,
    hint: ReadHint,
) -> str:
    """Cache key: crop hash + reader identity + hint constraints."""
    return "|".join(
        (
            crop_digest,
            reader_name,
            reader_version,
            hint_digest(hint),
        )
    )
