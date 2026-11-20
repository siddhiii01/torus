"""Apply EXIF orientation so phone photos render upright."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage


def apply_exif_orientation(image: PILImage.Image) -> tuple[PILImage.Image, bool]:
    """Return ``(image, transposed)`` with EXIF orientation applied when present.

    Uses Pillow's ``ImageOps.exif_transpose``. When the EXIF Orientation tag
    is missing or already 1, the image is returned unchanged (copied) and
    ``transposed`` is False.
    """
    from PIL import ImageOps

    # exif_transpose returns None only on failure in older Pillow; current
    # versions return a copy. Treat None as "no change".
    result = ImageOps.exif_transpose(image)
    if result is None:
        return image.copy(), False
    transposed = result is not image and result.size != image.size
    if not transposed:
        # Size can match for 180°; compare orientation tag when available.
        exif = image.getexif()
        orientation = exif.get(0x0112) if exif else None
        transposed = orientation is not None and int(orientation) not in (0, 1)
    return result, transposed
