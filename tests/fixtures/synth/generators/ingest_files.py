"""Synthetic on-disk files for ingest component tests (S8 / P2.1).

Writes only into a caller-supplied directory (usually ``tmp_path``).
Labels are generic placeholders — not real document content.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def write_synth_png(
    path: Path, *, label: str = "SYNTH", size: tuple[int, int] = (200, 120)
) -> Path:
    """Write a small RGB PNG with a non-PII label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(245, 245, 240))
    draw = ImageDraw.Draw(image)
    draw.text((10, 40), label, fill=(20, 20, 20))
    image.save(path, format="PNG")
    return path


def write_synth_jpeg_with_exif_orientation(
    path: Path,
    *,
    orientation: int = 6,
    label: str = "EXIF",
) -> Path:
    """Write a JPEG whose EXIF Orientation tag is set (default 6 = 90° CW).

    The pixel buffer is wider than tall; after EXIF transpose the upright
    image should be taller than wide for orientation 6.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Asymmetric: wide canvas with a mark near the right edge.
    image = Image.new("RGB", (120, 60), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 20, 115, 40), fill=(0, 0, 0))
    draw.text((5, 20), label, fill=(30, 30, 30))

    exif = image.getexif()
    exif[0x0112] = orientation
    image.save(path, format="JPEG", quality=90, exif=exif)
    return path


def write_synth_multipage_tiff(
    path: Path,
    *,
    page_count: int = 3,
    label_prefix: str = "PAGE",
) -> Path:
    """Write a multi-page TIFF with ``page_count`` frames."""
    if page_count < 1:
        msg = f"page_count must be >= 1, got {page_count}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    for index in range(page_count):
        frame = Image.new("RGB", (160, 100), color=(240, 240, 235))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 40), f"{label_prefix}{index}", fill=(10, 10, 10))
        frames.append(frame)
    first, rest = frames[0], frames[1:]
    first.save(path, format="TIFF", save_all=True, append_images=rest)
    return path


def write_synth_pdf_with_text(
    path: Path,
    *,
    lines: tuple[str, ...] = ("SYNTH LINE ONE", "SYNTH LINE TWO", "SYNTH LINE THREE"),
) -> Path:
    """Write a one-page PDF with a readable embedded text layer."""
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=595, height=842)
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=14)
            y += 24.0
        doc.save(path)
    finally:
        doc.close()
    return path


def write_synth_pdf_image_only(path: Path, *, label: str = "SCAN") -> Path:
    """Write a PDF whose page is a raster image (no useful text layer)."""
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    raster = Image.new("RGB", (400, 300), color=(250, 250, 248))
    draw = ImageDraw.Draw(raster)
    draw.text((20, 140), label, fill=(0, 0, 0))

    doc = pymupdf.open()
    try:
        page = doc.new_page(width=400, height=300)
        # Insert as image so the text layer stays empty / unusable.
        page.insert_image(page.rect, pixmap=_pil_to_pixmap(raster, pymupdf))
        doc.save(path)
    finally:
        doc.close()
    return path


def write_synth_pdf_multipage(path: Path, *, page_count: int = 2) -> Path:
    """Write a multi-page PDF with short placeholder labels."""
    import pymupdf

    if page_count < 1:
        msg = f"page_count must be >= 1, got {page_count}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        for index in range(page_count):
            page = doc.new_page(width=595, height=842)
            # Enough characters to pass embedded_text_min_chars (default 25).
            page.insert_text(
                (72, 72),
                f"SYNTH PDF PAGE {index} PLACEHOLDER LINE",
                fontsize=14,
            )
            page.insert_text(
                (72, 96),
                "SECOND LINE OF SYNTH TEXT",
                fontsize=14,
            )
        doc.save(path)
    finally:
        doc.close()
    return path


def _pil_to_pixmap(image: Image.Image, pymupdf: object) -> object:
    """Convert a PIL RGB image to a PyMuPDF pixmap without temp files."""
    rgb = image.convert("RGB")
    samples = rgb.tobytes()
    # pymupdf.Pixmap(colorspace, width, height, samples, alpha)
    return pymupdf.Pixmap(pymupdf.csRGB, rgb.width, rgb.height, samples, False)  # type: ignore[attr-defined]
