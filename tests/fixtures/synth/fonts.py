"""Font resolution for synthetic fixtures.

Prefers optional public fonts under ``tests/fixtures/synth/fonts/`` (see
SOURCES.md). Falls back to common system paths, then Pillow's built-in bitmap
font so CI never needs a download.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# Relative names under fonts/ (optional; not committed by default).
_OPTIONAL_PRINTED = (
    "LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
)
_OPTIONAL_HANDWRITING = (
    "Caveat-Regular.ttf",
    "PatrickHand-Regular.ttf",
)

_SYSTEM_PRINTED = (
    # Windows
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    # Linux
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    # macOS
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)


@lru_cache(maxsize=16)
def load_printed_font(size: int = 24) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a printed (sans) font at ``size`` pixels."""
    for name in _OPTIONAL_PRINTED:
        path = _FONTS_DIR / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    for path in _SYSTEM_PRINTED:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=16)
def load_handwriting_font(
    size: int = 28,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a handwriting-style font when available; else printed fallback."""
    for name in _OPTIONAL_HANDWRITING:
        path = _FONTS_DIR / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return load_printed_font(size=size)


# Script -> candidate font files (optional OFL fonts first, then system paths).
_SCRIPT_FONTS: dict[str, tuple[Path, ...]] = {
    "devanagari": (
        _FONTS_DIR / "NotoSansDevanagari-Regular.ttf",
        Path(r"C:\Windows\Fonts\Nirmala.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"),
    ),
    "tamil": (
        _FONTS_DIR / "NotoSansTamil-Regular.ttf",
        Path(r"C:\Windows\Fonts\Nirmala.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"),
    ),
}


@lru_cache(maxsize=16)
def load_script_font(
    script: str, size: int = 24
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont | None:
    """Font covering ``script``, or ``None`` when none is installed."""
    for path in _SCRIPT_FONTS.get(script, ()):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return None


def text_size(
    draw: object,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    """Return ``(width, height)`` of ``text`` under ``font``."""
    # ImageDraw.textbbox is preferred; keep a duck-typed draw for tests.
    bbox = draw.textbbox((0, 0), text, font=font)  # type: ignore[attr-defined]
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
