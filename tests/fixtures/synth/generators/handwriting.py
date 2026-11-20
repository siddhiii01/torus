"""Handwriting-like text crops (optional public fonts + stroke jitter)."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFilter

from tests.fixtures.synth.fonts import load_handwriting_font, text_size
from tests.fixtures.synth.geometry import pixel_bbox_to_norm
from tests.fixtures.synth.types import RegionTruth, SynthPage
from torus.core import Region, RegionKind

_SAMPLE_PHRASES = (
    "alpha seven",
    "box twelve",
    "unit nine",
    "code forty",
)


def generate_handwriting(
    seed: int = 0,
    *,
    text: str | None = None,
    width: int = 560,
    height: int = 160,
    page_index: int = 0,
) -> SynthPage:
    """Render handwriting-like text with per-glyph jitter.

    When a public handwriting font is present under ``fonts/`` it is used;
    otherwise the printed fallback font is jittered to look pen-like.
    """
    rng = random.Random(seed)
    if text is None:
        text = _SAMPLE_PHRASES[rng.randint(0, len(_SAMPLE_PHRASES) - 1)]

    image = Image.new("RGB", (width, height), color=(255, 255, 252))
    draw = ImageDraw.Draw(image)
    font = load_handwriting_font(size=32)

    # Draw glyph-by-glyph with small position/rotation-free offsets.
    cursor_x = 28.0 + rng.uniform(0, 20)
    baseline = 50.0 + rng.uniform(0, 30)
    min_x, min_y = width, height
    max_x, max_y = 0.0, 0.0

    for ch in text:
        ox = rng.uniform(-1.5, 1.5)
        oy = rng.uniform(-2.5, 2.5)
        # Slight ink darkness variation.
        shade = int(rng.randint(5, 35))
        color = (shade, shade, shade)
        x = cursor_x + ox
        y = baseline + oy
        draw.text((x, y), ch, fill=color, font=font)
        cw, ch_h = text_size(draw, ch if ch != " " else "n", font)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + cw)
        max_y = max(max_y, y + ch_h)
        cursor_x += cw * (0.85 + rng.uniform(0, 0.25))

    # Soften edges slightly so strokes look less printer-sharp.
    image = image.filter(ImageFilter.GaussianBlur(radius=0.35))

    bbox = pixel_bbox_to_norm(
        x0=min_x - 4,
        y0=min_y - 4,
        x1=max_x + 4,
        y1=max_y + 4,
        width=width,
        height=height,
    )
    region = Region(
        id=f"handwriting-{seed}",
        page_index=page_index,
        bbox=bbox,
        kind=RegionKind.TEXT_LINE,
    )
    return SynthPage(
        kind="handwriting",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=(RegionTruth(region=region, text=text),),
        meta={"phrase": text},
    )


def generate_handwriting_page(
    seed: int = 0,
    *,
    width: int = 520,
    height: int = 400,
    page_index: int = 0,
) -> SynthPage:
    """Multi-line handwriting-like page for orientation / layout tests.

    Stacks two handwriting strips plus a short printed header so upright
    vs upside-down has a clear vertical ink-mass signal.
    """
    rng = random.Random(seed)
    top_phrase = _SAMPLE_PHRASES[rng.randint(0, len(_SAMPLE_PHRASES) - 1)]
    bottom_phrase = _SAMPLE_PHRASES[rng.randint(0, len(_SAMPLE_PHRASES) - 1)]
    top = generate_handwriting(seed, text=top_phrase, width=width - 40, height=140)
    bottom = generate_handwriting(
        seed + 1, text=bottom_phrase, width=width - 40, height=140
    )

    image = Image.new("RGB", (width, height), color=(255, 255, 252))
    draw = ImageDraw.Draw(image)
    header_font = load_handwriting_font(size=22)
    header = "NOTE LINE"
    draw.text((24, 16), header, fill=(20, 20, 20), font=header_font)
    image.paste(top.image, (20, 48))
    image.paste(bottom.image, (20, 220))

    regions = (
        RegionTruth(
            region=Region(
                id=f"handwriting-page-header-{seed}",
                page_index=page_index,
                bbox=pixel_bbox_to_norm(
                    x0=20, y0=12, x1=width - 20, y1=44, width=width, height=height
                ),
                kind=RegionKind.TEXT_LINE,
            ),
            text=header,
        ),
        RegionTruth(
            region=Region(
                id=f"handwriting-page-top-{seed}",
                page_index=page_index,
                bbox=pixel_bbox_to_norm(
                    x0=20, y0=48, x1=width - 20, y1=188, width=width, height=height
                ),
                kind=RegionKind.TEXT_LINE,
            ),
            text=top_phrase,
        ),
        RegionTruth(
            region=Region(
                id=f"handwriting-page-bottom-{seed}",
                page_index=page_index,
                bbox=pixel_bbox_to_norm(
                    x0=20, y0=220, x1=width - 20, y1=360, width=width, height=height
                ),
                kind=RegionKind.TEXT_LINE,
            ),
            text=bottom_phrase,
        ),
    )
    return SynthPage(
        kind="handwriting",
        seed=seed,
        image=image,
        page_index=page_index,
        width=width,
        height=height,
        orientation=0,
        regions=regions,
        meta={"phrases": f"{header}|{top_phrase}|{bottom_phrase}"},
    )
