"""Skew, photo-perspective and bordered-page synth generators (S8 / P2.3)."""

from __future__ import annotations

import random
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from tests.fixtures.synth.generators.printed_form import generate_printed_form
from tests.fixtures.synth.types import SynthPage

SkewDirection = Literal["cw", "ccw"]


def generate_skewed(
    seed: int = 0,
    *,
    angle_deg: float | None = None,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Rotate a base page by a small skew angle (positive = clockwise)."""
    rng = random.Random(seed)
    if angle_deg is None:
        # Avoid near-zero angles that deskew correctly ignores.
        magnitude = 2.0 + 6.0 * rng.random()
        angle_deg = magnitude if rng.random() < 0.5 else -magnitude

    source = base if base is not None else generate_printed_form(seed)
    # Pillow rotate is CCW; negate so positive angle_deg is clockwise content skew.
    skewed = source.image.rotate(
        -angle_deg,
        expand=True,
        fillcolor=(255, 255, 255),
    ).convert("RGB")
    new_w, new_h = skewed.size
    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="skewed",
        seed=seed,
        image=skewed,
        page_index=idx,
        width=new_w,
        height=new_h,
        orientation=0,
        regions=(),
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "angle_deg": float(angle_deg),
            "base_width": source.width,
            "base_height": source.height,
        },
    )


def generate_perspective_photo(
    seed: int = 0,
    *,
    base: SynthPage | None = None,
    page_index: int | None = None,
    frame_pad: int | None = None,
) -> SynthPage:
    """Place a form on a dark desk with a mild perspective warp (phone photo)."""
    rng = random.Random(seed)
    source = base if base is not None else generate_printed_form(seed)
    pad = frame_pad if frame_pad is not None else 80 + rng.randint(0, 40)
    doc_w, doc_h = source.width, source.height
    canvas_w = doc_w + 2 * pad
    canvas_h = doc_h + 2 * pad

    # Destination corner jitter (source is the upright document rect).
    jitter = 18 + rng.randint(0, 22)
    dst = np.asarray(
        [
            [pad + rng.randint(-jitter, jitter), pad + rng.randint(-jitter, jitter)],
            [
                pad + doc_w + rng.randint(-jitter, jitter),
                pad + rng.randint(-jitter // 2, jitter),
            ],
            [
                pad + doc_w + rng.randint(-jitter, jitter),
                pad + doc_h + rng.randint(-jitter, jitter),
            ],
            [
                pad + rng.randint(-jitter, jitter),
                pad + doc_h + rng.randint(-jitter // 2, jitter),
            ],
        ],
        dtype=np.float32,
    )
    src = np.asarray(
        [
            [0.0, 0.0],
            [doc_w - 1.0, 0.0],
            [doc_w - 1.0, doc_h - 1.0],
            [0.0, doc_h - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    desk = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
    # Soft shadow under the paper.
    shadow = desk.copy()
    cv2.fillConvexPoly(shadow, dst.astype(np.int32), (18, 18, 18))
    desk = cv2.GaussianBlur(shadow, (21, 21), 0)

    doc_rgb = np.asarray(source.image.convert("RGB"))
    warped_doc = cv2.warpPerspective(
        doc_rgb,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    paper_mask = np.full((doc_h, doc_w), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        paper_mask,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_bool = warped_mask > 0
    composed = desk.copy()
    composed[mask_bool] = warped_doc[mask_bool]
    image = Image.fromarray(composed, mode="RGB")

    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="perspective_photo",
        seed=seed,
        image=image,
        page_index=idx,
        width=canvas_w,
        height=canvas_h,
        orientation=0,
        regions=(),
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "base_width": source.width,
            "base_height": source.height,
            "frame_pad": pad,
            "dest_quad": [(float(x), float(y)) for x, y in dst],
        },
    )


def generate_bordered(
    seed: int = 0,
    *,
    border: int | None = None,
    base: SynthPage | None = None,
    page_index: int | None = None,
) -> SynthPage:
    """Paste an upright form onto a dark border (no perspective warp)."""
    rng = random.Random(seed)
    source = base if base is not None else generate_printed_form(seed)
    b = border if border is not None else 40 + rng.randint(0, 40)
    canvas_w = source.width + 2 * b
    canvas_h = source.height + 2 * b
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(28, 28, 30))
    canvas.paste(source.image.convert("RGB"), (b, b))
    idx = source.page_index if page_index is None else page_index
    return SynthPage(
        kind="bordered",
        seed=seed,
        image=canvas,
        page_index=idx,
        width=canvas_w,
        height=canvas_h,
        orientation=0,
        regions=(),
        meta={
            "base_kind": source.kind,
            "base_seed": source.seed,
            "base_width": source.width,
            "base_height": source.height,
            "border": b,
        },
    )
