"""Unit tests for stamp/signature mask and presence helpers (P4.6)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from torus.config import LayoutConfig, LayoutStampConfig, load_config
from torus.core import BBox, Region, RegionKind
from torus.layout.stamp import (
    OverlayPresence,
    build_overlay_mask,
    mask_overlays,
    merge_overlays_into_regions,
    process_overlays,
    signed_yes_no,
    stamped_yes_no,
)


def test_layout_stamp_config_loads() -> None:
    cfg = load_config().pipeline.layout
    assert isinstance(cfg, LayoutConfig)
    assert cfg.stamp.enabled is True
    assert cfg.stamp.detect_stamps is True
    assert cfg.stamp.detect_signatures is True


def test_signed_yes_no_from_regions() -> None:
    sig = Region(
        id="s1",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.4, y1=0.3),
        kind=RegionKind.SIGNATURE,
    )
    stamp = Region(
        id="t1",
        page_index=0,
        bbox=BBox(x0=0.5, y0=0.5, x1=0.7, y1=0.7),
        kind=RegionKind.STAMP,
    )
    assert signed_yes_no([sig, stamp]) == "yes"
    assert signed_yes_no([stamp]) == "no"
    assert stamped_yes_no([sig, stamp]) == "yes"
    assert stamped_yes_no([sig]) == "no"


def test_merge_overlays_skips_same_kind_overlap() -> None:
    existing = Region(
        id="a",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.3),
        kind=RegionKind.STAMP,
    )
    dup = Region(
        id="b",
        page_index=0,
        bbox=BBox(x0=0.12, y0=0.12, x1=0.28, y1=0.28),
        kind=RegionKind.STAMP,
    )
    other = Region(
        id="c",
        page_index=0,
        bbox=BBox(x0=0.6, y0=0.6, x1=0.8, y1=0.75),
        kind=RegionKind.SIGNATURE,
    )
    merged = merge_overlays_into_regions([existing], [dup, other], nms_iou=0.4)
    ids = {r.id for r in merged}
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids


def test_stamp_mask_removes_chroma_keeps_dark_text() -> None:
    img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Red stamp ring.
    draw.ellipse((20, 20, 100, 100), outline=(200, 40, 40), width=6)
    # Dark printed text under the stamp.
    draw.rectangle((50, 55, 70, 65), fill=(20, 20, 20))

    stamp = Region(
        id="stamp",
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9),
        kind=RegionKind.STAMP,
    )
    cfg = LayoutStampConfig(protect_text_max_gray=70, stamp_min_saturation=40)
    mask = build_overlay_mask(img, [stamp], config=cfg)
    assert int(np.count_nonzero(mask)) > 0
    # Centre of the dark text block should not be marked for inpaint.
    assert mask[60, 60] == 0


def test_mask_overlays_reduces_red_chroma() -> None:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((15, 15, 85, 85), outline=(210, 30, 30), width=8)

    stamp = Region(
        id="stamp",
        page_index=0,
        bbox=BBox(x0=0.05, y0=0.05, x1=0.95, y1=0.95),
        kind=RegionKind.STAMP,
    )
    cleaned = mask_overlays(img, [stamp], config=LayoutStampConfig())
    before = np.asarray(img)
    after = np.asarray(cleaned)
    # Ring pixels were strongly red; after mask they should be less saturated.
    before_red = before[20, 50]
    after_red = after[20, 50]
    assert int(before_red[0]) - int(before_red[1]) > 80
    assert int(after_red[0]) - int(after_red[1]) < int(before_red[0]) - int(
        before_red[1]
    )


def test_process_overlays_empty_when_disabled() -> None:
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    result = process_overlays(img, config=LayoutStampConfig(enabled=False))
    assert result.detections == ()
    assert result.signed == "no"
    assert result.stamped == "no"
    assert result.masked_image is not None


def test_overlay_detection_signed_property() -> None:
    from torus.layout.stamp import OverlayDetection

    sig = OverlayDetection(
        region=Region(
            id="s",
            page_index=0,
            bbox=BBox(x0=0.1, y0=0.1, x1=0.4, y1=0.25),
            kind=RegionKind.SIGNATURE,
        ),
        presence=OverlayPresence.YES,
    )
    stamp = OverlayDetection(
        region=Region(
            id="t",
            page_index=0,
            bbox=BBox(x0=0.5, y0=0.5, x1=0.7, y1=0.7),
            kind=RegionKind.STAMP,
        ),
        presence=OverlayPresence.YES,
    )
    assert sig.signed == "yes"
    assert stamp.signed == "no"
