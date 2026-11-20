"""Stamp and signature overlays: detect, mask, signed yes/no (P4.6).

Detection reuses the classic OpenCV stamp/signature finders (ratios only —
no absolute page coordinates). Masking removes overlay ink so text under a
seal can be read; near-black printed text inside a stamp bbox is protected.
Signature presence on a page answers signed yes/no columns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict

from torus.config.schemas import LayoutClassicConfig, LayoutStampConfig
from torus.core.models import BBox, Region
from torus.core.types import RegionKind
from torus.layout.classic import detect_classic_regions
from torus.layout.geometry import bbox_iou

PixelBox = tuple[int, int, int, int]  # x0, y0, x1, y1
GrayU8 = NDArray[np.uint8]
RgbU8 = NDArray[np.uint8]

SignedAnswer = Literal["yes", "no"]


class OverlayPresence(StrEnum):
    """Whether an overlay region was found (for signed / stamped columns)."""

    YES = "yes"
    NO = "no"


class OverlayDetection(BaseModel):
    """One stamp or signature region with presence for template columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: Region
    presence: OverlayPresence = OverlayPresence.YES
    pixel_bbox: tuple[int, int, int, int] | None = None

    @property
    def signed(self) -> SignedAnswer:
        """Map a signature detection to a signed yes/no column value."""
        if self.region.kind != RegionKind.SIGNATURE:
            return "no"
        return "yes" if self.presence is OverlayPresence.YES else "no"


class OverlayProcessResult(BaseModel):
    """Detections plus a page image with overlays masked for reading."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    detections: tuple[OverlayDetection, ...] = ()
    masked_image: Image.Image | None = None
    # Page-level answers for columns such as "signed" / "stamped".
    signed: SignedAnswer = "no"
    stamped: SignedAnswer = "no"


def detect_stamp_and_signature_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutStampConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[Region]:
    """Locate ``STAMP`` and ``SIGNATURE`` regions via classic detectors."""
    cfg = config if config is not None else LayoutStampConfig()
    if not cfg.enabled:
        return []

    classic_cfg = _classic_overlay_config(cfg, classic_config)
    raw = detect_classic_regions(image, page_index=page_index, config=classic_cfg)
    kinds: set[RegionKind] = set()
    if cfg.detect_stamps:
        kinds.add(RegionKind.STAMP)
    if cfg.detect_signatures:
        kinds.add(RegionKind.SIGNATURE)
    return [r for r in raw if r.kind in kinds]


def detect_stamp_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutStampConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[Region]:
    """Locate only ``RegionKind.STAMP`` regions."""
    cfg = config if config is not None else LayoutStampConfig()
    return [
        r
        for r in detect_stamp_and_signature_regions(
            image,
            page_index=page_index,
            config=cfg,
            classic_config=classic_config,
        )
        if r.kind == RegionKind.STAMP
    ]


def detect_signature_regions(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutStampConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
) -> list[Region]:
    """Locate only ``RegionKind.SIGNATURE`` regions."""
    cfg = config if config is not None else LayoutStampConfig()
    return [
        r
        for r in detect_stamp_and_signature_regions(
            image,
            page_index=page_index,
            config=cfg,
            classic_config=classic_config,
        )
        if r.kind == RegionKind.SIGNATURE
    ]


def build_overlay_mask(
    image: Image.Image,
    overlays: list[Region],
    *,
    config: LayoutStampConfig | None = None,
) -> GrayU8:
    """Binary mask (255 = remove) of stamp chroma / signature ink in overlays."""
    cfg = config if config is not None else LayoutStampConfig()
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if height < 2 or width < 2 or not overlays:
        return mask

    gray = cast(GrayU8, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    for region in overlays:
        if region.kind not in (RegionKind.STAMP, RegionKind.SIGNATURE):
            continue
        x0, y0, x1, y1 = _norm_to_pixel(region.bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            continue
        roi_gray = gray[y0:y1, x0:x1]
        if region.kind == RegionKind.STAMP:
            local = _stamp_chroma_mask(
                cast(GrayU8, hue[y0:y1, x0:x1]),
                cast(GrayU8, sat[y0:y1, x0:x1]),
                cast(GrayU8, val[y0:y1, x0:x1]),
                roi_gray,
                cfg,
            )
        else:
            local = _signature_ink_mask(roi_gray, cfg)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)

    if cfg.mask_dilate_px > 0 and int(np.count_nonzero(mask)) > 0:
        k = 2 * cfg.mask_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cast(GrayU8, cv2.dilate(mask, kernel))
    return mask


def mask_overlays(
    image: Image.Image,
    overlays: list[Region],
    *,
    config: LayoutStampConfig | None = None,
) -> Image.Image:
    """Return a full-page RGB image with stamp/signature ink inpainted.

    Near-black printed text inside stamp boxes is kept out of the mask so
    underlying values remain readable after inpainting.
    """
    cfg = config if config is not None else LayoutStampConfig()
    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    if not overlays:
        return Image.fromarray(rgb.copy())

    mask = build_overlay_mask(image, overlays, config=cfg)
    if int(np.count_nonzero(mask)) == 0:
        return Image.fromarray(rgb.copy())

    cleaned = cast(
        RgbU8,
        cv2.inpaint(rgb, mask, cfg.inpaint_radius, cv2.INPAINT_TELEA),
    )
    return Image.fromarray(cleaned)


def signed_yes_no(regions: list[Region]) -> SignedAnswer:
    """Page-level signed column: yes if any signature region is present."""
    return "yes" if any(r.kind == RegionKind.SIGNATURE for r in regions) else "no"


def stamped_yes_no(regions: list[Region]) -> SignedAnswer:
    """Page-level stamped column: yes if any stamp region is present."""
    return "yes" if any(r.kind == RegionKind.STAMP for r in regions) else "no"


def merge_overlays_into_regions(
    regions: list[Region],
    overlays: list[Region],
    *,
    nms_iou: float = 0.4,
) -> list[Region]:
    """Append stamp/signature regions that do not overlap same-kind peers."""
    if not overlays:
        return list(regions)
    merged = list(regions)
    for cand in overlays:
        if cand.kind not in (RegionKind.STAMP, RegionKind.SIGNATURE):
            continue
        if any(
            r.kind == cand.kind and bbox_iou(r.bbox, cand.bbox) >= nms_iou
            for r in merged
        ):
            continue
        merged.append(cand)
    return merged


def process_overlays(
    image: Image.Image,
    *,
    page_index: int = 0,
    config: LayoutStampConfig | None = None,
    classic_config: LayoutClassicConfig | None = None,
    existing_regions: list[Region] | None = None,
) -> OverlayProcessResult:
    """Detect (or reuse) overlays, mask them, and answer signed/stamped."""
    cfg = config if config is not None else LayoutStampConfig()
    if not cfg.enabled:
        return OverlayProcessResult(
            detections=(),
            masked_image=Image.fromarray(
                cast(RgbU8, np.asarray(image.convert("RGB"))).copy()
            ),
            signed="no",
            stamped="no",
        )

    if existing_regions is not None:
        allowed: set[RegionKind] = set()
        if cfg.detect_stamps:
            allowed.add(RegionKind.STAMP)
        if cfg.detect_signatures:
            allowed.add(RegionKind.SIGNATURE)
        overlays = [r for r in existing_regions if r.kind in allowed]
    else:
        overlays = detect_stamp_and_signature_regions(
            image,
            page_index=page_index,
            config=cfg,
            classic_config=classic_config,
        )

    rgb = cast(RgbU8, np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    detections: list[OverlayDetection] = []
    for region in overlays:
        detections.append(
            OverlayDetection(
                region=region,
                presence=OverlayPresence.YES,
                pixel_bbox=_norm_to_pixel(region.bbox, width, height),
            )
        )

    masked = mask_overlays(image, overlays, config=cfg)
    return OverlayProcessResult(
        detections=tuple(detections),
        masked_image=masked,
        signed=signed_yes_no(overlays),
        stamped=stamped_yes_no(overlays),
    )


# --- internals -----------------------------------------------------------------


def _classic_overlay_config(
    stamp_cfg: LayoutStampConfig,
    classic_config: LayoutClassicConfig | None,
) -> LayoutClassicConfig:
    base = classic_config if classic_config is not None else LayoutClassicConfig()
    return base.model_copy(
        update={
            "enabled": True,
            "detect_text_lines": False,
            "detect_text_blocks": False,
            "detect_tables": False,
            "detect_photos": False,
            "detect_stamps": stamp_cfg.detect_stamps,
            "detect_signatures": stamp_cfg.detect_signatures,
            "stamp_min_saturation": stamp_cfg.stamp_min_saturation,
            "stamp_min_value": stamp_cfg.stamp_min_value,
            "signature_min_area_ratio": stamp_cfg.signature_min_area_ratio,
            "signature_max_area_ratio": stamp_cfg.signature_max_area_ratio,
            "signature_min_aspect": stamp_cfg.signature_min_aspect,
            "signature_max_height_ratio": stamp_cfg.signature_max_height_ratio,
            "signature_max_fill_ratio": stamp_cfg.signature_max_fill_ratio,
        }
    )


def _stamp_chroma_mask(
    hue: GrayU8,
    sat: GrayU8,
    val: GrayU8,
    gray: GrayU8,
    cfg: LayoutStampConfig,
) -> GrayU8:
    red = ((hue <= 10) | (hue >= 170)) & (sat >= cfg.stamp_min_saturation)
    blue = (hue >= 90) & (hue <= 130) & (sat >= cfg.stamp_min_saturation)
    chroma = (red | blue) & (val >= cfg.stamp_min_value)
    # Keep near-black printed text: do not inpaint dark gray pixels.
    protect = gray <= cfg.protect_text_max_gray
    keep = chroma & ~protect
    return cast(GrayU8, (keep.astype(np.uint8) * 255).astype(np.uint8))


def _signature_ink_mask(gray: GrayU8, cfg: LayoutStampConfig) -> GrayU8:
    ink = cast(
        GrayU8,
        ((gray < cfg.signature_ink_threshold).astype(np.uint8) * 255).astype(np.uint8),
    )
    # Thin sparse strokes: open slightly so solid printed blocks shrink.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cast(GrayU8, cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel))
    return opened


def _norm_to_pixel(bbox: BBox, width: int, height: int) -> PixelBox:
    x0 = max(0, min(width - 1, int(round(bbox.x0 * width))))
    y0 = max(0, min(height - 1, int(round(bbox.y0 * height))))
    x1 = max(x0 + 1, min(width, int(round(bbox.x1 * width))))
    y1 = max(y0 + 1, min(height, int(round(bbox.y1 * height))))
    return x0, y0, x1, y1
