"""Component tests: stamp/signature detect, mask, signed yes/no (P4.6).

Uses random-layout synth seeds so detectors cannot rely on fixed coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.fixtures.synth import (
    generate_layout_page,
    generate_printed_form,
    generate_signature,
    generate_stamp,
)

from torus.config import load_config
from torus.core import Page, Region, RegionKind
from torus.layout import (
    bbox_iou,
    detect_layout,
    detect_signature_regions,
    detect_stamp_regions,
    locate_matches,
    mask_overlays,
    process_overlays,
    signed_yes_no,
    stamped_yes_no,
)


def _page_from_synth(synth: object, index: int = 0) -> Page:
    page = Page(
        index=index,
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def _gt_kind(synth: object, kind: RegionKind) -> list[Region]:
    return [
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == kind
    ]


@pytest.mark.parametrize("seed", [1, 7, 14, 21, 28, 35, 42])
def test_stamp_detected_on_random_overlay(seed: int) -> None:
    synth = generate_stamp(seed=seed)
    cfg = load_config().pipeline.layout.stamp
    found = detect_stamp_regions(synth.image, page_index=0, config=cfg)
    assert found, f"no stamp detected seed={seed}"
    gt = _gt_kind(synth, RegionKind.STAMP)[0]
    best = max(bbox_iou(r.bbox, gt.bbox) for r in found)
    assert best >= 0.3, f"IoU {best:.3f} too low seed={seed}"


@pytest.mark.parametrize("seed", [2, 9, 16, 23, 30, 37, 44])
def test_signature_detected_when_present(seed: int) -> None:
    synth = generate_signature(seed=seed, present=True)
    cfg = load_config().pipeline.layout.stamp
    found = detect_signature_regions(synth.image, page_index=0, config=cfg)
    assert found, f"no signature detected seed={seed}"
    gt = _gt_kind(synth, RegionKind.SIGNATURE)[0]
    best = max(bbox_iou(r.bbox, gt.bbox) for r in found)
    assert best >= 0.25, f"IoU {best:.3f} too low seed={seed}"


def test_stamp_location_recall_across_seeds() -> None:
    """≥97% of expected stamp regions located across random seeds."""
    cfg = load_config().pipeline.layout
    matched = 0
    total = 0
    for seed in range(50, 90):
        synth = generate_stamp(seed=seed)
        found = detect_stamp_regions(synth.image, page_index=0, config=cfg.stamp)
        expected = _gt_kind(synth, RegionKind.STAMP)
        hit, n = locate_matches(found, expected, iou_threshold=cfg.locate_iou)
        matched += hit
        total += n
    recall = matched / total if total else 0.0
    assert recall >= 0.97, f"stamp recall {recall:.3f} < 0.97 ({matched}/{total})"


def test_signature_location_recall_across_seeds() -> None:
    """≥97% of expected signature regions located across random seeds."""
    cfg = load_config().pipeline.layout
    matched = 0
    total = 0
    for seed in range(60, 100):
        synth = generate_signature(seed=seed, present=True)
        found = detect_signature_regions(synth.image, page_index=0, config=cfg.stamp)
        expected = _gt_kind(synth, RegionKind.SIGNATURE)
        hit, n = locate_matches(found, expected, iou_threshold=cfg.locate_iou)
        matched += hit
        total += n
    recall = matched / total if total else 0.0
    assert recall >= 0.97, f"signature recall {recall:.3f} < 0.97 ({matched}/{total})"


@pytest.mark.parametrize("seed", [3, 11, 19, 27])
def test_mask_reduces_stamp_chroma_over_text(seed: int) -> None:
    synth = generate_stamp(seed=seed)
    stamps = _gt_kind(synth, RegionKind.STAMP)
    assert stamps
    cfg = load_config().pipeline.layout.stamp
    cleaned = mask_overlays(synth.image, stamps, config=cfg)

    before = np.asarray(synth.image.convert("RGB"))
    after = np.asarray(cleaned.convert("RGB"))
    gt = stamps[0].bbox
    h, w = before.shape[:2]
    y0 = max(0, int(gt.y0 * h))
    y1 = min(h, int(gt.y1 * h))
    x0 = max(0, int(gt.x0 * w))
    x1 = min(w, int(gt.x1 * w))
    roi_before = before[y0:y1, x0:x1].astype(np.int16)
    roi_after = after[y0:y1, x0:x1].astype(np.int16)
    # Red-minus-green chroma should drop inside the stamp bbox.
    chroma_before = float(
        np.mean(np.clip(roi_before[:, :, 0] - roi_before[:, :, 1], 0, None))
    )
    chroma_after = float(
        np.mean(np.clip(roi_after[:, :, 0] - roi_after[:, :, 1], 0, None))
    )
    assert chroma_after < chroma_before * 0.85, (
        f"chroma {chroma_after:.1f} not reduced from {chroma_before:.1f} seed={seed}"
    )


@pytest.mark.parametrize("seed", [5, 15, 25])
def test_signed_yes_no_columns(seed: int) -> None:
    signed = generate_signature(seed=seed, present=True)
    unsigned = generate_signature(seed=seed, present=False)
    cfg = load_config().pipeline.layout.stamp

    signed_result = process_overlays(signed.image, config=cfg)
    unsigned_result = process_overlays(unsigned.image, config=cfg)

    assert signed.meta["signed"] == "yes"
    assert unsigned.meta["signed"] == "no"
    assert signed_result.signed == "yes"
    assert unsigned_result.signed == "no"
    assert signed_yes_no([d.region for d in signed_result.detections]) == "yes"
    assert signed_yes_no([d.region for d in unsigned_result.detections]) == "no"


@pytest.mark.parametrize("seed", [4, 12, 20])
def test_stamped_yes_no_from_process(seed: int) -> None:
    stamped = generate_stamp(seed=seed)
    plain = generate_printed_form(seed=seed, field_count=2)
    cfg = load_config().pipeline.layout.stamp

    stamped_result = process_overlays(stamped.image, config=cfg)
    plain_result = process_overlays(plain.image, config=cfg)

    assert stamped_result.stamped == "yes"
    assert plain_result.stamped == "no"
    assert stamped_yes_no([d.region for d in stamped_result.detections]) == "yes"


def test_detect_layout_includes_stamp_source() -> None:
    synth = generate_layout_page(seed=17)
    page = _page_from_synth(synth)
    cfg = load_config().pipeline.layout
    result = detect_layout(page, cfg)
    assert "stamp" in result.sources or RegionKind.STAMP in {
        r.kind for r in result.regions
    }
    kinds = {r.kind for r in result.regions}
    assert RegionKind.STAMP in kinds


def test_no_fixed_coordinates_in_stamp_module() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "src" / "torus" / "layout" / "stamp.py"
    text = path.read_text(encoding="utf-8")
    banned = ("x0 > 0.", "x0<", "page == 0", "page_index == 0")
    for token in banned:
        assert token not in text, f"stamp.py contains {token!r}"
