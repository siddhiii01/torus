"""Component tests: checkbox/radio detect, fill, label map on random synth (P4.4)."""

from __future__ import annotations

import pytest
from tests.fixtures.synth import generate_checkbox, generate_checkbox_group

from torus.config import load_config
from torus.core import Page, Region, RegionKind
from torus.layout import (
    CheckboxFillState,
    bbox_iou,
    detect_checkboxes,
    detect_layout,
    locate_matches,
    map_labels_by_bbox,
    measure_checkbox_fill,
    process_checkboxes,
)

_MARK_TO_STATE = {
    "empty": CheckboxFillState.EMPTY,
    "tick": CheckboxFillState.TICKED,
    "cross": CheckboxFillState.CROSSED,
}


def _page_from_synth(synth: object, index: int = 0) -> Page:
    page = Page(
        index=index,
        width=synth.width,  # type: ignore[attr-defined]
        height=synth.height,  # type: ignore[attr-defined]
        orientation=synth.orientation,  # type: ignore[attr-defined]
    )
    page.bind_image(synth.image)  # type: ignore[attr-defined]
    return page


def _gt_checkboxes(synth: object) -> list[Region]:
    return [
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == RegionKind.CHECKBOX
    ]


def _gt_labels(synth: object) -> list[Region]:
    return [
        item.region
        for item in synth.regions  # type: ignore[attr-defined]
        if item.region.kind == RegionKind.TEXT_LINE
    ]


def _label_texts(synth: object) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in synth.regions:  # type: ignore[attr-defined]
        if item.region.kind == RegionKind.TEXT_LINE and item.text:
            out[item.region.id] = item.text
    return out


@pytest.mark.parametrize("seed", [1, 7, 14, 21, 28, 35, 42, 49])
@pytest.mark.parametrize("shape", ["square", "circle"])
@pytest.mark.parametrize("mark", ["empty", "tick", "cross"])
def test_checkbox_detected_on_random_layout(seed: int, shape: str, mark: str) -> None:
    synth = generate_checkbox(
        seed=seed,
        shape=shape,  # type: ignore[arg-type]
        mark=mark,  # type: ignore[arg-type]
        label="Option A",
        label_side="right",
    )
    cfg = load_config().pipeline.layout.checkbox
    found = detect_checkboxes(synth.image, page_index=0, config=cfg)
    assert found, f"no checkbox seed={seed} shape={shape} mark={mark}"
    gt = _gt_checkboxes(synth)[0]
    best = max(bbox_iou(d.region.bbox, gt.bbox) for d in found)
    assert best >= 0.3, f"IoU {best:.3f} too low seed={seed}"


def test_checkbox_location_recall_across_seeds() -> None:
    """≥97% of expected checkbox regions located across random seeds/shapes."""
    cfg = load_config().pipeline.layout
    matched = 0
    total = 0
    for seed in range(50, 100):
        for shape in ("square", "circle"):
            for mark in ("empty", "tick", "cross"):
                synth = generate_checkbox(
                    seed=seed,
                    shape=shape,  # type: ignore[arg-type]
                    mark=mark,  # type: ignore[arg-type]
                    label_side="right",
                )
                found = detect_checkboxes(
                    synth.image, page_index=0, config=cfg.checkbox
                )
                expected = _gt_checkboxes(synth)
                hit, n = locate_matches(
                    [d.region for d in found],
                    expected,
                    iou_threshold=cfg.locate_iou,
                )
                matched += hit
                total += n
    recall = matched / total if total else 0.0
    assert recall >= 0.97, f"checkbox recall {recall:.3f} < 0.97 ({matched}/{total})"


@pytest.mark.parametrize("seed", [2, 8, 16, 24, 32])
@pytest.mark.parametrize("mark", ["empty", "tick", "cross"])
def test_fill_state_matches_ground_truth(seed: int, mark: str) -> None:
    synth = generate_checkbox(
        seed=seed,
        shape="square",
        mark=mark,  # type: ignore[arg-type]
        label="Yes",
        label_side="right",
    )
    cfg = load_config().pipeline.layout.checkbox
    gt = _gt_checkboxes(synth)[0]
    # Isolate fill measurement from detection: use GT bbox.
    _ratio, state = measure_checkbox_fill(synth.image, gt, config=cfg)
    assert state is _MARK_TO_STATE[mark], f"seed={seed} mark={mark} got {state}"


@pytest.mark.parametrize("seed", [3, 11, 19, 27, 33])
def test_label_mapped_to_adjacent_option(seed: int) -> None:
    synth = generate_checkbox(
        seed=seed,
        shape="square",
        mark="empty",
        label="Agree",
        label_side="right",
    )
    cfg = load_config().pipeline.layout.checkbox
    labels = _gt_labels(synth)
    texts = _label_texts(synth)
    detections = detect_checkboxes(synth.image, page_index=0, config=cfg)
    assert detections
    # Use GT checkbox so this test isolates mapping.
    from torus.layout.checkbox import CheckboxDetection

    gt = _gt_checkboxes(synth)[0]
    det = CheckboxDetection(
        region=gt.model_copy(update={"label_anchor": None}),
        shape="square",
        fill_ratio=0.0,
        state=CheckboxFillState.EMPTY,
    )
    mapped = map_labels_by_bbox([det], labels, texts, config=cfg)
    assert mapped[0].region.label_anchor is not None
    assert mapped[0].region.label_anchor.text == "Agree"
    assert mapped[0].region.label_anchor.bbox is not None
    assert bbox_iou(mapped[0].region.label_anchor.bbox, labels[0].bbox) >= 0.5


@pytest.mark.parametrize("seed", [5, 15, 25, 35])
@pytest.mark.parametrize("shape", ["square", "circle"])
def test_checkbox_group_all_options_located(seed: int, shape: str) -> None:
    synth = generate_checkbox_group(seed=seed, n_options=3, shape=shape)  # type: ignore[arg-type]
    cfg = load_config().pipeline.layout
    found = detect_checkboxes(synth.image, page_index=0, config=cfg.checkbox)
    expected = _gt_checkboxes(synth)
    hit, n = locate_matches(
        [d.region for d in found],
        expected,
        iou_threshold=cfg.locate_iou,
    )
    assert hit == n, f"group locate {hit}/{n} seed={seed} shape={shape}"


def test_process_checkboxes_end_to_end() -> None:
    synth = generate_checkbox(
        seed=9, shape="square", mark="cross", label="Option B", label_side="right"
    )
    cfg = load_config().pipeline.layout.checkbox
    result = process_checkboxes(
        synth.image,
        page_index=0,
        config=cfg,
        label_regions=_gt_labels(synth),
        label_texts=_label_texts(synth),
    )
    assert result
    gt = _gt_checkboxes(synth)[0]
    best = max(result, key=lambda d: bbox_iou(d.region.bbox, gt.bbox))
    assert bbox_iou(best.region.bbox, gt.bbox) >= 0.3
    assert best.state is CheckboxFillState.CROSSED
    assert best.region.label_anchor is not None
    assert best.region.label_anchor.text == "Option B"


def test_detect_layout_includes_checkbox_source() -> None:
    synth = generate_checkbox(
        seed=12, shape="square", mark="tick", label="Yes", label_side="right"
    )
    page = _page_from_synth(synth)
    cfg = load_config().pipeline.layout
    result = detect_layout(page, cfg)
    assert "checkbox" in result.sources
    assert any(r.kind == RegionKind.CHECKBOX for r in result.regions)


def test_radio_detection_on_full_resolution_page_is_fast() -> None:
    """Hough runs on a downscaled copy: a 300 dpi A4 page must stay quick."""
    import time

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(image)
    cx, cy, r = 600, 900, 24
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=4)

    start = time.perf_counter()
    detections = detect_checkboxes(image)
    elapsed = time.perf_counter() - start

    assert elapsed < 15.0
    assert any(
        d.pixel_bbox is not None
        and d.pixel_bbox[0] <= cx <= d.pixel_bbox[2]
        and d.pixel_bbox[1] <= cy <= d.pixel_bbox[3]
        for d in detections
    )
