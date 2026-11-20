"""Public layout detection entry points (P4.1).

Combines the classic OpenCV detectors with an optional layout-model backend.
Model output is merged with classic regions using per-kind NMS.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import LayoutConfig
from torus.core.errors import ImageNotLoadedError, LayoutError
from torus.core.models import Page, Region
from torus.core.types import RegionKind
from torus.layout.checkbox import (
    detect_checkbox_regions,
    merge_checkboxes_into_regions,
)
from torus.layout.classic import detect_classic_regions
from torus.layout.comb import detect_comb_regions, merge_combs_into_regions
from torus.layout.geometry import bbox_iou
from torus.layout.model import LayoutModelBackend, load_layout_model
from torus.layout.stamp import (
    detect_stamp_and_signature_regions,
    merge_overlays_into_regions,
)
from torus.layout.table import detect_table_regions, merge_tables_into_regions


class LayoutResult(BaseModel):
    """Regions located on one page plus which detectors contributed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_index: int = Field(ge=0)
    regions: list[Region] = Field(default_factory=list)
    sources: tuple[str, ...] = ()


def detect_layout(
    page: Page,
    config: LayoutConfig | None = None,
    *,
    model: LayoutModelBackend | None = None,
) -> LayoutResult:
    """Locate layout regions on a page image.

    Classic detection always runs when enabled. The layout model runs only when
    configured and available; failures soft-skip (empty model regions).
    """
    cfg = config if config is not None else LayoutConfig()
    if not cfg.enabled:
        return LayoutResult(page_index=page.index, regions=[], sources=())

    if not page.has_image:
        raise ImageNotLoadedError(
            f"page {page.index} has no image or image loader bound"
        )

    try:
        image = page.image
    except ImageNotLoadedError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface as LayoutError
        raise LayoutError(f"failed to load page {page.index} image") from exc

    sources: list[str] = []
    regions: list[Region] = []

    if cfg.classic.enabled:
        try:
            classic = detect_classic_regions(
                image,
                page_index=page.index,
                config=cfg.classic,
            )
        except Exception as exc:  # noqa: BLE001 — classic failure is hard
            raise LayoutError(
                f"classic layout detection failed on page {page.index}"
            ) from exc
        regions.extend(classic)
        sources.append("classic")

    if cfg.comb.enabled:
        try:
            combs = detect_comb_regions(
                image,
                page_index=page.index,
                config=cfg.comb,
            )
        except Exception as exc:  # noqa: BLE001 — comb failure is hard
            raise LayoutError(f"comb detection failed on page {page.index}") from exc
        if combs:
            regions = merge_combs_into_regions(
                regions,
                combs,
                prefer_over_table_iou=cfg.comb.prefer_over_table_iou,
            )
            sources.append("comb")

    if cfg.checkbox.enabled:
        try:
            boxes = detect_checkbox_regions(
                image,
                page_index=page.index,
                config=cfg.checkbox,
            )
        except Exception as exc:  # noqa: BLE001 — checkbox failure is hard
            raise LayoutError(
                f"checkbox detection failed on page {page.index}"
            ) from exc
        if boxes:
            regions = merge_checkboxes_into_regions(
                regions,
                boxes,
                nms_iou=cfg.checkbox.nms_iou,
            )
            sources.append("checkbox")

    if cfg.table.enabled:
        try:
            table_cells = detect_table_regions(
                image,
                page_index=page.index,
                config=cfg.table,
                classic_config=cfg.classic,
            )
        except Exception as exc:  # noqa: BLE001 — table failure is hard
            raise LayoutError(f"table detection failed on page {page.index}") from exc
        if table_cells:
            regions = merge_tables_into_regions(
                regions,
                table_cells,
                nms_iou=cfg.table.nms_iou,
            )
            sources.append("table")

    if cfg.stamp.enabled:
        try:
            overlays = detect_stamp_and_signature_regions(
                image,
                page_index=page.index,
                config=cfg.stamp,
                classic_config=cfg.classic,
            )
        except Exception as exc:  # noqa: BLE001 — stamp failure is hard
            raise LayoutError(
                f"stamp/signature detection failed on page {page.index}"
            ) from exc
        if overlays:
            regions = merge_overlays_into_regions(
                regions,
                overlays,
                nms_iou=cfg.stamp.nms_iou,
            )
            sources.append("stamp")

    backend = model if model is not None else load_layout_model(cfg.model)
    if cfg.model.enabled:
        try:
            model_regions = backend.detect(image, page_index=page.index)
        except Exception:
            # Soft-skip: missing/broken model must not crash the pipeline.
            model_regions = []
        else:
            regions = _merge_regions(
                regions, model_regions, nms_iou=cfg.classic.nms_iou
            )
            sources.append(backend.name)

    return LayoutResult(
        page_index=page.index,
        regions=regions,
        sources=tuple(sources),
    )


def locate_matches(
    predicted: list[Region],
    expected: list[Region],
    *,
    iou_threshold: float = 0.3,
    require_kind: bool = True,
) -> tuple[int, int]:
    """Count how many expected regions have a matching prediction.

    Returns ``(matched, total_expected)``. Greedy best-IoU matching per
    expected region; each prediction may match at most once.
    """
    remaining = list(predicted)
    matched = 0
    for gt in expected:
        best_i = -1
        best_iou = 0.0
        for i, pred in enumerate(remaining):
            if require_kind and pred.kind != gt.kind:
                continue
            score = bbox_iou(pred.bbox, gt.bbox)
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i >= 0 and best_iou >= iou_threshold:
            matched += 1
            remaining.pop(best_i)
    return matched, len(expected)


def _merge_regions(
    primary: list[Region],
    secondary: list[Region],
    *,
    nms_iou: float,
) -> list[Region]:
    """Keep primary regions; add secondary when not overlapping same kind."""
    merged = list(primary)
    for cand in secondary:
        if any(
            r.kind == cand.kind and bbox_iou(r.bbox, cand.bbox) >= nms_iou
            for r in merged
        ):
            continue
        merged.append(cand)
    return merged


def regions_by_kind(regions: list[Region], kind: RegionKind) -> list[Region]:
    return [r for r in regions if r.kind == kind]
