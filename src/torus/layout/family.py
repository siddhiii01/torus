"""Form families: auto-save repeated label layouts as config hints (P4.7).

Positions live only in ``config/families/*.yaml``. Code never hard-codes
page coordinates. A family file is a *hint* that may bias placement search;
``anchor_fields`` must succeed without any family.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import yaml

from torus.config.loader import load_families
from torus.config.schemas import (
    FamilyLabelHint,
    FamilyOffset,
    FamilyPoint,
    FamilySize,
    FormFamilySpec,
    FormFamilyStatus,
    LabelAnchorConfig,
    LayoutFamiliesConfig,
)
from torus.core.models import BBox, Region
from torus.layout.relative import RelativePlacement


class _AnchoredLike(Protocol):
    """Minimal shape of ``AnchoredField`` (avoids importing ``anchor``)."""

    field_key: str
    label_region: Region
    value_region: Region
    placement: RelativePlacement


def _centre(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def _size(bbox: BBox) -> tuple[float, float]:
    return (max(0.0, bbox.x1 - bbox.x0), max(0.0, bbox.y1 - bbox.y0))


def _quantize(value: float, bin_size: float) -> int:
    return int(round(value / bin_size))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def family_id_for_fingerprint(fingerprint: str) -> str:
    """Stable auto id derived from the layout fingerprint."""
    return f"fam_{fingerprint}"


def fingerprint_layout(
    labels: Sequence[tuple[str, float, float]],
    *,
    position_bin: float = 0.05,
) -> str:
    """Translation-invariant fingerprint of field_key → label centres.

    ``labels`` is ``(field_key, centre_x, centre_y)``. Relative offsets are
    quantised so small OCR jitter does not spawn a new family; absolute page
    origin is not part of the hash (same printed form, shifted crop → same id).
    """
    if len(labels) < 1:
        msg = "fingerprint_layout requires at least one label"
        raise ValueError(msg)
    ordered = sorted(labels, key=lambda item: item[0])
    ox, oy = ordered[0][1], ordered[0][2]
    parts: list[str] = [ordered[0][0]]
    for key, cx, cy in ordered[1:]:
        dx = _quantize(cx - ox, position_bin)
        dy = _quantize(cy - oy, position_bin)
        parts.append(f"{key}:{dx}:{dy}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def fingerprint_anchored(
    anchored: Sequence[_AnchoredLike],
    *,
    position_bin: float = 0.05,
) -> str:
    """Fingerprint from anchored fields' label centres."""
    points = [(a.field_key, *_centre(a.label_region.bbox)) for a in anchored]
    return fingerprint_layout(points, position_bin=position_bin)


def labels_from_anchored(anchored: Sequence[_AnchoredLike]) -> list[FamilyLabelHint]:
    """Build label hints from a successful anchor pass (no document text stored)."""
    hints: list[FamilyLabelHint] = []
    for item in sorted(anchored, key=lambda a: a.field_key):
        lcx, lcy = _centre(item.label_region.bbox)
        lw, lh = _size(item.label_region.bbox)
        vcx, vcy = _centre(item.value_region.bbox)
        dx = max(-1.0, min(1.0, vcx - lcx))
        dy = max(-1.0, min(1.0, vcy - lcy))
        hints.append(
            FamilyLabelHint(
                field_key=item.field_key,
                label_center=FamilyPoint(x=_clamp01(lcx), y=_clamp01(lcy)),
                label_size=FamilySize(
                    w=max(1e-4, min(1.0, lw)),
                    h=max(1e-4, min(1.0, lh)),
                ),
                preferred_placement=item.placement.value,
                value_offset=FamilyOffset(dx=dx, dy=dy),
            )
        )
    return hints


def _blend_point(old: FamilyPoint, new: FamilyPoint, *, weight: float) -> FamilyPoint:
    """Running mean: weight is 1/(n) for the new observation after increment."""
    return FamilyPoint(
        x=_clamp01((1.0 - weight) * old.x + weight * new.x),
        y=_clamp01((1.0 - weight) * old.y + weight * new.y),
    )


def _blend_offset(
    old: FamilyOffset, new: FamilyOffset, *, weight: float
) -> FamilyOffset:
    return FamilyOffset(
        dx=max(-1.0, min(1.0, (1.0 - weight) * old.dx + weight * new.dx)),
        dy=max(-1.0, min(1.0, (1.0 - weight) * old.dy + weight * new.dy)),
    )


def _blend_size(old: FamilySize, new: FamilySize, *, weight: float) -> FamilySize:
    return FamilySize(
        w=max(1e-4, min(1.0, (1.0 - weight) * old.w + weight * new.w)),
        h=max(1e-4, min(1.0, (1.0 - weight) * old.h + weight * new.h)),
    )


def _merge_labels(
    previous: Sequence[FamilyLabelHint],
    observed: Sequence[FamilyLabelHint],
    *,
    observation_count: int,
) -> list[FamilyLabelHint]:
    """Blend centres for shared keys; keep previous placement unless missing."""
    weight = 1.0 / float(observation_count)
    prev_map = {lab.field_key: lab for lab in previous}
    out: list[FamilyLabelHint] = []
    seen: set[str] = set()
    for lab in observed:
        seen.add(lab.field_key)
        old = prev_map.get(lab.field_key)
        if old is None:
            out.append(lab)
            continue
        offset = lab.value_offset
        if old.value_offset is not None and lab.value_offset is not None:
            offset = _blend_offset(old.value_offset, lab.value_offset, weight=weight)
        elif old.value_offset is not None:
            offset = old.value_offset
        out.append(
            FamilyLabelHint(
                field_key=lab.field_key,
                label_center=_blend_point(
                    old.label_center, lab.label_center, weight=weight
                ),
                label_size=_blend_size(old.label_size, lab.label_size, weight=weight),
                preferred_placement=lab.preferred_placement or old.preferred_placement,
                value_offset=offset,
            )
        )
    for key, old in prev_map.items():
        if key not in seen:
            out.append(old)
    out.sort(key=lambda lab: lab.field_key)
    return out


def family_path(families_dir: Path, family_id: str) -> Path:
    return families_dir / f"{family_id}.yaml"


def save_family(spec: FormFamilySpec, families_dir: Path) -> Path:
    """Write a family YAML. Directory is created if missing."""
    families_dir.mkdir(parents=True, exist_ok=True)
    path = family_path(families_dir, spec.id)
    payload = spec.model_dump(mode="json")
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path


def observe_and_maybe_save(
    anchored: Sequence[_AnchoredLike],
    families_dir: Path,
    *,
    config: LayoutFamiliesConfig | None = None,
    template_id: str | None = None,
    page_type: str | None = None,
    existing: Mapping[str, FormFamilySpec] | None = None,
) -> FormFamilySpec | None:
    """Record a repeated label layout; auto-save when observation threshold is met.

    Returns the updated family spec, or ``None`` when families are disabled,
    auto-save is off, or too few labels were anchored.
    """
    cfg = config if config is not None else LayoutFamiliesConfig()
    if not cfg.enabled or not cfg.auto_save:
        return None
    if len(anchored) < cfg.min_labels:
        return None

    fp = fingerprint_anchored(anchored, position_bin=cfg.position_bin)
    fam_id = family_id_for_fingerprint(fp)
    observed_labels = labels_from_anchored(anchored)

    catalogue = dict(existing) if existing is not None else load_families(families_dir)
    prior = catalogue.get(fam_id)
    if prior is not None and prior.fingerprint == fp:
        count = prior.observation_count + 1
        status = prior.status
        if status is FormFamilyStatus.OBSERVING and count >= cfg.min_observations:
            status = FormFamilyStatus.GENERATED
        # Do not downgrade reviewed/rejected via auto-save.
        if prior.status in (FormFamilyStatus.REVIEWED, FormFamilyStatus.REJECTED):
            status = prior.status
        elif prior.status is FormFamilyStatus.GENERATED:
            status = FormFamilyStatus.GENERATED
        merged = _merge_labels(prior.labels, observed_labels, observation_count=count)
        spec = FormFamilySpec(
            id=fam_id,
            status=status,
            fingerprint=fp,
            observation_count=count,
            template_id=template_id if template_id is not None else prior.template_id,
            page_type=page_type if page_type is not None else prior.page_type,
            labels=merged,
            notes=prior.notes,
        )
    else:
        status = (
            FormFamilyStatus.GENERATED
            if cfg.min_observations <= 1
            else FormFamilyStatus.OBSERVING
        )
        spec = FormFamilySpec(
            id=fam_id,
            status=status,
            fingerprint=fp,
            observation_count=1,
            template_id=template_id,
            page_type=page_type,
            labels=observed_labels,
        )

    save_family(spec, families_dir)
    return spec


def _relative_geometry(
    labels: Sequence[FamilyLabelHint],
) -> dict[str, tuple[float, float]]:
    """Map field_key → (dx, dy) relative to the lexicographically first label."""
    if not labels:
        return {}
    ordered = sorted(labels, key=lambda lab: lab.field_key)
    ox = ordered[0].label_center.x
    oy = ordered[0].label_center.y
    return {
        lab.field_key: (
            lab.label_center.x - ox,
            lab.label_center.y - oy,
        )
        for lab in ordered
    }


def _geometry_distance(
    a: Mapping[str, tuple[float, float]],
    b: Mapping[str, tuple[float, float]],
) -> float | None:
    """Mean L2 distance over shared keys; ``None`` if fewer than two shared."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    total = 0.0
    for key in shared:
        ax, ay = a[key]
        bx, by = b[key]
        total += math.hypot(ax - bx, ay - by)
    return total / float(len(shared))


def match_family(
    anchored: Sequence[_AnchoredLike],
    families: Mapping[str, FormFamilySpec],
    *,
    config: LayoutFamiliesConfig | None = None,
) -> FormFamilySpec | None:
    """Find a stored family that matches this layout (fingerprint, then soft geom)."""
    cfg = config if config is not None else LayoutFamiliesConfig()
    if not cfg.enabled or len(anchored) < cfg.min_labels:
        return None

    hint_statuses = frozenset(cfg.use_as_hint)
    usable = [
        fam
        for fam in families.values()
        if fam.status.value in hint_statuses and len(fam.labels) >= cfg.min_labels
    ]
    if not usable:
        return None

    fp = fingerprint_anchored(anchored, position_bin=cfg.position_bin)
    for fam in usable:
        if fam.fingerprint == fp:
            return fam

    observed = labels_from_anchored(anchored)
    obs_geom = _relative_geometry(observed)
    best: FormFamilySpec | None = None
    best_dist = float("inf")
    for fam in usable:
        dist = _geometry_distance(obs_geom, _relative_geometry(fam.labels))
        if dist is None:
            continue
        if dist <= cfg.match_tolerance and dist < best_dist:
            best = fam
            best_dist = dist
    return best


def placement_prefer_for_field(
    family: FormFamilySpec | None,
    field_key: str,
    *,
    default_prefer: Sequence[str],
) -> list[str]:
    """Reorder ``prefer`` so the family's preferred placement comes first."""
    base = [p for p in default_prefer]
    if family is None:
        return base
    hint = next((lab for lab in family.labels if lab.field_key == field_key), None)
    if hint is None:
        return base
    preferred = hint.preferred_placement
    return [preferred] + [p for p in base if p != preferred]


def label_center_bonus(
    family: FormFamilySpec | None,
    field_key: str,
    label_bbox: BBox,
    *,
    match_tolerance: float,
) -> float:
    """Higher is better: how close a candidate label is to the family prior.

    Returns 0.0 when no hint applies. Used only to break ties among text matches.
    """
    if family is None:
        return 0.0
    hint = next((lab for lab in family.labels if lab.field_key == field_key), None)
    if hint is None:
        return 0.0
    cx, cy = _centre(label_bbox)
    dist = math.hypot(cx - hint.label_center.x, cy - hint.label_center.y)
    if dist > match_tolerance:
        return 0.0
    # Closer → larger bonus in [0, 1].
    return 1.0 - (dist / match_tolerance)


def anchor_config_for_field(
    base: LabelAnchorConfig,
    family: FormFamilySpec | None,
    field_key: str,
) -> LabelAnchorConfig:
    """Copy of ``base`` with ``prefer`` reordered from a family hint."""
    prefer = placement_prefer_for_field(family, field_key, default_prefer=base.prefer)
    if prefer == list(base.prefer):
        return base
    return base.model_copy(update={"prefer": prefer})


def regions_near_family_label(
    regions: Sequence[Region],
    family: FormFamilySpec,
    field_key: str,
    *,
    match_tolerance: float,
    label_kinds: frozenset[str],
    exclude_ids: frozenset[str] | None = None,
) -> list[Region]:
    """Regions near the family's stored label centre for ``field_key``.

    Hint helper only — callers must still match printed text to a field spec.
    """
    hint = next((lab for lab in family.labels if lab.field_key == field_key), None)
    if hint is None:
        return []
    blocked = exclude_ids if exclude_ids is not None else frozenset()
    hits: list[tuple[float, Region]] = []
    for region in regions:
        if region.id in blocked:
            continue
        if region.kind.value not in label_kinds:
            continue
        cx, cy = _centre(region.bbox)
        dist = math.hypot(cx - hint.label_center.x, cy - hint.label_center.y)
        if dist <= match_tolerance:
            hits.append((dist, region))
    hits.sort(key=lambda item: item[0])
    return [r for _, r in hits]
