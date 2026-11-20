"""Label anchoring: read labels → match field specs → locate value regions (P4.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import FieldSpec, FormFamilySpec, LabelAnchorConfig
from torus.core.models import LabelAnchor, Reading, Region
from torus.core.protocols import Reader
from torus.core.types import RegionKind
from torus.layout.embed import CharNgramEmbedder, LabelEmbedder
from torus.layout.family import anchor_config_for_field, label_center_bonus
from torus.layout.match import LabelFieldMatch, LabelMatchKind, match_label_to_fields
from torus.layout.relative import RelativePlacement, find_value_near_label


class AnchoredField(BaseModel):
    """A template field located via its printed label and a nearby value region."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_key: str = Field(min_length=1)
    match: LabelFieldMatch
    label_region: Region
    value_region: Region
    placement: RelativePlacement


def read_region_texts(
    image: object,
    regions: Sequence[Region],
    reader: Reader,
    *,
    hint: object | None = None,
) -> dict[str, str]:
    """Run a printed-text reader on each region; soft-skip failures.

    ``hint`` is forwarded when it is a ``ReadHint``; otherwise the reader
    receives a default empty hint via ``Reading`` path — callers that have a
    real ``ReadHint`` should pass it.
    """
    from torus.core.models import ReadHint

    rh = hint if isinstance(hint, ReadHint) else ReadHint()
    out: dict[str, str] = {}
    for region in regions:
        try:
            reading: Reading = reader.read(image, region, rh)
        except Exception:  # noqa: BLE001 — reader hard fail → skip region
            continue
        if reading.error or not reading.text.strip():
            continue
        out[region.id] = reading.text.strip()
    return out


def anchor_fields(
    regions: Sequence[Region],
    fields: Sequence[FieldSpec],
    label_texts: Mapping[str, str],
    *,
    config: LabelAnchorConfig | None = None,
    embedder: LabelEmbedder | None = None,
    family_hint: FormFamilySpec | None = None,
    family_match_tolerance: float = 0.08,
) -> list[AnchoredField]:
    """Match labelled regions to field specs and attach nearby value regions.

    ``label_texts`` maps region id → printed OCR text (from a real reader or a
    mock). Positions come only from ``regions``; nothing is hard-coded.

    ``family_hint`` is optional (P4.7): it may reorder relative placement search
    and break ties toward stored label centres. Anchoring must succeed without it.
    """
    cfg = config if config is not None else LabelAnchorConfig()
    if not cfg.enabled:
        return []

    backend = embedder if embedder is not None else CharNgramEmbedder()
    label_kinds = frozenset(cfg.label_kinds)
    # Only non-passthrough / non-duplicate columns are anchored.
    target_fields = [f for f in fields if not f.passthrough and f.duplicate_of is None]
    if not target_fields:
        return []

    # Score every label-kind region that has text.
    candidates: list[tuple[Region, LabelFieldMatch]] = []
    for region in regions:
        if region.kind.value not in label_kinds:
            continue
        text = label_texts.get(region.id)
        if text is None or not text.strip():
            continue
        hit = match_label_to_fields(
            text,
            target_fields,
            embedder=backend,
            embedding_min=cfg.embedding_min,
        )
        if hit is None:
            continue
        candidates.append((region, hit))

    # Greedy: match kind, then score, then optional family centre proximity.
    candidates.sort(
        key=lambda item: (
            0 if item[1].kind is LabelMatchKind.EXACT else 1,
            0 if item[1].kind is LabelMatchKind.SYNONYM else 1,
            -item[1].score,
            -label_center_bonus(
                family_hint,
                item[1].field_key,
                item[0].bbox,
                match_tolerance=family_match_tolerance,
            ),
        )
    )

    used_fields: set[str] = set()
    used_labels: set[str] = set()
    anchored: list[AnchoredField] = []

    for label_region, match in candidates:
        if match.field_key in used_fields or label_region.id in used_labels:
            continue
        exclude = frozenset(used_labels) | {label_region.id}
        field_cfg = anchor_config_for_field(cfg, family_hint, match.field_key)
        found = find_value_near_label(
            label_region.bbox,
            regions,
            config=field_cfg,
            exclude_ids=exclude,
        )
        if found is None:
            continue
        value_region, placement = found
        # Avoid claiming another region's box that is itself a matched label.
        if value_region.id in used_labels:
            continue
        if value_region.id == label_region.id:
            continue

        value = Region(
            id=f"anchored:{match.field_key}:{label_region.page_index}",
            page_index=label_region.page_index,
            bbox=value_region.bbox,
            kind=RegionKind.FIELD_VALUE,
            label_anchor=LabelAnchor(text=match.label_text, bbox=label_region.bbox),
        )
        anchored.append(
            AnchoredField(
                field_key=match.field_key,
                match=match,
                label_region=label_region,
                value_region=value,
                placement=placement,
            )
        )
        used_fields.add(match.field_key)
        used_labels.add(label_region.id)
        used_labels.add(value_region.id)

    return anchored
