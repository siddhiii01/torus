"""Escalation on disagreement (P6.5).

When the agreement bar (P6.2) is not met, ask stronger or slower readers, read
wider crops and read alternately preprocessed images, then re-vote. Tiers come
from ``config/pipeline.yaml`` (``escalation``) and run cheapest first. The loop
stops as soon as agreement is met; whatever is still unresolved is returned
as-is for status assignment (P6.4), which sends it to review.

Escalation only ever adds evidence. It never edits text, and extra views of one
reader share that reader's family, so they cannot satisfy a "two different
families" rule on their own. Any dissenting view still blocks a critical field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import EscalationConfig, VotingRules
from torus.consensus.agreement import AgreementDecision, check_agreement
from torus.consensus.vote import VoteResult, vote
from torus.consensus.weights import ReaderWeights
from torus.core.models import BBox, ReadHint, Reading, Region, ValidationResult
from torus.logging import get_logger

log = get_logger(__name__)

ReadFn = Callable[[str, Any, Region, ReadHint | None], Reading]
"""``ReaderRunner.read`` fits: (reader_name, image, region, hint) -> Reading."""

ValidateFn = Callable[[str], Sequence[ValidationResult]]
"""Validators re-run on each newly voted text; escalation may change the text."""

ImageVariant = Callable[[Any], Any]


class EscalationOutcome(BaseModel):
    """Final vote after escalation, with what it cost and why it stopped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vote: VoteResult
    agreement: AgreementDecision
    readings: list[Reading]
    escalated: bool = Field(description="True when at least one extra read was made.")
    tiers_run: int = Field(ge=0)
    extra_reads: int = Field(ge=0)
    resolved: bool = Field(description="True when agreement was met at the end.")


def grow_bbox(bbox: BBox, margin: float) -> BBox:
    """Grow by ``margin`` of the box's own size per side, clamped to the page."""
    dx = (bbox.x1 - bbox.x0) * margin
    dy = (bbox.y1 - bbox.y0) * margin
    return BBox(
        x0=max(0.0, bbox.x0 - dx),
        y0=max(0.0, bbox.y0 - dy),
        x1=min(1.0, bbox.x1 + dx),
        y1=min(1.0, bbox.y1 + dy),
    )


def escalate(
    region: Region,
    image: Any,
    readings: Sequence[Reading],
    *,
    read: ReadFn,
    config: EscalationConfig,
    critical: bool,
    rules: VotingRules,
    validate: ValidateFn,
    hint: ReadHint | None = None,
    families: Mapping[str, str] | None = None,
    weights: ReaderWeights | Mapping[str, float] | None = None,
    field_type: str | None = None,
    variants: Mapping[str, ImageVariant] | None = None,
) -> EscalationOutcome:
    """Re-read ``region`` tier by tier until agreement is met or tiers run out."""
    fam: dict[str, str] = dict(families or {})
    current = list(readings)

    def decide() -> tuple[VoteResult, AgreementDecision]:
        res = vote(
            region.id, current, weights=weights, field_type=field_type, families=fam
        )
        return res, check_agreement(
            res,
            current,
            critical=critical,
            validator_results=validate(res.candidate.text),
            rules=rules,
        )

    result, decision = decide()
    extra = 0
    tiers_run = 0
    if config.enabled:
        for tier in config.tiers:
            if decision.met or extra >= config.max_extra_reads:
                break
            tiers_run += 1
            for reader_name in tier.readers:
                views = _views(
                    region, image, tier.crop_margins, tier.variants, variants
                )
                for view, view_region, view_image in views:
                    if extra >= config.max_extra_reads:
                        break
                    label = reader_name if view == "base" else f"{reader_name}@{view}"
                    if _already_read(current, label):
                        continue
                    extra += 1
                    got = read(reader_name, view_image, view_region, hint)
                    if got.error is not None:
                        # Unavailable or failed readers abstain; never a crash.
                        continue
                    if label != reader_name:
                        fam.setdefault(label, fam.get(reader_name, reader_name))
                    current.append(
                        got.model_copy(
                            update={"reader_name": label, "region_id": region.id}
                        )
                    )
            result, decision = decide()

    log.info(
        "escalation_finished",
        region_id=region.id,
        tiers_run=tiers_run,
        extra_reads=extra,
        resolved=decision.met,
        status_code="resolved" if decision.met else "unresolved",
    )
    return EscalationOutcome(
        vote=result,
        agreement=decision,
        readings=current,
        escalated=extra > 0,
        tiers_run=tiers_run,
        extra_reads=extra,
        resolved=decision.met,
    )


def _views(
    region: Region,
    image: Any,
    margins: Sequence[float],
    variant_names: Sequence[str],
    variants: Mapping[str, ImageVariant] | None,
) -> list[tuple[str, Region, Any]]:
    """Base view first, then wider crops, then image variants (unknown ones skipped)."""
    out: list[tuple[str, Region, Any]] = [("base", region, image)]
    for m in margins:
        grown = region.model_copy(update={"bbox": grow_bbox(region.bbox, m)})
        out.append((f"margin{m:g}", grown, image))
    for name in variant_names:
        fn = (variants or {}).get(name)
        if fn is None:
            continue
        out.append((name, region, fn(image)))
    return out


def _already_read(readings: Sequence[Reading], label: str) -> bool:
    return any(r.reader_name == label and r.error is None for r in readings)
