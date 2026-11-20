"""P6.5 escalation on synthetic readings and a fake reader function."""

from __future__ import annotations

from typing import Any

from torus.config import load_config
from torus.config.schemas import EscalationConfig, EscalationTier, VotingRules
from torus.consensus import EscalationOutcome, escalate, grow_bbox
from torus.core.models import BBox, ReadHint, Reading, Region, ValidationResult

RULES = VotingRules(critical_min_families=2, noncritical_min_readers=1)
FAMILIES = {"p1": "paddle", "t1": "tess", "big": "vlm", "big2": "vlm2"}
REGION = Region(
    id="r",
    page_index=3,
    bbox=BBox(x0=0.2, y0=0.2, x1=0.4, y1=0.3),
    kind="field_value",
)


def _r(
    name: str, text: str, score: float | None = 0.95, error: str | None = None
) -> Reading:
    return Reading(
        region_id="r",
        reader_name=name,
        reader_version="1",
        text=text,
        raw_score=score,
        error=error,
    )


def _no_validators(_: str) -> list[ValidationResult]:
    return []


def _passing(_: str) -> list[ValidationResult]:
    return [ValidationResult(validator_name="fmt", passed=True)]


class FakeRead:
    """Scripted reader: answers by reader name and records every call."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, BBox, Any]] = []

    def __call__(
        self, name: str, image: Any, region: Region, hint: ReadHint | None
    ) -> Reading:
        self.calls.append((name, region.bbox, image))
        text = self.answers.get(name)
        if text is None:
            return _r(name, "", None, error="unavailable")
        return _r(name, text)


def _cfg(
    *tiers: EscalationTier, cap: int = 24, enabled: bool = True
) -> EscalationConfig:
    return EscalationConfig(enabled=enabled, tiers=list(tiers), max_extra_reads=cap)


def _run(
    readings: list[Reading],
    cfg: EscalationConfig,
    fake: FakeRead,
    *,
    critical: bool = True,
    validate: Any = _no_validators,
    variants: Any = None,
) -> EscalationOutcome:
    return escalate(
        REGION,
        "img",
        readings,
        read=fake,
        config=cfg,
        critical=critical,
        rules=RULES,
        validate=validate,
        families=FAMILIES,
        variants=variants,
    )


def test_agreement_already_met_makes_no_extra_reads() -> None:
    fake = FakeRead({"big": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]))
    out = _run([_r("p1", "42"), _r("t1", "42")], cfg, fake)
    assert out.resolved and not out.escalated and fake.calls == []


def test_critical_dissent_is_not_resolved_by_a_third_vote() -> None:
    fake = FakeRead({"big": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]))
    out = _run([_r("p1", "42"), _r("t1", "47")], cfg, fake)
    # The extra family agrees with 42, but t1 still dissents: never confirmable.
    assert out.escalated and out.extra_reads == 1
    assert out.vote.candidate.text == "42"
    assert "t1" in out.vote.dissenting_readers
    assert not out.resolved


def test_noncritical_weak_reading_resolved_by_stronger_reader() -> None:
    fake = FakeRead({"big": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]))
    weak = [_r("p1", "42", 0.3)]
    out = _run(weak, cfg, fake, critical=False, validate=_passing)
    assert out.resolved and out.tiers_run == 1


def test_stops_after_first_tier_that_resolves() -> None:
    fake = FakeRead({"big": "42", "big2": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]), EscalationTier(readers=["big2"]))
    out = _run([_r("p1", "42")], cfg, fake)
    assert out.resolved and out.tiers_run == 1
    assert [c[0] for c in fake.calls] == ["big"]


def test_unavailable_reader_is_skipped_and_stays_unresolved() -> None:
    fake = FakeRead({})
    cfg = _cfg(EscalationTier(readers=["big"]))
    out = _run([_r("p1", "42"), _r("t1", "43")], cfg, fake)
    assert not out.resolved and out.escalated
    assert len(out.readings) == 2


def test_extra_views_never_add_a_family() -> None:
    fake = FakeRead({"p1": "42"})
    cfg = _cfg(EscalationTier(readers=["p1"], crop_margins=[0.1, 0.2]))
    out = _run([_r("p1", "42")], cfg, fake)
    assert out.vote.agreeing_families == ["paddle"]
    assert not out.resolved


def test_wider_crop_and_variants_are_used() -> None:
    fake = FakeRead({"big": "42"})
    tier = EscalationTier(
        readers=["big"], crop_margins=[0.5], variants=["contrast", "unknown"]
    )
    _run(
        [_r("p1", "41")],
        _cfg(tier),
        fake,
        variants={"contrast": lambda im: f"{im}+c"},
    )
    boxes = [c[1] for c in fake.calls]
    assert boxes[0] == REGION.bbox
    assert boxes[1].x0 < REGION.bbox.x0 and boxes[1].y1 > REGION.bbox.y1
    assert [c[2] for c in fake.calls] == ["img", "img", "img+c"]


def test_extra_read_cap_is_hard() -> None:
    fake = FakeRead({"big": "x"})
    tier = EscalationTier(readers=["big"], crop_margins=[0.1, 0.2, 0.3])
    out = _run([_r("p1", "41")], _cfg(tier, cap=2), fake)
    assert out.extra_reads == 2 and len(fake.calls) == 2


def test_disabled_config_is_a_noop() -> None:
    fake = FakeRead({"big": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]), enabled=False)
    out = _run([_r("p1", "41")], cfg, fake)
    assert not out.escalated and fake.calls == []


def test_validator_reruns_on_new_text() -> None:
    seen: list[str] = []

    def validate(text: str) -> list[ValidationResult]:
        seen.append(text)
        return [ValidationResult(validator_name="fmt", passed=text == "42")]

    fake = FakeRead({"big": "42"})
    cfg = _cfg(EscalationTier(readers=["big"]))
    _run([_r("p1", "43", 0.5)], cfg, fake, critical=False, validate=validate)
    assert seen[0] == "43" and "42" in seen


def test_grow_bbox_clamps_to_page() -> None:
    g = grow_bbox(BBox(x0=0.0, y0=0.9, x1=0.5, y1=1.0), 0.5)
    assert g.x0 == 0.0 and g.y1 == 1.0 and g.y0 < 0.9 and g.x1 > 0.5


def test_default_pipeline_config_has_valid_escalation() -> None:
    cfg = load_config().pipeline.escalation
    assert cfg.enabled and cfg.tiers
