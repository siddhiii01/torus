"""Deterministic mapping (P8.4): direct fills skip the LLM."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from torus.config.schemas import Cardinality, FieldSpec, TemplateConfig
from torus.core.errors import MappingError
from torus.core.models import BBox, ValidationResult
from torus.core.types import FieldStatus, FieldType
from torus.mapping.deterministic import (
    REASON_NOT_VALIDATED,
    REASON_VALIDATOR_FAILED,
    decide_direct,
)
from torus.mapping.llm_mapper import REASON_LLM_DISABLED, LlmFieldMapper
from torus.mapping.rows import FieldObservation, PageOwner, plan_rows


class CountingClient:
    model_id = "fake"

    def __init__(self, reply: object | None = None) -> None:
        self.reply = reply
        self.calls: list[Sequence[dict[str, str]]] = []

    def complete_json(
        self, messages: Sequence[dict[str, str]], schema: Mapping[str, object]
    ) -> str:
        self.calls.append(messages)
        if self.reply is None:
            raise MappingError("unexpected call")
        return json.dumps(self.reply)


def _spec(key: str, validators: list[str] | None = None) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=key,
        description=f"synthetic {key}",
        type=FieldType.TEXT,
        cardinality=Cardinality.PER_PERSON,
        validators=validators or [],
    )


def _ok(name: str) -> ValidationResult:
    return ValidationResult(validator_name=name, passed=True)


def _obs(
    column: str = "c",
    rid: str = "r1",
    text: str = "v1",
    *,
    anchored: bool = True,
    status: FieldStatus = FieldStatus.CONFIRMED,
    conf: float = 0.9,
    results: list[ValidationResult] | None = None,
) -> FieldObservation:
    return FieldObservation(
        column_key=column,
        region_id=rid,
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
        text=text,
        confidence=conf,
        status=status,
        anchored=anchored,
        validator_results=results or [],
    )


def test_anchored_validated_single_value_fills_directly() -> None:
    fv = decide_direct(_spec("c", ["v"]), [_obs(results=[_ok("v")])])
    assert fv is not None
    assert (fv.value, fv.status, fv.confidence, fv.reason) == (
        "v1",
        FieldStatus.CONFIRMED,
        0.9,
        None,
    )
    assert [e.region_id for e in fv.evidence] == ["r1"]


def test_agreeing_regions_fill_directly_with_weakest_status() -> None:
    obs = [
        _obs(rid="r1", text="Same  1", conf=0.9),
        _obs(rid="r2", text="same 1", status=FieldStatus.LIKELY, conf=0.7),
    ]
    fv = decide_direct(_spec("c"), obs)
    assert fv is not None
    assert (fv.status, fv.confidence) == (FieldStatus.LIKELY, 0.7)
    assert {e.region_id for e in fv.evidence} == {"r1", "r2"}


def test_unanchored_or_conflicting_is_judgement() -> None:
    assert decide_direct(_spec("c"), [_obs(anchored=False)]) is None
    assert (
        decide_direct(_spec("c"), [_obs(rid="a", text="x1"), _obs(rid="b", text="y2")])
        is None
    )
    assert decide_direct(_spec("c"), []) is None


def test_failed_validator_is_review_and_not_corrected() -> None:
    bad = ValidationResult(validator_name="v", passed=False, message="m")
    fv = decide_direct(_spec("c", ["v"]), [_obs(results=[bad])])
    assert fv is not None
    assert fv.status is FieldStatus.NEEDS_REVIEW
    assert fv.value == "v1"
    assert fv.reason == f"{REASON_VALIDATOR_FAILED}:v"


def test_missing_required_validator_is_review() -> None:
    fv = decide_direct(_spec("c", ["v", "w"]), [_obs(results=[_ok("v")])])
    assert fv is not None
    assert fv.status is FieldStatus.NEEDS_REVIEW
    assert fv.reason == f"{REASON_NOT_VALIDATED}:w"


def test_status_is_never_raised() -> None:
    fv = decide_direct(_spec("c"), [_obs(status=FieldStatus.NEEDS_REVIEW)])
    assert fv is not None and fv.status is FieldStatus.NEEDS_REVIEW


def _plan(obs: list[FieldObservation], keys: list[str]):  # type: ignore[no-untyped-def]
    template = TemplateConfig(id="t", fields=[_spec(k) for k in keys])
    plan = plan_rows([PageOwner(page_index=0, person_index=0)], obs, template)
    return plan, template


def test_all_direct_row_makes_no_llm_call() -> None:
    plan, template = _plan([_obs("a", "r1"), _obs("b", "r2", "w2")], ["a", "b"])
    client = CountingClient()
    row = LlmFieldMapper(client).map_plan(plan, template)[0]
    assert client.calls == []
    assert row["a"].status is FieldStatus.CONFIRMED
    assert row["b"].value == "w2"


def test_llm_sees_only_judgement_columns() -> None:
    obs = [
        _obs("a", "r1"),
        _obs("b", "r2", "x1", anchored=False),
        _obs("b", "r3", "y2", anchored=False),
    ]
    plan, template = _plan(obs, ["a", "b"])
    client = CountingClient({"b": {"region_ids": ["r3"], "value": "y2"}})
    row = LlmFieldMapper(client).map_plan(plan, template)[0]
    asked = json.loads(client.calls[0][1]["content"])["columns"]
    assert [c["key"] for c in asked] == ["b"]
    assert row["a"].status is FieldStatus.CONFIRMED
    assert row["b"].status is FieldStatus.LIKELY


def test_without_llm_judgement_columns_go_to_review_direct_still_fill() -> None:
    obs = [_obs("a", "r1"), _obs("b", "r2", "x1", anchored=False)]
    plan, template = _plan(obs, ["a", "b"])
    row = LlmFieldMapper(None).map_plan(plan, template)[0]
    assert row["a"].status is FieldStatus.CONFIRMED
    assert row["b"].status is FieldStatus.NEEDS_REVIEW
    assert row["b"].reason == REASON_LLM_DISABLED
