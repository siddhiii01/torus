"""LLM field mapper (P8.3) with a fake client: citation rules, status caps, locality.

Synthetic data only; no network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from torus.config.schemas import Cardinality, FieldSpec, MappingConfig, TemplateConfig
from torus.core.errors import MappingError
from torus.core.models import BBox
from torus.core.types import FieldStatus, FieldType
from torus.mapping.llm_client import (
    LocalHttpLlmClient,
    assert_local_url,
    build_default_llm_client,
    clean_json_text,
    parse_chat_content,
)
from torus.mapping.llm_mapper import (
    REASON_BAD_CITATION,
    REASON_COMPOSED,
    REASON_DECLINED,
    REASON_LLM_ERROR,
    REASON_LLM_INVALID,
    REASON_NO_CANDIDATES,
    REASON_OMITTED,
    REASON_SELECTED,
    REASON_UNASSIGNED,
    REASON_UNCITED,
    REASON_UNSUPPORTED,
    LlmFieldMapper,
)
from torus.mapping.rows import FieldObservation, PageOwner, plan_rows


class FakeClient:
    model_id = "fake-qwen3"

    def __init__(self, reply: object | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], Mapping[str, object]]] = []

    def complete_json(
        self, messages: Sequence[dict[str, str]], schema: Mapping[str, object]
    ) -> str:
        self.calls.append((list(messages), schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def _spec(key: str) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=key,
        description=f"synthetic {key}",
        type=FieldType.TEXT,
        cardinality=Cardinality.PER_PERSON,
    )


TEMPLATE = TemplateConfig(id="t", fields=[_spec("col_a"), _spec("col_b")])


def _obs(
    column: str,
    rid: str,
    text: str,
    status: FieldStatus = FieldStatus.CONFIRMED,
    conf: float = 0.95,
) -> FieldObservation:
    return FieldObservation(
        column_key=column,
        region_id=rid,
        page_index=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
        text=text,
        confidence=conf,
        status=status,
    )


def _run(obs: list[FieldObservation], reply: object | Exception):  # type: ignore[no-untyped-def]
    plan = plan_rows([PageOwner(page_index=0, person_index=0)], obs, TEMPLATE)
    client = FakeClient(reply)
    rows = LlmFieldMapper(client).map_plan(plan, TEMPLATE)
    assert len(rows) == 1
    return rows[0], client


def _ans(ids: list[str], value: str | None) -> dict[str, object]:
    return {"region_ids": ids, "value": value}


def test_sole_candidate_verbatim_keeps_consensus_status() -> None:
    row, _ = _run(
        [_obs("col_a", "r1", "alpha one")], {"col_a": _ans(["r1"], "Alpha One")}
    )
    fv = row["col_a"]
    assert (fv.value, fv.status, fv.confidence) == (
        "alpha one",
        FieldStatus.CONFIRMED,
        0.95,
    )
    assert [e.region_id for e in fv.evidence] == ["r1"]


def test_no_candidates_is_not_found_and_makes_no_llm_call() -> None:
    row, client = _run([], {})
    assert row["col_a"].status is FieldStatus.NOT_FOUND
    assert row["col_a"].reason == REASON_NO_CANDIDATES
    assert client.calls == []


def test_uncited_value_is_rejected() -> None:
    row, _ = _run([_obs("col_a", "r1", "x1")], {"col_a": _ans([], "x1")})
    assert row["col_a"].status is FieldStatus.NEEDS_REVIEW
    assert row["col_a"].value is None
    assert row["col_a"].reason == REASON_UNCITED


def test_citing_a_region_from_another_column_is_rejected() -> None:
    obs = [_obs("col_a", "r1", "x1"), _obs("col_b", "r2", "y2")]
    row, _ = _run(obs, {"col_a": _ans(["r2"], "y2"), "col_b": _ans(["r2"], "y2")})
    assert row["col_a"].reason == REASON_BAD_CITATION
    assert row["col_b"].status is FieldStatus.CONFIRMED


def test_value_not_in_cited_text_is_rejected() -> None:
    row, _ = _run(
        [_obs("col_a", "r1", "alpha one")], {"col_a": _ans(["r1"], "alpha two")}
    )
    assert row["col_a"].reason == REASON_UNSUPPORTED
    assert row["col_a"].value is None


def test_declining_existing_candidates_is_review_not_not_found() -> None:
    row, _ = _run([_obs("col_a", "r1", "x1")], {"col_a": _ans([], None)})
    assert row["col_a"].status is FieldStatus.NEEDS_REVIEW
    assert row["col_a"].reason == REASON_DECLINED


def test_choosing_between_conflicting_candidates_is_capped_at_likely() -> None:
    obs = [_obs("col_a", "r1", "same 1"), _obs("col_a", "r2", "other 2")]
    row, _ = _run(obs, {"col_a": _ans(["r2"], "other 2")})
    fv = row["col_a"]
    assert (fv.value, fv.status, fv.reason) == (
        "other 2",
        FieldStatus.LIKELY,
        REASON_SELECTED,
    )
    assert [e.region_id for e in fv.evidence] == ["r2"]


def test_composed_value_is_capped_and_never_exceeds_weakest_cited() -> None:
    obs = [
        _obs("col_a", "r1", "house 5", FieldStatus.CONFIRMED, 0.9),
        _obs("col_a", "r2", "lane 3", FieldStatus.NEEDS_REVIEW, 0.4),
    ]
    row, _ = _run(obs, {"col_a": _ans(["r1", "r2"], "house 5 lane 3")})
    fv = row["col_a"]
    assert fv.status is FieldStatus.NEEDS_REVIEW
    assert fv.confidence == 0.4
    assert fv.reason == REASON_SELECTED  # two distinct candidate texts
    assert {e.region_id for e in fv.evidence} == {"r1", "r2"}


def test_composed_from_confirmed_cited_is_likely_never_confirmed() -> None:
    obs = [_obs("col_a", "r1", "one"), _obs("col_a", "r2", "two")]
    row, _ = _run(obs, {"col_a": _ans(["r1", "r2"], "one two")})
    assert row["col_a"].status is FieldStatus.LIKELY
    assert row["col_a"].reason in {REASON_SELECTED, REASON_COMPOSED}


def test_omitted_column_goes_to_review() -> None:
    obs = [_obs("col_a", "r1", "x1"), _obs("col_b", "r2", "y2")]
    row, _ = _run(obs, {"col_a": _ans(["r1"], "x1")})
    assert row["col_b"].reason == REASON_OMITTED


def test_llm_error_sends_every_asked_column_to_review() -> None:
    obs = [_obs("col_a", "r1", "x1"), _obs("col_b", "r2", "y2")]
    row, _ = _run(obs, MappingError("boom"))
    assert {fv.reason for fv in row.values()} == {REASON_LLM_ERROR}
    assert all(fv.status is FieldStatus.NEEDS_REVIEW for fv in row.values())


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", "42"])
def test_non_object_output_goes_to_review(reply: str) -> None:
    row, _ = _run([_obs("col_a", "r1", "x1")], reply)
    assert row["col_a"].reason == REASON_LLM_INVALID


def test_malformed_answer_shape_goes_to_review() -> None:
    row, _ = _run(
        [_obs("col_a", "r1", "x1")], {"col_a": {"region_ids": "r1", "value": "x1"}}
    )
    assert row["col_a"].reason == REASON_LLM_INVALID


def test_unassigned_candidates_force_review_not_not_found() -> None:
    # Two people, value on a shared page: the planner cannot place it.
    owners = [
        PageOwner(page_index=0, person_index=0),
        PageOwner(page_index=1, person_index=1),
        PageOwner(page_index=2),
    ]
    stray = _obs("col_a", "r9", "x9").model_copy(update={"page_index": 2})
    plan = plan_rows(owners, [stray], TEMPLATE)
    rows = LlmFieldMapper(FakeClient({})).map_plan(plan, TEMPLATE)
    for row in rows:
        assert row["col_a"].status is FieldStatus.NEEDS_REVIEW
        assert (row["col_a"].reason or "").startswith(REASON_UNASSIGNED)


def test_prompt_and_schema_carry_ids_and_constrain_citations() -> None:
    obs = [_obs("col_a", "r1", "x1"), _obs("col_a", "r2", "x2")]
    _, client = _run(obs, {"col_a": _ans(["r1"], "x1")})
    messages, schema = client.calls[0]
    body = json.loads(messages[1]["content"])
    assert [c["region_id"] for c in body["columns"][0]["candidates"]] == ["r1", "r2"]
    col = schema["properties"]["col_a"]  # type: ignore[index]
    assert col["properties"]["region_ids"]["items"]["enum"] == ["r1", "r2"]
    assert schema["additionalProperties"] is False


def test_mapping_is_deterministic() -> None:
    obs = [_obs("col_a", "r1", "x1"), _obs("col_a", "r2", "x2")]
    a, _ = _run(obs, {"col_a": _ans(["r1"], "x1")})
    b, _ = _run(list(reversed(obs)), {"col_a": _ans(["r1"], "x1")})
    assert a == b


# ---- client -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:8000",
        "http://10.1.2.3:8000",
        "http://192.168.0.5",
        "http://gpu-box.internal",
    ],
)
def test_local_urls_allowed(url: str) -> None:
    assert_local_url(url)


@pytest.mark.parametrize(
    "url",
    ["https://api.example.com", "http://8.8.8.8", "http://172.217.0.1", "http://"],
)
def test_public_urls_refused(url: str) -> None:
    with pytest.raises(MappingError):
        assert_local_url(url)


def test_client_requires_model_and_local_host() -> None:
    with pytest.raises(MappingError):
        LocalHttpLlmClient("http://localhost:1", MappingConfig())
    with pytest.raises(MappingError):
        LocalHttpLlmClient("https://example.com", MappingConfig(model="m"))


def test_default_client_is_none_when_disabled_or_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORUS_LLM_URL", "http://localhost:1")
    assert build_default_llm_client(MappingConfig(enabled=False, model="m")) is None
    monkeypatch.delenv("TORUS_LLM_URL")
    assert build_default_llm_client(MappingConfig(enabled=True, model="m")) is None
    monkeypatch.setenv("TORUS_LLM_URL", "http://localhost:1")
    client = build_default_llm_client(MappingConfig(enabled=True, model="m"))
    assert client is not None and client.model_id == "m"


def test_temperature_is_pinned_to_zero() -> None:
    with pytest.raises(ValueError):
        MappingConfig(temperature=0.7)


def test_parse_and_clean_responses() -> None:
    ollama = json.dumps(
        {"message": {"content": '<think>hmm</think>```json\n{"a": 1}\n```'}}
    )
    assert json.loads(clean_json_text(parse_chat_content(ollama))) == {"a": 1}
    openai = json.dumps({"choices": [{"message": {"content": "{}"}}]})
    assert parse_chat_content(openai) == "{}"
    with pytest.raises(MappingError):
        parse_chat_content("nope")
    with pytest.raises(MappingError):
        parse_chat_content("{}")
