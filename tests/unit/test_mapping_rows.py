"""Row planning (P8.2): cardinality rules and person/address assignment.

Synthetic data only: abstract column keys, placeholder text.
"""

from __future__ import annotations

import pytest

from torus.config.schemas import Cardinality, FieldSpec, TemplateConfig
from torus.core.errors import MappingError
from torus.core.models import BBox
from torus.core.types import FieldType
from torus.mapping.rows import (
    UNASSIGNED_AMBIGUOUS_ADDRESS,
    UNASSIGNED_PASSTHROUGH,
    UNASSIGNED_SHARED_PAGE,
    FieldObservation,
    PageOwner,
    plan_rows,
)


def _spec(
    key: str, cardinality: Cardinality, *, passthrough: bool = False
) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=key,
        description=f"synthetic {key}",
        type=FieldType.TEXT,
        cardinality=cardinality,
        passthrough=passthrough,
    )


TEMPLATE = TemplateConfig(
    id="synth_t",
    fields=[
        _spec("pack_f", Cardinality.PER_PACK),
        _spec("person_f", Cardinality.PER_PERSON),
        _spec("addr_f", Cardinality.PER_ADDRESS),
        _spec("ops_f", Cardinality.PER_PACK, passthrough=True),
    ],
)

_BOX = BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2)


def _obs(
    column: str,
    page: int,
    text: str = "v",
    *,
    group: str | None = None,
    y0: float = 0.1,
    rid: str | None = None,
) -> FieldObservation:
    return FieldObservation(
        column_key=column,
        region_id=rid or f"r-{column}-{page}-{text}",
        page_index=page,
        bbox=BBox(x0=0.1, y0=y0, x1=0.3, y1=y0 + 0.05),
        text=text,
        confidence=0.9,
        address_group=group,
    )


OWNERS = [
    PageOwner(page_index=0, person_index=0, person_role="r0"),
    PageOwner(page_index=1, person_index=1, person_role="r1"),
    PageOwner(page_index=2),  # shared page
]


def test_one_row_per_person_when_no_addresses() -> None:
    plan = plan_rows(OWNERS, [], TEMPLATE)
    assert [r.person_index for r in plan.rows] == [0, 1]
    assert [r.person_role for r in plan.rows] == ["r0", "r1"]
    assert all(r.address_group is None for r in plan.rows)


def test_person_values_go_to_owning_person_only() -> None:
    plan = plan_rows(
        OWNERS, [_obs("person_f", 0, "a"), _obs("person_f", 1, "b")], TEMPLATE
    )
    assert [o.text for o in plan.rows[0].values["person_f"]] == ["a"]
    assert [o.text for o in plan.rows[1].values["person_f"]] == ["b"]
    assert plan.unassigned == []


def test_pack_values_replicate_to_every_row() -> None:
    plan = plan_rows(OWNERS, [_obs("pack_f", 2, "p")], TEMPLATE)
    for row in plan.rows:
        assert [o.text for o in row.values["pack_f"]] == ["p"]


def test_address_groups_split_one_person_into_rows() -> None:
    obs = [
        _obs("addr_f", 0, "h", group="g1", y0=0.1),
        _obs("addr_f", 0, "w", group="g2", y0=0.5),
        _obs("person_f", 0, "name"),
    ]
    plan = plan_rows(OWNERS, obs, TEMPLATE)
    p0 = [r for r in plan.rows if r.person_index == 0]
    assert [r.address_group for r in p0] == ["g1", "g2"]
    # Person-level value is shared by both of that person's address rows.
    assert all([o.text for o in r.values["person_f"]] == ["name"] for r in p0)
    assert [o.text for o in p0[0].values["addr_f"]] == ["h"]
    assert [o.text for o in p0[1].values["addr_f"]] == ["w"]
    # The other person is untouched: still one row.
    assert len([r for r in plan.rows if r.person_index == 1]) == 1


def test_address_group_order_is_reading_order_not_input_order() -> None:
    obs = [
        _obs("addr_f", 0, "later", group="gb", y0=0.6),
        _obs("addr_f", 0, "first", group="ga", y0=0.1),
    ]
    plan = plan_rows(OWNERS, obs, TEMPLATE)
    assert [r.address_group for r in plan.rows if r.person_index == 0] == ["ga", "gb"]


def test_same_group_label_for_different_people_stays_separate() -> None:
    obs = [
        _obs("addr_f", 0, "x", group="g"),
        _obs("addr_f", 1, "y", group="g"),
    ]
    plan = plan_rows(OWNERS, obs, TEMPLATE)
    assert [(r.person_index, r.address_group) for r in plan.rows] == [
        (0, "g"),
        (1, "g"),
    ]
    assert [o.text for o in plan.rows[1].values["addr_f"]] == ["y"]


def test_person_value_on_shared_page_in_multi_person_pack_is_unassigned() -> None:
    plan = plan_rows(OWNERS, [_obs("person_f", 2, "z")], TEMPLATE)
    assert [(u.observation.text, u.reason) for u in plan.unassigned] == [
        ("z", UNASSIGNED_SHARED_PAGE)
    ]
    assert all("person_f" not in r.values for r in plan.rows)


def test_person_value_on_shared_page_goes_to_sole_person() -> None:
    owners = [PageOwner(page_index=0, person_index=0), PageOwner(page_index=1)]
    plan = plan_rows(owners, [_obs("person_f", 1, "z")], TEMPLATE)
    assert plan.unassigned == []
    assert [o.text for o in plan.rows[0].values["person_f"]] == ["z"]


def test_missing_address_group_with_several_groups_is_ambiguous() -> None:
    obs = [
        _obs("addr_f", 0, "a", group="g1"),
        _obs("addr_f", 0, "b", group="g2"),
        _obs("addr_f", 0, "c", group=None),
    ]
    plan = plan_rows(OWNERS, obs, TEMPLATE)
    assert [(u.observation.text, u.reason) for u in plan.unassigned] == [
        ("c", UNASSIGNED_AMBIGUOUS_ADDRESS)
    ]


def test_missing_address_group_with_no_groups_uses_single_row() -> None:
    plan = plan_rows(OWNERS, [_obs("addr_f", 0, "a", group=None)], TEMPLATE)
    p0 = [r for r in plan.rows if r.person_index == 0]
    assert len(p0) == 1
    assert [o.text for o in p0[0].values["addr_f"]] == ["a"]
    assert plan.unassigned == []


def test_passthrough_column_is_never_assigned() -> None:
    plan = plan_rows(OWNERS, [_obs("ops_f", 0, "q")], TEMPLATE)
    assert [u.reason for u in plan.unassigned] == [UNASSIGNED_PASSTHROUGH]
    assert all("ops_f" not in r.values for r in plan.rows)


def test_unknown_column_raises() -> None:
    with pytest.raises(MappingError):
        plan_rows(OWNERS, [_obs("nope", 0)], TEMPLATE)


def test_observation_on_unknown_page_raises() -> None:
    with pytest.raises(MappingError):
        plan_rows(OWNERS, [_obs("person_f", 9)], TEMPLATE)


def test_duplicate_page_owner_raises() -> None:
    with pytest.raises(MappingError):
        plan_rows([PageOwner(page_index=0), PageOwner(page_index=0)], [], TEMPLATE)


def test_conflicting_values_are_all_kept_not_picked() -> None:
    obs = [_obs("person_f", 0, "a"), _obs("person_f", 0, "b", y0=0.4)]
    plan = plan_rows(OWNERS, obs, TEMPLATE)
    assert [o.text for o in plan.rows[0].values["person_f"]] == ["a", "b"]


def test_pack_with_no_people_yields_one_pack_row() -> None:
    owners = [PageOwner(page_index=0), PageOwner(page_index=1)]
    plan = plan_rows(owners, [_obs("pack_f", 0, "p")], TEMPLATE)
    assert len(plan.rows) == 1
    assert plan.rows[0].person_index is None
    assert [o.text for o in plan.rows[0].values["pack_f"]] == ["p"]


def test_planning_is_deterministic_under_input_shuffle() -> None:
    obs = [
        _obs("addr_f", 0, "h", group="g1", y0=0.1),
        _obs("addr_f", 1, "w", group="g2", y0=0.2),
        _obs("person_f", 1, "n"),
        _obs("pack_f", 2, "p"),
    ]
    a = plan_rows(OWNERS, obs, TEMPLATE)
    b = plan_rows(list(reversed(OWNERS)), list(reversed(obs)), TEMPLATE)
    assert a == b
