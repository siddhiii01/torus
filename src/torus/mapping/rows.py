"""Row planning (P8.2): how many rows a pack yields and who owns each value.

Row count and value placement follow the template's per-field ``cardinality``
(config, S1); nothing here knows any document, person or address. The planner
only *assigns*: it never chooses between conflicting values and never guesses
an owner. Anything it cannot place with certainty is returned in
``RowPlan.unassigned`` with a reason, so a later stage sends it to review
instead of writing it into the wrong person's row.

Inputs are mapping-local models so this stage does not import another stage's
internals; the pipeline adapts person detection (P3.4) and label anchoring
(P4) output into them.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from torus.config.schemas import Cardinality, TemplateConfig
from torus.core.errors import MappingError
from torus.core.models import BBox, ValidationResult
from torus.core.types import FieldStatus

UNASSIGNED_PASSTHROUGH = "passthrough_column"
UNASSIGNED_SHARED_PAGE = "shared_page_multi_person"
UNASSIGNED_AMBIGUOUS_ADDRESS = "ambiguous_address_group"


class PageOwner(BaseModel):
    """Which person slot a page belongs to; ``person_index=None`` means shared."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_index: int = Field(ge=0)
    person_index: int | None = Field(default=None, ge=0)
    person_role: str | None = None


class FieldObservation(BaseModel):
    """One located, voted value for a column, with its evidence region."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column_key: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    bbox: BBox
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: FieldStatus = Field(
        default=FieldStatus.NEEDS_REVIEW,
        description="Consensus status (P6.4); the mapper never raises it.",
    )
    anchored: bool = Field(
        default=False,
        description="True when a label anchor located this value (P4.2).",
    )
    validator_results: list[ValidationResult] = Field(
        default_factory=list,
        description="Validator outcomes for this value (P7); empty = not validated.",
    )
    address_group: str | None = Field(
        default=None,
        description=(
            "Opaque id of the address block this value sits in, scoped to its "
            "person. Set by layout; None when the block is unknown."
        ),
    )


class PlannedRow(BaseModel):
    """One output row: a person x address block, with the values assigned to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    person_index: int | None
    person_role: str | None
    address_group: str | None
    values: dict[str, list[FieldObservation]] = Field(default_factory=dict)


class Unassigned(BaseModel):
    """An observation the planner could not place with certainty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation: FieldObservation
    reason: str = Field(min_length=1)


class RowPlan(BaseModel):
    """Planned rows plus everything that could not be placed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[PlannedRow]
    unassigned: list[Unassigned] = Field(default_factory=list)


def _reading_order(obs: FieldObservation) -> tuple[int, float, float, str, str]:
    return (obs.page_index, obs.bbox.y0, obs.bbox.x0, obs.region_id, obs.column_key)


def plan_rows(
    owners: Sequence[PageOwner],
    observations: Sequence[FieldObservation],
    template: TemplateConfig,
) -> RowPlan:
    """Decide the rows of a pack and assign each observation to them."""
    owner_by_page: dict[int, PageOwner] = {}
    for owner in owners:
        if owner.page_index in owner_by_page:
            raise MappingError(f"page {owner.page_index} has more than one owner")
        owner_by_page[owner.page_index] = owner

    person_roles: dict[int, str | None] = {}
    for owner in sorted(owner_by_page.values(), key=lambda o: o.page_index):
        if owner.person_index is None:
            continue
        if person_roles.get(owner.person_index) is None:
            person_roles[owner.person_index] = owner.person_role
    persons: list[int | None] = [*sorted(person_roles)] if person_roles else [None]

    ordered = sorted(observations, key=_reading_order)

    # Pass 1: classify each observation to a target person (or unassigned).
    unassigned: list[Unassigned] = []
    pack_obs: list[FieldObservation] = []
    person_obs: dict[int | None, list[FieldObservation]] = {p: [] for p in persons}
    address_obs: dict[int | None, list[FieldObservation]] = {p: [] for p in persons}
    groups: dict[int | None, list[str]] = {p: [] for p in persons}

    for obs in ordered:
        spec = template.field(obs.column_key)
        if spec is None:
            raise MappingError(f"column {obs.column_key!r} is not in the template")
        if obs.page_index not in owner_by_page:
            raise MappingError(
                f"observation for {obs.column_key!r} is on page {obs.page_index}, "
                "which has no owner entry"
            )
        if spec.passthrough:
            unassigned.append(
                Unassigned(observation=obs, reason=UNASSIGNED_PASSTHROUGH)
            )
            continue
        if spec.cardinality is Cardinality.PER_PACK:
            pack_obs.append(obs)
            continue

        person = owner_by_page[obs.page_index].person_index
        if person is None:
            if len(persons) != 1:
                unassigned.append(
                    Unassigned(observation=obs, reason=UNASSIGNED_SHARED_PAGE)
                )
                continue
            # A single-person (or person-less) pack has only one possible owner.
            person = persons[0]

        if spec.cardinality is Cardinality.PER_PERSON:
            person_obs[person].append(obs)
            continue

        address_obs[person].append(obs)
        if obs.address_group is not None and obs.address_group not in groups[person]:
            groups[person].append(obs.address_group)

    # Pass 2: build rows, one per (person, address group).
    rows: list[PlannedRow] = []
    for person in persons:
        person_groups: list[str | None] = list(groups[person]) or [None]
        for group in person_groups:
            values: dict[str, list[FieldObservation]] = {}
            for obs in pack_obs + person_obs[person]:
                values.setdefault(obs.column_key, []).append(obs)
            for obs in address_obs[person]:
                if obs.address_group is None and group is not None:
                    continue
                if obs.address_group != group:
                    continue
                values.setdefault(obs.column_key, []).append(obs)
            rows.append(
                PlannedRow(
                    person_index=person,
                    person_role=person_roles.get(person)
                    if person is not None
                    else None,
                    address_group=group,
                    values={
                        k: sorted(v, key=_reading_order) for k, v in values.items()
                    },
                )
            )
        # Address values with no group cannot be tied to one of several blocks.
        if groups[person]:
            for obs in address_obs[person]:
                if obs.address_group is None:
                    unassigned.append(
                        Unassigned(observation=obs, reason=UNASSIGNED_AMBIGUOUS_ADDRESS)
                    )

    unassigned.sort(key=lambda u: _reading_order(u.observation))
    return RowPlan(rows=rows, unassigned=unassigned)
