"""Multi-person detection: assign pages to roles (P3.4).

Uses config-driven role text cues and structural pack order. Never matches
customer names — only role phrases and page-type boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from torus.classify.pack import LogicalDocument, split_pack
from torus.classify.text_cues import normalize_text
from torus.config.schemas import MultiPersonConfig, PageTypesConfig, PersonRoleSpec
from torus.core.errors import ClassifyError
from torus.core.models import Document, Page

PageTextLookup = Mapping[int, str]

AssignmentSource = Literal["cue", "structural", "inherited", "shared"]


class PagePersonAssignment(BaseModel):
    """Person-role attribution for one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_index: int = Field(ge=0)
    page_type: str = Field(min_length=1)
    person_role: str | None = Field(
        default=None,
        description="Role id from config, or None for shared / blank pages.",
    )
    person_index: int | None = Field(
        default=None,
        ge=0,
        description="0-based person slot within the pack; None when shared.",
    )
    source: AssignmentSource
    matched_cue: str | None = Field(
        default=None,
        description="Normalised cue phrase that won, when source is cue.",
    )


class LogicalDocumentPersons(BaseModel):
    """Person summary for one logical document after pack split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    logical_index: int = Field(ge=0)
    page_indices: list[int] = Field(min_length=1)
    primary_page_type: str = Field(min_length=1)
    person_role: str | None = None
    person_index: int | None = Field(default=None, ge=0)
    mixed_roles: bool = Field(
        default=False,
        description="True when person-bearing pages in the group disagree on role.",
    )


class MultiPersonResult(BaseModel):
    """Full multi-person attribution for a pack."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: list[PagePersonAssignment]
    logical_documents: list[LogicalDocumentPersons]
    person_count: int = Field(ge=0)


def detect_role_cue(
    page_text: str,
    roles: Sequence[PersonRoleSpec],
) -> tuple[str, str] | None:
    """Return ``(role_id, matched_cue)`` for the most specific cue hit.

    Longer normalised cues win so ``co applicant`` beats bare ``applicant``.
    """
    haystack = normalize_text(page_text)
    if not haystack:
        return None
    best_len = -1
    best: tuple[str, str] | None = None
    for role in roles:
        for cue in role.text_cues:
            needle = normalize_text(cue)
            if not needle:
                continue
            if needle in haystack and len(needle) > best_len:
                best_len = len(needle)
                best = (role.id, needle)
    return best


def assign_persons(
    pages: Sequence[Page],
    page_types_config: PageTypesConfig,
    *,
    page_text_lookup: PageTextLookup | None = None,
    multi_person: MultiPersonConfig | None = None,
    logical_documents: Sequence[LogicalDocument] | None = None,
) -> MultiPersonResult:
    """Assign each classified page to a person role or mark it shared.

    Pages must already have ``page_type`` set. Optional ``page_text_lookup``
    supplies OCR/embedded text keyed by page index for role cues.
    """
    rules = multi_person if multi_person is not None else page_types_config.multi_person
    texts = page_text_lookup or {}
    bearing = frozenset(rules.person_bearing_types)
    shared = frozenset(rules.shared_types)
    boundaries = frozenset(rules.person_boundary_types)
    blank_id = rules.blank_page_type
    role_order = list(rules.default_role_order)
    role_to_slot = {role_id: i for i, role_id in enumerate(role_order)}

    assignments: list[PagePersonAssignment] = []
    current_role: str | None = None
    current_index: int | None = None
    current_had_boundary = False
    next_structural_slot = 0
    max_person_index = -1

    for page in pages:
        page_type = _require_page_type(page)
        is_blank = page.blank or page_type == blank_id

        if is_blank or page_type in shared or page_type not in bearing:
            assignments.append(
                PagePersonAssignment(
                    page_index=page.index,
                    page_type=page_type,
                    person_role=None,
                    person_index=None,
                    source="shared",
                )
            )
            continue

        page_text = texts.get(page.index, "")
        cue = detect_role_cue(page_text, rules.roles)
        is_boundary = page_type in boundaries

        if cue is not None:
            role_id, matched = cue
            start_new = (
                current_role is None
                or (is_boundary and current_had_boundary)
                or role_id != current_role
            )

            if start_new:
                # A second boundary page gets a fresh slot even if the cue
                # repeats a known role id (two people can share a role label).
                if is_boundary and current_had_boundary and current_role is not None:
                    slot = next_structural_slot
                else:
                    slot = _slot_for_cued_role(
                        role_id,
                        role_to_slot=role_to_slot,
                        next_structural_slot=next_structural_slot,
                        current_role=None,
                        current_index=None,
                    )
                current_role = role_id
                current_index = slot
                current_had_boundary = is_boundary
                next_structural_slot = max(next_structural_slot, slot + 1)
            elif is_boundary:
                current_had_boundary = True

            assignments.append(
                PagePersonAssignment(
                    page_index=page.index,
                    page_type=page_type,
                    person_role=current_role,
                    person_index=current_index,
                    source="cue",
                    matched_cue=matched,
                )
            )
            assert current_index is not None
            max_person_index = max(max_person_index, current_index)
            continue

        if is_boundary and (current_role is None or current_had_boundary):
            role_id, slot = _allocate_structural(
                role_order,
                overflow_role=rules.overflow_role,
                next_slot=next_structural_slot,
            )
            current_role = role_id
            current_index = slot
            current_had_boundary = True
            next_structural_slot = slot + 1
            assignments.append(
                PagePersonAssignment(
                    page_index=page.index,
                    page_type=page_type,
                    person_role=current_role,
                    person_index=current_index,
                    source="structural",
                )
            )
            max_person_index = max(max_person_index, slot)
            continue

        if current_role is not None and current_index is not None:
            if is_boundary:
                current_had_boundary = True
            assignments.append(
                PagePersonAssignment(
                    page_index=page.index,
                    page_type=page_type,
                    person_role=current_role,
                    person_index=current_index,
                    source="inherited",
                )
            )
            continue

        # First person-bearing page without cue and without an open person.
        role_id, slot = _allocate_structural(
            role_order,
            overflow_role=rules.overflow_role,
            next_slot=next_structural_slot,
        )
        current_role = role_id
        current_index = slot
        current_had_boundary = is_boundary
        next_structural_slot = slot + 1
        assignments.append(
            PagePersonAssignment(
                page_index=page.index,
                page_type=page_type,
                person_role=current_role,
                person_index=current_index,
                source="structural",
            )
        )
        max_person_index = max(max_person_index, slot)

    docs = (
        list(logical_documents)
        if logical_documents is not None
        else split_pack(pages, page_types_config)
    )
    logical = _summarise_logical_docs(docs, assignments)
    person_count = max_person_index + 1 if max_person_index >= 0 else 0
    return MultiPersonResult(
        pages=assignments,
        logical_documents=logical,
        person_count=person_count,
    )


def assign_document_persons(
    document: Document,
    page_types_config: PageTypesConfig,
    *,
    page_text_lookup: PageTextLookup | None = None,
    multi_person: MultiPersonConfig | None = None,
) -> MultiPersonResult:
    """Assign person roles for every page in a classified ``Document``."""
    return assign_persons(
        document.pages,
        page_types_config,
        page_text_lookup=page_text_lookup,
        multi_person=multi_person,
    )


def _require_page_type(page: Page) -> str:
    if page.page_type is None or page.page_type == "":
        raise ClassifyError(
            f"page {page.index} has no page_type; "
            "classify before multi-person assignment"
        )
    return page.page_type


def _allocate_structural(
    role_order: Sequence[str],
    *,
    overflow_role: str,
    next_slot: int,
) -> tuple[str, int]:
    if next_slot < len(role_order):
        return role_order[next_slot], next_slot
    return overflow_role, next_slot


def _slot_for_cued_role(
    role_id: str,
    *,
    role_to_slot: Mapping[str, int],
    next_structural_slot: int,
    current_role: str | None,
    current_index: int | None,
) -> int:
    if current_role == role_id and current_index is not None:
        return current_index
    if role_id in role_to_slot:
        return role_to_slot[role_id]
    return next_structural_slot


def _summarise_logical_docs(
    docs: Sequence[LogicalDocument],
    assignments: Sequence[PagePersonAssignment],
) -> list[LogicalDocumentPersons]:
    by_page = {a.page_index: a for a in assignments}
    out: list[LogicalDocumentPersons] = []
    for doc in docs:
        roles: list[tuple[str, int]] = []
        for page_index in doc.page_indices:
            assignment = by_page.get(page_index)
            if (
                assignment is not None
                and assignment.person_role is not None
                and assignment.person_index is not None
            ):
                roles.append((assignment.person_role, assignment.person_index))
        unique = list(dict.fromkeys(roles))
        mixed = len(unique) > 1
        if not unique:
            out.append(
                LogicalDocumentPersons(
                    logical_index=doc.index,
                    page_indices=list(doc.page_indices),
                    primary_page_type=doc.primary_page_type,
                    person_role=None,
                    person_index=None,
                    mixed_roles=False,
                )
            )
            continue
        role, index = unique[0]
        out.append(
            LogicalDocumentPersons(
                logical_index=doc.index,
                page_indices=list(doc.page_indices),
                primary_page_type=doc.primary_page_type,
                person_role=None if mixed else role,
                person_index=None if mixed else index,
                mixed_roles=mixed,
            )
        )
    return out
