"""Pack splitting: consecutive pages → logical documents (P3.3).

Grouping rules come from ``PageTypesConfig.pack_split``. No document-specific
page indices or names live here.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torus.config.schemas import BlankPolicy, PackSplitConfig, PageTypesConfig
from torus.core.errors import ClassifyError
from torus.core.models import Document, Page


class LogicalDocument(BaseModel):
    """One logical document inside a multi-document pack."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0, description="0-based order within the pack.")
    page_indices: list[int] = Field(min_length=1)
    page_types: list[str] = Field(min_length=1)
    primary_page_type: str = Field(
        min_length=1,
        description="First non-blank page type in the group (else blank type).",
    )

    @model_validator(mode="after")
    def _aligned_lengths(self) -> LogicalDocument:
        if len(self.page_indices) != len(self.page_types):
            msg = "page_indices and page_types must have the same length"
            raise ValueError(msg)
        return self


def split_pack(
    pages: Sequence[Page],
    page_types_config: PageTypesConfig,
    *,
    pack_split: PackSplitConfig | None = None,
) -> list[LogicalDocument]:
    """Group consecutive classified pages into logical documents.

    Pages must already have ``page_type`` set (run classification first).
    Raises ``ClassifyError`` if any page lacks a type.
    """
    rules = pack_split if pack_split is not None else page_types_config.pack_split
    blank_id = rules.blank_page_type
    merge_same = frozenset(rules.merge_same)

    groups: list[_OpenGroup] = []
    current: _OpenGroup | None = None

    for page in pages:
        page_type = _require_page_type(page)
        is_blank = page.blank or page_type == blank_id

        if is_blank:
            current = _handle_blank(
                page,
                page_type=page_type,
                current=current,
                groups=groups,
                policy=rules.blank_policy,
            )
            continue

        if current is not None and _can_join(
            current,
            page_type,
            merge_same=merge_same,
            continuations=rules.continuations,
            blank_id=blank_id,
        ):
            current.append(page.index, page_type)
            continue

        current = _OpenGroup(page.index, page_type)
        groups.append(current)

    return [
        LogicalDocument(
            index=i,
            page_indices=list(g.page_indices),
            page_types=list(g.page_types),
            primary_page_type=g.primary_page_type(blank_id),
        )
        for i, g in enumerate(groups)
    ]


def split_document(
    document: Document,
    page_types_config: PageTypesConfig,
    *,
    pack_split: PackSplitConfig | None = None,
) -> list[LogicalDocument]:
    """Split a classified ``Document`` into logical documents."""
    return split_pack(document.pages, page_types_config, pack_split=pack_split)


def _require_page_type(page: Page) -> str:
    if page.page_type is None or page.page_type == "":
        raise ClassifyError(
            f"page {page.index} has no page_type; classify before pack splitting"
        )
    return page.page_type


def _handle_blank(
    page: Page,
    *,
    page_type: str,
    current: _OpenGroup | None,
    groups: list[_OpenGroup],
    policy: BlankPolicy,
) -> _OpenGroup | None:
    if policy is BlankPolicy.SKIP:
        return current
    if policy is BlankPolicy.ATTACH_PREVIOUS and current is not None:
        current.append(page.index, page_type)
        return current
    # OWN_GROUP, or ATTACH_PREVIOUS with nothing open yet.
    new_group = _OpenGroup(page.index, page_type)
    groups.append(new_group)
    return new_group


def _can_join(
    current: _OpenGroup,
    page_type: str,
    *,
    merge_same: frozenset[str],
    continuations: dict[str, list[str]],
    blank_id: str,
) -> bool:
    last = current.last_non_blank_type(blank_id)
    if last is None:
        # Current group is blank-only; a content page starts a new group.
        return False
    allowed_priors = continuations.get(page_type)
    if allowed_priors is not None and last in allowed_priors:
        return True
    return page_type == last and page_type in merge_same


class _OpenGroup:
    """Mutable builder used only inside ``split_pack``."""

    __slots__ = ("page_indices", "page_types")

    def __init__(self, page_index: int, page_type: str) -> None:
        self.page_indices: list[int] = [page_index]
        self.page_types: list[str] = [page_type]

    def append(self, page_index: int, page_type: str) -> None:
        self.page_indices.append(page_index)
        self.page_types.append(page_type)

    def last_non_blank_type(self, blank_id: str) -> str | None:
        for page_type in reversed(self.page_types):
            if page_type != blank_id:
                return page_type
        return None

    def primary_page_type(self, blank_id: str) -> str:
        for page_type in self.page_types:
            if page_type != blank_id:
                return page_type
        return self.page_types[0]
