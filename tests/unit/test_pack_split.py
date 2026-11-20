"""Unit tests for pack splitting (P3.3). No models required."""

from __future__ import annotations

import pytest

from torus.classify import LogicalDocument, split_document, split_pack
from torus.config import load_config
from torus.config.schemas import (
    BlankPolicy,
    PackSplitConfig,
    PageTypesConfig,
    PageTypeSpec,
)
from torus.core import ClassifyError, Document, Page


def _types_with_split() -> PageTypesConfig:
    return PageTypesConfig(
        page_types=[
            PageTypeSpec(id="application_form", description="App first page."),
            PageTypeSpec(id="form_continuation", description="App later page."),
            PageTypeSpec(id="id_card", description="ID copy."),
            PageTypeSpec(id="bank_statement", description="Statement."),
            PageTypeSpec(id="checklist", description="Checklist."),
            PageTypeSpec(id="blank", description="Blank."),
            PageTypeSpec(id="other", description="Other."),
        ],
        pack_split=PackSplitConfig(
            continuations={
                "form_continuation": ["application_form", "form_continuation"]
            },
            merge_same=["id_card", "bank_statement", "checklist"],
            blank_policy=BlankPolicy.ATTACH_PREVIOUS,
            blank_page_type="blank",
        ),
    )


def _page(index: int, page_type: str, *, blank: bool = False) -> Page:
    return Page(
        index=index,
        width=100,
        height=140,
        page_type=page_type,
        blank=blank,
    )


def test_empty_pack() -> None:
    assert split_pack([], _types_with_split()) == []


def test_single_page() -> None:
    groups = split_pack([_page(0, "checklist")], _types_with_split())
    assert len(groups) == 1
    assert groups[0] == LogicalDocument(
        index=0,
        page_indices=[0],
        page_types=["checklist"],
        primary_page_type="checklist",
    )


def test_application_plus_continuation_one_group() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "form_continuation"),
        _page(2, "form_continuation"),
    ]
    groups = split_pack(pages, _types_with_split())
    assert len(groups) == 1
    assert groups[0].page_indices == [0, 1, 2]
    assert groups[0].primary_page_type == "application_form"


def test_type_change_starts_new_group() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "form_continuation"),
        _page(2, "id_card"),
        _page(3, "checklist"),
    ]
    groups = split_pack(pages, _types_with_split())
    assert [g.page_indices for g in groups] == [[0, 1], [2], [3]]
    assert [g.primary_page_type for g in groups] == [
        "application_form",
        "id_card",
        "checklist",
    ]


def test_merge_same_for_multi_page_statement() -> None:
    pages = [
        _page(0, "bank_statement"),
        _page(1, "bank_statement"),
        _page(2, "bank_statement"),
    ]
    groups = split_pack(pages, _types_with_split())
    assert len(groups) == 1
    assert groups[0].page_indices == [0, 1, 2]


def test_consecutive_application_forms_do_not_merge() -> None:
    """Two first pages without continuation are two logical documents."""
    pages = [
        _page(0, "application_form"),
        _page(1, "application_form"),
    ]
    groups = split_pack(pages, _types_with_split())
    assert [g.page_indices for g in groups] == [[0], [1]]


def test_blank_attaches_to_previous() -> None:
    pages = [
        _page(0, "id_card"),
        _page(1, "blank", blank=True),
        _page(2, "id_card"),
    ]
    groups = split_pack(pages, _types_with_split())
    # blank attaches to first id_card; second id_card merges via merge_same
    # after looking past the blank's non-blank prior.
    assert len(groups) == 1
    assert groups[0].page_indices == [0, 1, 2]
    assert groups[0].primary_page_type == "id_card"


def test_leading_blank_becomes_own_group_then_content() -> None:
    pages = [
        _page(0, "blank", blank=True),
        _page(1, "checklist"),
    ]
    groups = split_pack(pages, _types_with_split())
    assert [g.page_indices for g in groups] == [[0], [1]]
    assert groups[0].primary_page_type == "blank"


def test_blank_skip_policy() -> None:
    cfg = _types_with_split()
    rules = PackSplitConfig(
        continuations=dict(cfg.pack_split.continuations),
        merge_same=list(cfg.pack_split.merge_same),
        blank_policy=BlankPolicy.SKIP,
        blank_page_type="blank",
    )
    pages = [
        _page(0, "checklist"),
        _page(1, "blank", blank=True),
        _page(2, "id_card"),
    ]
    groups = split_pack(pages, cfg, pack_split=rules)
    assert [g.page_indices for g in groups] == [[0], [2]]


def test_blank_own_group_policy() -> None:
    cfg = _types_with_split()
    rules = PackSplitConfig(
        continuations=dict(cfg.pack_split.continuations),
        merge_same=list(cfg.pack_split.merge_same),
        blank_policy=BlankPolicy.OWN_GROUP,
        blank_page_type="blank",
    )
    pages = [
        _page(0, "checklist"),
        _page(1, "blank", blank=True),
        _page(2, "id_card"),
    ]
    groups = split_pack(pages, cfg, pack_split=rules)
    assert [g.page_indices for g in groups] == [[0], [1], [2]]


def test_orphan_continuation_starts_group() -> None:
    pages = [_page(0, "form_continuation"), _page(1, "form_continuation")]
    groups = split_pack(pages, _types_with_split())
    # Second continuation may follow first via continuations rule.
    assert len(groups) == 1
    assert groups[0].primary_page_type == "form_continuation"


def test_missing_page_type_raises() -> None:
    page = Page(index=0, width=10, height=10, page_type=None)
    with pytest.raises(ClassifyError, match="page_type"):
        split_pack([page], _types_with_split())


def test_split_document_wrapper() -> None:
    doc = Document(
        id="abc",
        source_path="synth.pdf",
        pages=[
            _page(0, "application_form"),
            _page(1, "form_continuation"),
            _page(2, "other"),
        ],
    )
    groups = split_document(doc, _types_with_split())
    assert [g.page_indices for g in groups] == [[0, 1], [2]]


def test_repo_pack_split_config_loads() -> None:
    cfg = load_config()
    rules = cfg.page_types.pack_split
    assert rules.blank_policy is BlankPolicy.ATTACH_PREVIOUS
    assert "form_continuation" in rules.continuations
    assert "application_form" in rules.continuations["form_continuation"]
    assert "bank_statement" in rules.merge_same
    assert "application_form" not in rules.merge_same


def test_invalid_pack_split_unknown_type() -> None:
    with pytest.raises(ValueError, match="merge_same"):
        PageTypesConfig(
            page_types=[
                PageTypeSpec(id="other", description="x"),
                PageTypeSpec(id="blank", description="b"),
            ],
            pack_split=PackSplitConfig(merge_same=["not_a_type"]),
        )
