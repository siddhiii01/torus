"""Unit tests for multi-person page assignment (P3.4). No models required."""

from __future__ import annotations

import pytest

from torus.classify import (
    assign_document_persons,
    assign_persons,
    detect_role_cue,
)
from torus.config import load_config
from torus.config.schemas import (
    BlankPolicy,
    MultiPersonConfig,
    PackSplitConfig,
    PageTypesConfig,
    PageTypeSpec,
    PersonRoleSpec,
)
from torus.core import ClassifyError, Document, Page


def _roles() -> list[PersonRoleSpec]:
    return [
        PersonRoleSpec(
            id="borrower",
            text_cues=["primary applicant", "borrower", "applicant"],
        ),
        PersonRoleSpec(
            id="co_applicant",
            text_cues=["co-applicant", "co applicant", "coapplicant"],
        ),
        PersonRoleSpec(
            id="guarantor",
            text_cues=["guarantor", "surety"],
        ),
    ]


def _types_with_persons() -> PageTypesConfig:
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
        multi_person=MultiPersonConfig(
            roles=_roles(),
            default_role_order=["borrower", "co_applicant", "guarantor"],
            overflow_role="other_person",
            person_bearing_types=[
                "application_form",
                "form_continuation",
                "id_card",
                "bank_statement",
            ],
            shared_types=["checklist", "blank", "other"],
            person_boundary_types=["application_form"],
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


def test_detect_role_prefers_longer_cue() -> None:
    """``co applicant`` must beat bare ``applicant``."""
    hit = detect_role_cue("Form for co-applicant details", _roles())
    assert hit is not None
    assert hit[0] == "co_applicant"


def test_detect_role_borrower_cue() -> None:
    hit = detect_role_cue("Primary applicant declaration", _roles())
    assert hit is not None
    assert hit[0] == "borrower"


def test_detect_role_none_without_cues() -> None:
    assert detect_role_cue("account statement period", _roles()) is None


def test_single_person_structural_pack() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "form_continuation"),
        _page(2, "id_card"),
        _page(3, "checklist"),
    ]
    result = assign_persons(pages, _types_with_persons())
    assert result.person_count == 1
    assert [a.person_role for a in result.pages] == [
        "borrower",
        "borrower",
        "borrower",
        None,
    ]
    assert [a.source for a in result.pages] == [
        "structural",
        "inherited",
        "inherited",
        "shared",
    ]


def test_two_application_forms_structural_roles() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "id_card"),
        _page(2, "application_form"),
        _page(3, "id_card"),
        _page(4, "checklist"),
    ]
    result = assign_persons(pages, _types_with_persons())
    assert result.person_count == 2
    by_page = {a.page_index: a for a in result.pages}
    assert by_page[0].person_role == "borrower"
    assert by_page[0].person_index == 0
    assert by_page[1].person_role == "borrower"
    assert by_page[2].person_role == "co_applicant"
    assert by_page[2].source == "structural"
    assert by_page[2].person_index == 1
    assert by_page[3].person_role == "co_applicant"
    assert by_page[4].person_role is None


def test_cue_overrides_structural_order() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "id_card"),
        _page(2, "application_form"),
        _page(3, "id_card"),
    ]
    texts = {
        0: "Guarantor particulars",
        2: "Co-applicant form",
    }
    result = assign_persons(pages, _types_with_persons(), page_text_lookup=texts)
    by_page = {a.page_index: a for a in result.pages}
    assert by_page[0].person_role == "guarantor"
    assert by_page[0].source == "cue"
    assert by_page[1].person_role == "guarantor"
    assert by_page[1].source == "inherited"
    assert by_page[2].person_role == "co_applicant"
    assert by_page[2].source == "cue"
    assert by_page[3].person_role == "co_applicant"


def test_three_people_then_overflow_role() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "application_form"),
        _page(2, "application_form"),
        _page(3, "application_form"),
    ]
    result = assign_persons(pages, _types_with_persons())
    roles = [a.person_role for a in result.pages]
    assert roles == [
        "borrower",
        "co_applicant",
        "guarantor",
        "other_person",
    ]
    assert result.person_count == 4


def test_blank_and_shared_never_get_role() -> None:
    pages = [
        _page(0, "blank", blank=True),
        _page(1, "application_form"),
        _page(2, "other"),
    ]
    result = assign_persons(pages, _types_with_persons())
    assert result.pages[0].source == "shared"
    assert result.pages[0].person_role is None
    assert result.pages[1].person_role == "borrower"
    assert result.pages[2].person_role is None


def test_logical_document_summary() -> None:
    pages = [
        _page(0, "application_form"),
        _page(1, "form_continuation"),
        _page(2, "id_card"),
        _page(3, "checklist"),
    ]
    result = assign_persons(pages, _types_with_persons())
    assert len(result.logical_documents) >= 2
    app = result.logical_documents[0]
    assert app.primary_page_type == "application_form"
    assert app.person_role == "borrower"
    assert app.mixed_roles is False
    checklist = next(
        d for d in result.logical_documents if d.primary_page_type == "checklist"
    )
    assert checklist.person_role is None


def test_assign_document_persons_wrapper() -> None:
    doc = Document(
        id="abc",
        source_path="synth.pdf",
        pages=[
            _page(0, "application_form"),
            _page(1, "id_card"),
        ],
    )
    result = assign_document_persons(doc, _types_with_persons())
    assert result.person_count == 1
    assert result.pages[0].person_role == "borrower"


def test_missing_page_type_raises() -> None:
    page = Page(index=0, width=10, height=10, page_type=None)
    with pytest.raises(ClassifyError, match="page_type"):
        assign_persons([page], _types_with_persons())


def test_repo_multi_person_config_loads() -> None:
    cfg = load_config()
    mp = cfg.page_types.multi_person
    assert [r.id for r in mp.roles] == ["borrower", "co_applicant", "guarantor"]
    assert mp.default_role_order == ["borrower", "co_applicant", "guarantor"]
    assert "application_form" in mp.person_boundary_types
    assert "checklist" in mp.shared_types
    assert "id_card" in mp.person_bearing_types


def test_invalid_multi_person_unknown_type() -> None:
    with pytest.raises(ValueError, match="person_bearing_types"):
        PageTypesConfig(
            page_types=[
                PageTypeSpec(id="other", description="x"),
                PageTypeSpec(id="blank", description="b"),
            ],
            multi_person=MultiPersonConfig(
                roles=[PersonRoleSpec(id="borrower")],
                default_role_order=["borrower"],
                person_bearing_types=["application_form"],
                shared_types=["other", "blank"],
                person_boundary_types=["application_form"],
                blank_page_type="blank",
            ),
        )


def test_role_change_via_section_cue_on_continuation() -> None:
    """A co-applicant section cue on a later page switches the open person."""
    pages = [
        _page(0, "application_form"),
        _page(1, "form_continuation"),
        _page(2, "id_card"),
    ]
    texts = {
        0: "Borrower details",
        1: "Co-applicant particulars",
    }
    result = assign_persons(pages, _types_with_persons(), page_text_lookup=texts)
    by_page = {a.page_index: a for a in result.pages}
    assert by_page[0].person_role == "borrower"
    assert by_page[1].person_role == "co_applicant"
    assert by_page[1].source == "cue"
    assert by_page[2].person_role == "co_applicant"
