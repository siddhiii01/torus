"""P12.3: a new document family (invoices) is config + synthetic examples only."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.synth import generate_label_anchor_page
from tests.integration.test_scale_out import _run_page

from torus.classify.factory import DisabledPageClassifier
from torus.config import load_config

_DEFS = (
    ("invoice_no", "Invoice No", ("Inv No", "Bill No"), "INV0042"),
    ("invoice_date", "Invoice Date", ("Date", "Bill Date"), "01-01-2000"),
    ("total_amount", "Total Amount", ("Grand Total", "Amount Due"), "1250.00"),
)


def test_invoice_page_type_and_template_are_consistent() -> None:
    cfg = load_config()
    ids = {p.id for p in cfg.page_types.page_types}
    assert "invoice" in ids
    template = cfg.template("invoice_v1")
    assert all(set(f.look_in) <= ids and f.look_in for f in template.fields)
    invoice = next(p for p in cfg.page_types.page_types if p.id == "invoice")
    assert invoice.visual_cues and invoice.text_cues and invoice.examples


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_three_synthetic_invoices_fill_the_template(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    page = generate_label_anchor_page(seed=seed, field_count=3, field_defs=_DEFS)
    # _run_page reads the template id "onboard_v1"; alias it to the real one.
    cfg = cfg.model_copy(
        update={"templates": {"onboard_v1": cfg.template("invoice_v1")}}
    )
    monkeypatch.setattr(
        "tests.integration.test_scale_out.DisabledPageClassifier",
        lambda *_: DisabledPageClassifier("invoice"),
    )
    values, _ = _run_page(cfg, page, tmp_path, monkeypatch)
    assert values == {k: v for k, _h, _s, v in _DEFS}


def test_fields_are_not_searched_on_other_page_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    page = generate_label_anchor_page(seed=1, field_count=3, field_defs=_DEFS)
    cfg = cfg.model_copy(
        update={"templates": {"onboard_v1": cfg.template("invoice_v1")}}
    )
    values, _ = _run_page(cfg, page, tmp_path, monkeypatch)  # classifier -> "other"
    assert values == {}
