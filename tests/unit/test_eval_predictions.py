"""Unit tests for prediction JSON loading (P1.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torus.core.errors import EvalError
from torus.eval.predictions import (
    load_prediction_rows,
    parse_prediction_payload,
    resolve_prediction_path,
)


def test_parse_rows_object_and_list() -> None:
    rows = parse_prediction_payload(
        {"rows": [{"person_name": "Synthetic A", "mobile_no": "1"}]},
        case_id="c1",
    )
    assert rows == [{"person_name": "Synthetic A", "mobile_no": "1"}]
    bare = parse_prediction_payload(
        [{"person_name": "Synthetic B"}],
        case_id="c1",
    )
    assert bare == [{"person_name": "Synthetic B"}]


def test_parse_strips_underscore_keys() -> None:
    rows = parse_prediction_payload(
        {"rows": [{"_excel_row": 2, "person_name": "X"}]},
        case_id="c1",
    )
    assert rows == [{"person_name": "X"}]


def test_parse_rejects_nested_cell() -> None:
    with pytest.raises(EvalError, match="JSON scalar"):
        parse_prediction_payload(
            {"rows": [{"person_name": {"nested": True}}]},
            case_id="c1",
        )


def test_resolve_prediction_path_order(tmp_path: Path) -> None:
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "case_a_digit_crop.json").write_text('{"rows":[]}\n', encoding="utf-8")
    (pred_dir / "case_a.json").write_text('{"rows":[]}\n', encoding="utf-8")
    nested = pred_dir / "case_b"
    nested.mkdir()
    (nested / "pred.json").write_text('{"rows":[]}\n', encoding="utf-8")

    assert (
        resolve_prediction_path(pred_dir, "case_a", variant="digit_crop")
        == pred_dir / "case_a_digit_crop.json"
    )
    assert resolve_prediction_path(pred_dir, "case_a") == pred_dir / "case_a.json"
    assert resolve_prediction_path(pred_dir, "case_b") == nested / "pred.json"
    assert resolve_prediction_path(pred_dir, "missing") is None


def test_load_prediction_rows(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps({"case_id": "c1", "rows": [{"mobile_no": "9"}]}) + "\n",
        encoding="utf-8",
    )
    assert load_prediction_rows(path, case_id="c1") == [{"mobile_no": "9"}]
