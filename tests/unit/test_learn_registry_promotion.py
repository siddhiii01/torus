"""Registry (P10.3), promotion (P10.4) and active-learning order (P10.5). Synthetic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from torus.config import load_config
from torus.config.schemas import ReadersConfig, ReaderSpec, RegressionGates
from torus.core.models import BBox, Evidence
from torus.core.types import FieldStatus
from torus.eval.metrics import (
    CaseScore,
    FieldMetrics,
    MicroMetrics,
    RowAlignmentMetrics,
    StatusRates,
)
from torus.eval.report import EvalRunReport, RunVersions, TimingSummary
from torus.learn.active import (
    ActiveLearningConfig,
    column_correction_rates,
    load_active_config,
    prioritise,
)
from torus.learn.dataset import LearnError
from torus.learn.promotion import evaluate_promotion, promote
from torus.learn.registry import ModelRegistry, apply_registry
from torus.review.queue import ReadingView, ReviewItem
from torus.review.store import CorrectionRecord

GATES = RegressionGates(noise_margin=0.01, silent_error_tolerance=0)


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry")


def _weights(tmp_path: Path, name: str = "w.bin", data: bytes = b"abc") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _report(
    *,
    acc: float,
    auto: float,
    silent: int = 0,
    split: str | None = "hidden",
    cases: tuple[str, ...] = ("a", "b"),
) -> EvalRunReport:
    micro = MicroMetrics(
        tp=1,
        fp=0,
        fn=0,
        compared=10,
        correct=int(acc * 10),
        accuracy=acc,
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )
    fm = FieldMetrics(
        key="col",
        tp=1,
        fp=0,
        fn=0,
        compared=10,
        correct=int(acc * 10),
        accuracy=acc,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        silent_errors=silent,
    )
    score = CaseScore(
        template_id="t",
        rows=RowAlignmentMetrics(
            gold=1,
            pred=1,
            aligned=1,
            unmatched_gold=0,
            unmatched_pred=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
        ),
        fields_micro=micro,
        per_field={"col": fm},
        silent_errors=silent,
        status_rates=StatusRates(
            total=10, counts={}, rates={}, automatic_rate=auto, review_rate=0.1
        ),
    )
    return EvalRunReport(
        set_id="s",
        split=split,
        created_at="2026-01-01T00:00:00Z",
        case_count=len(cases),
        case_ids=list(cases),
        scores=score,
        timings=TimingSummary(total_ms=0.0, case_count=len(cases)),
        versions=RunVersions(config_hash="h", template_id="t"),
    )


# --- registry --------------------------------------------------------------


def test_register_versions_and_verify(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    kw = {"base_model": "base", "recipe": "r", "data_hash": "h"}
    e1 = reg.register("char_cell", _weights(tmp_path), metrics={"acc": 0.9}, **kw)
    e2 = reg.register("char_cell", _weights(tmp_path, data=b"xyz"), **kw)
    assert (e1.version, e2.version) == ("1", "2")
    assert reg.versions("char_cell") == ["1", "2"]
    assert reg.verify("char_cell", "1")
    reg.weights_path("char_cell", "1").write_bytes(b"tampered")
    assert not reg.verify("char_cell", "1")
    with pytest.raises(LearnError):
        reg.set_current("char_cell", "1")


def test_registry_rejects_unsafe_names(tmp_path: Path) -> None:
    with pytest.raises(LearnError):
        _registry(tmp_path).versions("../escape")


def test_apply_registry_resolves_current(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.register(
        "char_cell", _weights(tmp_path), base_model="b", recipe="r", data_hash="h"
    )
    reg.set_current("char_cell", "1")
    spec = ReaderSpec(
        name="char_cell",
        family="char_cell",
        enabled=True,
        min_vram_gb=0,
        min_ram_gb=1,
        registry_version="current",
        model="old",
        version="0",
    )
    out = apply_registry(ReadersConfig(readers=[spec]), reg).readers[0]
    assert out.version == "1" and out.model != "old"


def test_apply_registry_enabled_missing_fails_disabled_skips(tmp_path: Path) -> None:
    def spec(enabled: bool) -> ReadersConfig:
        return ReadersConfig(
            readers=[
                ReaderSpec(
                    name="r",
                    family="f",
                    enabled=enabled,
                    min_vram_gb=0,
                    min_ram_gb=0,
                    registry_version="current",
                )
            ]
        )

    with pytest.raises(LearnError):
        apply_registry(spec(True), _registry(tmp_path))
    assert apply_registry(spec(False), _registry(tmp_path)).readers[0].model is None


# --- promotion -------------------------------------------------------------


def test_promotion_rule() -> None:
    base = _report(acc=0.8, auto=0.5)
    ok, why = evaluate_promotion(base, _report(acc=0.85, auto=0.6), gates=GATES)
    assert ok and not why
    for cand in (
        _report(acc=0.85, auto=0.6, silent=1),
        _report(acc=0.8, auto=0.5),
        _report(acc=0.9, auto=0.4),
        _report(acc=0.85, auto=0.6, split="dev"),
        _report(acc=0.85, auto=0.6, cases=("a", "c")),
    ):
        assert not evaluate_promotion(base, cand, gates=GATES)[0]


def test_promote_moves_pointer_and_rolls_back(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    kw = {"base_model": "b", "recipe": "r", "data_hash": "h"}
    reg.register("rd", _weights(tmp_path), **kw)
    reg.register("rd", _weights(tmp_path, data=b"new"), **kw)
    base, good = _report(acc=0.8, auto=0.5), _report(acc=0.9, auto=0.6)

    bad = promote(reg, "rd", "2", base, _report(acc=0.8, auto=0.5), gates=GATES)
    assert not bad.promoted and reg.current("rd") is None

    def boom() -> None:
        raise RuntimeError

    with pytest.raises(LearnError):
        promote(reg, "rd", "2", base, good, gates=GATES, recalibrate=boom)
    assert reg.current("rd") is None

    done = promote(reg, "rd", "2", base, good, gates=GATES)
    cur = reg.current("rd")
    assert done.promoted and done.calibration_pending
    assert cur is not None and cur.version == "2"
    hooked = promote(reg, "rd", "1", base, good, gates=GATES, recalibrate=lambda: None)
    assert not hooked.calibration_pending


# --- active learning -------------------------------------------------------

BOX = BBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2)


def _item(i: int, col: str, conf: float, texts: tuple[str, ...] = ()) -> ReviewItem:
    return ReviewItem(
        item_id=f"{i}:{col}",
        document_id="d",
        row=i,
        column_key=col,
        status=FieldStatus.NEEDS_REVIEW,
        confidence=conf,
        evidence=[Evidence(region_id="r", page_index=0, bbox=BOX)],
        readings=[
            ReadingView(region_id="r", reader_name=f"x{k}", text=t)
            for k, t in enumerate(texts)
        ],
    )


def _rec(col: str, action: str) -> CorrectionRecord:
    return CorrectionRecord(
        document_id="d",
        item_id=f"0:{col}",
        row=0,
        column_key=col,
        action=action,  # type: ignore[arg-type]
        value="v",
        status_before=FieldStatus.NEEDS_REVIEW,
        reviewer="t",
        decided_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_prioritise_orders_by_uncertainty_and_keeps_all() -> None:
    items = [_item(0, "a", 0.9), _item(1, "a", 0.2), _item(2, "b", 0.5)]
    out = prioritise(items, {}, ActiveLearningConfig(column_decay=0.0))
    assert [i.item_id for i in out] == ["1:a", "2:b", "0:a"]
    assert sorted(i.item_id for i in out) == sorted(i.item_id for i in items)


def test_disagreement_and_column_rate_boost() -> None:
    items = [_item(0, "a", 0.5, ("X", "X")), _item(1, "b", 0.5, ("X", "Y"))]
    assert prioritise(items, {})[0].item_id == "1:b"
    rates = column_correction_rates(
        [_rec("a", "correct")] * 4 + [_rec("b", "accept")] * 4
    )
    assert rates["a"] > rates["b"]
    same = [_item(0, "a", 0.5), _item(1, "b", 0.5)]
    assert prioritise(same, rates)[0].column_key == "a"


def test_column_decay_interleaves_columns() -> None:
    items = [_item(i, "a", 0.5) for i in range(3)] + [_item(9, "b", 0.45)]
    cols = [
        i.column_key
        for i in prioritise(items, {}, ActiveLearningConfig(column_decay=1.0))
    ]
    assert cols[:2] != ["a", "a"]


def test_shipped_active_config_loads() -> None:
    root = Path(__file__).resolve().parents[2] / "config"
    assert load_active_config(root / "active_learning.yaml").uncertainty_weight > 0
    assert load_config() is not None
