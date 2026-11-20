"""P12.1 / P12.2: onboard a new template and a new script with config only.

Everything here is synthetic. The point is the *absence* of pipeline changes:
a client workbook's headers become a template YAML (S1), the existing run
path fills it, and a non-Latin label script flows through the same code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook, load_workbook
from tests.fixtures.synth import generate_label_anchor_page
from tests.fixtures.synth.types import SynthPage

from torus.classify.factory import DisabledPageClassifier
from torus.config import load_config
from torus.config.loader import default_config_dir
from torus.config.schemas import TemplateConfig, TorusConfig
from torus.consensus.calibrate import (
    FEATURE_NAMES,
    Calibrator,
    LogisticModel,
    reader_fingerprint,
)
from torus.core.models import ReadHint, Reading, Region
from torus.layout.detect import LayoutResult
from torus.output.excel import write_blank_template, write_workbook
from torus.output.template import sync_template
from torus.readers.script import guess_script
from torus.service.run import RunContext, run_files

# Built from code points so the source holds no real-language text.
_DEV = {
    "doc_ref": "".join(chr(c) for c in (0x0915, 0x094D, 0x0930, 0x092E)),
    "doc_date": "".join(chr(c) for c in (0x0924, 0x093F, 0x0925, 0x093F)),
}

# (key, header, type, synonyms, sample value). Generic placeholders only.
_NEW_COLUMNS = (
    ("doc_ref", "Doc Ref", "text", ("Reference", "Ref No"), "AB12CD"),
    ("doc_date", "Doc Date", "date", ("Date", "Dated"), "01-01-2000"),
    ("total_amount", "Total", "amount", ("Amount", "Grand Total"), "1250.00"),
)


class _TruthReaders:
    """Reads back the generator's truth text for the best-overlapping region."""

    def __init__(self, page: SynthPage) -> None:
        self._page = page
        self._names = {"fake_a": "fam_a", "fake_b": "fam_b"}

    def names(self) -> list[str]:
        return list(self._names)

    def family(self, name: str) -> str:
        return self._names[name]

    def version(self, name: str) -> str:
        return "1"

    def prints_text(self, name: str) -> bool:
        return True

    def read(self, name: str, image: object, region: Region, hint: ReadHint) -> Reading:
        def iou(a: Region, b: Region) -> float:
            iw = max(0.0, min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0))
            ih = max(0.0, min(a.bbox.y1, b.bbox.y1) - max(a.bbox.y0, b.bbox.y0))
            area = lambda r: (r.bbox.x1 - r.bbox.x0) * (r.bbox.y1 - r.bbox.y0)  # noqa: E731
            union = area(a) + area(b) - iw * ih
            return iw * ih / union if union else 0.0

        best = max(self._page.regions, key=lambda t: iou(region, t.region))
        ok = iou(region, best.region) >= 0.5
        return Reading(
            region_id=region.id,
            reader_name=name,
            reader_version="1",
            text=(best.text or "") if ok else "",
            raw_score=0.99 if ok else None,
        )


def _calibrator(readers: _TruthReaders) -> Calibrator:
    n = len(FEATURE_NAMES)
    model = LogisticModel(
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=[0.0] * n,
        intercept=10.0,  # synthetic stand-in for a fitted model
        n_samples=2,
        n_correct=1,
    )
    fp = reader_fingerprint((name, "1") for name in readers.names())
    return Calibrator(
        feature_names=list(FEATURE_NAMES), reader_fingerprint=fp, global_model=model
    )


def _onboard(tmp_path: Path, synonyms: dict[str, tuple[str, ...]]) -> TorusConfig:
    """Client headers -> `template sync` -> field-spec edits (S1) -> load_config."""
    config_dir = tmp_path / "config"
    shutil.copytree(default_config_dir(), config_dir)

    xlsx = tmp_path / "client.xlsx"
    wb = Workbook()
    wb.active.append([h for _, h, *_ in _NEW_COLUMNS])  # headers only, no data
    wb.save(xlsx)
    _, path, _ = sync_template(xlsx, "onboard_v1", config_dir=config_dir)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_header = {f["excel_header"]: f for f in data["fields"]}
    for key, header, ftype, _syn, _val in _NEW_COLUMNS:
        spec = by_header[header]
        spec.update(
            key=key,
            type=ftype,
            critical=True,
            match="normalised",
            synonyms=[header, *synonyms[key]],
        )
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(config_dir)


def _run_page(
    cfg: TorusConfig, page: SynthPage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, str], Path]:
    page_path = tmp_path / "in" / "page.png"
    page_path.parent.mkdir()
    page.image.save(page_path)
    monkeypatch.setattr(
        "torus.service.run.detect_layout",
        lambda p, c=None, **_: LayoutResult(
            page_index=0, regions=[t.region for t in page.regions], sources=("truth",)
        ),
    )
    template: TemplateConfig = cfg.template("onboard_v1")
    readers = _TruthReaders(page)
    ctx = RunContext(
        config=cfg,
        template=template,
        readers=readers,
        classifier=DisabledPageClassifier("other"),
        calibrator=_calibrator(readers),
    )
    report = run_files([page_path], ctx)
    assert report.files_ok == 1 and report.files_failed == 0
    out = tmp_path / "out.xlsx"
    blank = write_blank_template(template, tmp_path / "blank.xlsx")
    write_workbook(blank, out, template, report.results, cfg.pipeline.output)
    row = report.results[0].rows[0]
    return {k: v.value for k, v in row.items() if v.value}, out


def test_p12_1_new_template_is_config_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _onboard(tmp_path, {k: s for k, _, _, s, _ in _NEW_COLUMNS})
    template = cfg.template("onboard_v1")
    assert [f.excel_header for f in template.fields] == [h for _, h, *_ in _NEW_COLUMNS]

    defs = tuple((k, h, s, v) for k, h, _t, s, v in _NEW_COLUMNS)
    page = generate_label_anchor_page(seed=5, field_count=3, field_defs=defs)
    values, out = _run_page(cfg, page, tmp_path, monkeypatch)

    expected = {k: v for k, _h, _t, _s, v in _NEW_COLUMNS}
    assert values == expected
    ws = load_workbook(out).worksheets[0]
    assert [c.value for c in ws[1]][:3] == [h for _, h, *_ in _NEW_COLUMNS]
    assert {c.value for c in ws[2] if c.value} == set(expected.values())


def test_p12_2_devanagari_labels_use_the_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syn = {k: s for k, _, _, s, _ in _NEW_COLUMNS}
    syn["doc_ref"] = (*syn["doc_ref"], _DEV["doc_ref"])
    syn["doc_date"] = (*syn["doc_date"], _DEV["doc_date"])
    cfg = _onboard(tmp_path, syn)

    defs = (
        ("doc_ref", "Doc Ref", (_DEV["doc_ref"],), "AB12CD"),
        ("doc_date", "Doc Date", (_DEV["doc_date"],), "01-01-2000"),
    )
    try:
        page = generate_label_anchor_page(
            seed=2, field_count=2, field_defs=defs, script="devanagari"
        )
    except LookupError:
        pytest.skip("no Devanagari font installed")

    labels = [t.text for t in page.regions if t.region.id.startswith("label")]
    assert any(guess_script(text or "").script == "devanagari" for text in labels)
    values, _ = _run_page(cfg, page, tmp_path, monkeypatch)
    assert set(values) <= {"doc_ref", "doc_date"}
    assert values.get("doc_ref") in (None, "AB12CD")


def test_p12_2_enabling_a_script_is_a_readers_yaml_change() -> None:
    readers = load_config().readers.readers
    supported = {s for spec in readers for s in spec.capabilities.scripts}
    assert "devanagari" in supported
