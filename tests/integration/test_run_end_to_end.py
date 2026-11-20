"""P8.6: synthetic page -> pipeline -> Excel (no real data, fake readers)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import load_workbook
from tests.fixtures.synth import generate_label_anchor_page
from tests.fixtures.synth.types import SynthPage

from torus.classify.factory import DisabledPageClassifier
from torus.cli import main
from torus.config import load_config
from torus.config.schemas import FieldSpec, TemplateConfig, TorusConfig
from torus.consensus.calibrate import (
    FEATURE_NAMES,
    Calibrator,
    LogisticModel,
    reader_fingerprint,
)
from torus.core.models import ReadHint, Reading, Region
from torus.core.types import FieldType
from torus.layout.detect import LayoutResult
from torus.output.excel import write_blank_template, write_workbook
from torus.service.run import RunContext, collect_inputs, run_files


def _iou(a: Region, b: Region) -> float:
    iw = max(0.0, min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0))
    ih = max(0.0, min(a.bbox.y1, b.bbox.y1) - max(a.bbox.y0, b.bbox.y0))
    inter = iw * ih
    area_a = (a.bbox.x1 - a.bbox.x0) * (a.bbox.y1 - a.bbox.y0)
    area_b = (b.bbox.x1 - b.bbox.x0) * (b.bbox.y1 - b.bbox.y0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


class FakeReaders:
    """Returns the generator's truth text for the best-overlapping region."""

    def __init__(self, page: SynthPage, names: dict[str, str]) -> None:
        self._page = page
        self._names = names  # reader name -> family

    def names(self) -> list[str]:
        return list(self._names)

    def family(self, name: str) -> str:
        return self._names[name]

    def version(self, name: str) -> str:
        return "1"

    def prints_text(self, name: str) -> bool:
        return True

    def read(self, name: str, image: object, region: Region, hint: ReadHint) -> Reading:
        best = max(self._page.regions, key=lambda t: _iou(region, t.region))
        ok = _iou(region, best.region) >= 0.5
        return Reading(
            region_id=region.id,
            reader_name=name,
            reader_version="1",
            text=(best.text or "") if ok else "",
            raw_score=0.99 if ok else None,
        )


def _spec(key: str, header: str, col: int, syn: list[str]) -> FieldSpec:
    return FieldSpec(
        key=key,
        excel_header=header,
        excel_col=col,
        description=header,
        type=FieldType.TEXT,
        critical=True,
        synonyms=syn,
    )


def _template() -> TemplateConfig:
    return TemplateConfig(
        id="synth_t",
        fields=[
            _spec("person_name", "Name", 1, ["Name", "Full Name", "Applicant Name"]),
            _spec("mobile_no", "Mobile No", 2, ["Mobile", "Mob. No.", "Cell"]),
            _spec("address_pincode", "PIN", 3, ["PIN Code", "Pincode", "Zip"]),
        ],
    )


def _config(template: TemplateConfig) -> TorusConfig:
    return load_config().model_copy(update={"templates": {template.id: template}})


def _calibrator(readers: FakeReaders) -> Calibrator:
    n = len(FEATURE_NAMES)
    model = LogisticModel(
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=[0.0] * n,
        intercept=10.0,  # near-certain: synthetic stand-in for a fitted model
        n_samples=2,
        n_correct=1,
    )
    fp = reader_fingerprint((name, "1") for name in readers.names())
    return Calibrator(
        feature_names=list(FEATURE_NAMES), reader_fingerprint=fp, global_model=model
    )


@pytest.fixture
def page_png(tmp_path: Path) -> tuple[Path, SynthPage]:
    page = generate_label_anchor_page(seed=3, field_count=3)
    path = tmp_path / "in" / "page.png"
    path.parent.mkdir()
    page.image.save(path)
    return path, page


@pytest.fixture(autouse=True)
def _truth_layout(
    monkeypatch: pytest.MonkeyPatch, page_png: tuple[Path, SynthPage]
) -> None:
    """Layout = the generator's truth regions, so this tests the glue only.

    Classic detection on this page can pick a merged line that also covers the
    label (a P4.2 matter, reported separately), which would hide what P8.6 does.
    """
    _, page = page_png

    def fake_detect(p: object, cfg: object = None, **_: object) -> LayoutResult:
        return LayoutResult(
            page_index=0,
            regions=[t.region for t in page.regions],
            sources=("truth",),
        )

    monkeypatch.setattr("torus.service.run.detect_layout", fake_detect)


def _ctx(
    page: SynthPage, names: dict[str, str], *, calibrate: bool = False
) -> RunContext:
    template = _template()
    readers = FakeReaders(page, names)
    return RunContext(
        config=_config(template),
        template=template,
        readers=readers,
        classifier=DisabledPageClassifier("other"),
        calibrator=_calibrator(readers) if calibrate else None,
    )


def test_uncalibrated_run_writes_values_but_never_confirms(
    tmp_path: Path, page_png: tuple[Path, SynthPage]
) -> None:
    path, page = page_png
    ctx = _ctx(page, {"fake_a": "fam_a", "fake_b": "fam_b"})
    report = run_files(collect_inputs(path), ctx)
    assert report.files_ok == 1 and report.files_failed == 0

    out = tmp_path / "o.xlsx"
    tpl = write_blank_template(ctx.template, tmp_path / "t.xlsx")
    write_workbook(tpl, out, ctx.template, report.results, ctx.config.pipeline.output)

    row = report.results[0].rows[0]
    assert {fv.status.value for fv in row.values()} <= {"needs_review", "not_found"}
    found = [fv for fv in row.values() if fv.value]
    assert found, "anchored fields should have been read"
    assert all("uncalibrated" in (fv.reason or "") for fv in found)
    assert all(fv.evidence for fv in found)  # every filled cell has evidence
    assert {fv.value for fv in found} <= {t.text for t in page.regions}
    ws = load_workbook(out).worksheets[0]
    cells = {c.value for r in ws.iter_rows(min_row=2) for c in r if c.value}
    assert cells == {fv.value for fv in found}


def test_calibrated_two_family_run_confirms_agreeing_values(
    page_png: tuple[Path, SynthPage],
) -> None:
    path, page = page_png
    ctx = _ctx(page, {"fake_a": "fam_a", "fake_b": "fam_b"}, calibrate=True)
    report = run_files([path], ctx)
    found = [fv for fv in report.results[0].rows[0].values() if fv.value]
    assert found
    assert {fv.status.value for fv in found} == {"confirmed"}


def test_single_family_cannot_confirm_critical_field(
    page_png: tuple[Path, SynthPage],
) -> None:
    path, page = page_png
    ctx = _ctx(page, {"only": "fam_a"}, calibrate=True)
    report = run_files([path], ctx)
    found = [fv for fv in report.results[0].rows[0].values() if fv.value]
    assert found
    assert all(fv.status.value != "confirmed" for fv in found)


def test_bad_file_is_reported_and_others_still_run(
    page_png: tuple[Path, SynthPage],
) -> None:
    path, page = page_png
    (path.parent / "broken.png").write_bytes(b"not an image")
    report = run_files(collect_inputs(path.parent), _ctx(page, {"fake_a": "fam_a"}))
    assert report.files_ok == 1 and report.files_failed == 1


def test_cli_run_without_readers_yields_all_not_found(
    tmp_path: Path,
    page_png: tuple[Path, SynthPage],
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = page_png
    out = tmp_path / "cli.xlsx"
    code = main(["run", str(path), "--template", "deb_v1", "--out", str(out)])
    assert code == 0
    ws = load_workbook(out).worksheets[0]
    assert all(c.value is None for r in ws.iter_rows(min_row=2) for c in r)
    assert "rows=1" in capsys.readouterr().out


def test_cli_unknown_template_is_a_config_error(
    tmp_path: Path, page_png: tuple[Path, SynthPage]
) -> None:
    path, _ = page_png
    args = ["run", str(path), "--template", "nope", "--out", str(tmp_path / "x.xlsx")]
    assert main(args) == 2
