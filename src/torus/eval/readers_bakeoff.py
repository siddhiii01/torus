"""Reader bake-off (P5.8): per reader × field type accuracy and speed on a dev set.

Each reader reads the **gold** value box of every labelled field (from
``locations.json``), so the numbers isolate reading quality from layout. A
reading is correct when it matches the gold value under the field's own match
rule (``torus.eval.match``). Reader errors and timeouts are counted separately
and never as a correct reading.

The report is metrics only: no crops, no document text, no values. It ends with
a *proposed* default reader set; ``config/readers.yaml`` is never edited here,
because a human approves defaults.

Silent errors (a wrong ``confirmed`` value) cannot be measured per reader: they
need voting and calibration (Phase 6). This table is evidence for choosing
voters, not a confidence claim.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from torus.config import default_config_dir, load_config
from torus.config.schemas import FieldSpec, TemplateConfig
from torus.core.errors import EvalError
from torus.core.models import ReadHint, Reading, Region
from torus.core.types import RegionKind
from torus.eval.match import values_match
from torus.eval.normalize import is_empty
from torus.eval.schema import Split
from torus.eval.validate import ValidatedCase, validate_testset

DEFAULT_MIN_SAMPLES = 10


class ReadRunner(Protocol):
    """The slice of ``ReaderRunner`` the bake-off uses (injectable in tests)."""

    def read(
        self, reader_name: str, image: Any, region: Region, hint: ReadHint | None = ...
    ) -> Reading: ...


@dataclass(frozen=True, slots=True)
class Outcome:
    """One reader's result on one gold field. Holds no text."""

    reader: str
    field_type: str
    correct: bool
    error: bool
    empty: bool
    time_ms: float


class CellStats(BaseModel):
    """Aggregate for one reader on one field type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reader: str
    field_type: str
    n: int = Field(ge=0)
    correct: int = Field(ge=0)
    errors: int = Field(ge=0)
    empty: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    median_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)


class Proposal(BaseModel):
    """Suggested voters for one field type. A human approves; nothing is applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_type: str
    primary: str | None = None
    second_family: str | None = None
    note: str = ""


class BakeoffReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    set_id: str
    template_id: str
    case_count: int
    field_samples: int
    skipped_fields: int
    min_samples: int
    readers_run: list[str]
    readers_skipped: dict[str, str]
    stats: list[CellStats]
    proposals: list[Proposal]
    proposed_enabled: list[str]


# --- pure scoring and aggregation ------------------------------------------


def hint_for_field(spec: FieldSpec) -> ReadHint:
    """Decoding constraints from the field spec, as the pipeline will pass them."""
    return ReadHint(
        charset=spec.charset,
        expected_length=spec.length,
        min_length=spec.min_length,
        max_length=spec.max_length,
    )


def score_outcome(reader: str, spec: FieldSpec, reading: Reading, gold: Any) -> Outcome:
    """Compare one reading to gold under the field's match rule."""
    failed = reading.error is not None
    text = "" if failed else reading.text
    return Outcome(
        reader=reader,
        field_type=spec.type.value,
        correct=(not failed) and values_match(spec, text, gold),
        error=failed,
        empty=(not failed) and not text.strip(),
        time_ms=reading.time_ms,
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list (``q`` in [0, 1])."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def aggregate(outcomes: Sequence[Outcome]) -> list[CellStats]:
    """Group by (field type, reader); sorted by field type then reader name."""
    groups: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for item in outcomes:
        groups[(item.field_type, item.reader)].append(item)
    stats: list[CellStats] = []
    for (field_type, reader), items in sorted(groups.items()):
        times = sorted(i.time_ms for i in items)
        n = len(items)
        correct = sum(i.correct for i in items)
        stats.append(
            CellStats(
                reader=reader,
                field_type=field_type,
                n=n,
                correct=correct,
                errors=sum(i.error for i in items),
                empty=sum(i.empty for i in items),
                accuracy=correct / n,
                median_ms=statistics.median(times),
                p95_ms=_percentile(times, 0.95),
            )
        )
    return stats


def propose_defaults(
    stats: Sequence[CellStats],
    families: Mapping[str, str],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> tuple[list[Proposal], list[str]]:
    """Per field type: best reader, plus best reader of a *different family*.

    Two families are proposed because a value needs two independent agreeing
    signals to be confirmed (AGENTS.md 3.4). Field types with fewer than
    ``min_samples`` readings for every reader get no proposal.
    """
    by_type: dict[str, list[CellStats]] = defaultdict(list)
    for cell in stats:
        if cell.n >= min_samples:
            by_type[cell.field_type].append(cell)

    seen_types = sorted({c.field_type for c in stats})
    proposals: list[Proposal] = []
    enabled: set[str] = set()
    for field_type in seen_types:
        cells = sorted(
            by_type.get(field_type, []),
            key=lambda c: (-c.accuracy, c.median_ms, c.reader),
        )
        if not cells:
            proposals.append(
                Proposal(
                    field_type=field_type,
                    note=f"fewer than {min_samples} samples: no proposal",
                )
            )
            continue
        best = cells[0]
        other = next(
            (
                c
                for c in cells[1:]
                if families.get(c.reader) != families.get(best.reader)
            ),
            None,
        )
        note = "" if other is not None else "no reader from a second family"
        proposals.append(
            Proposal(
                field_type=field_type,
                primary=best.reader,
                second_family=other.reader if other is not None else None,
                note=note,
            )
        )
        enabled.add(best.reader)
        if other is not None:
            enabled.add(other.reader)
    return proposals, sorted(enabled)


# --- rendering --------------------------------------------------------------


def render_markdown(report: BakeoffReport) -> str:
    lines = [
        f"# Reader bake-off: set `{report.set_id}`",
        "",
        f"- template: `{report.template_id}`; cases: {report.case_count}; "
        f"gold fields read: {report.field_samples}; "
        f"fields skipped: {report.skipped_fields}",
        f"- readers run: {', '.join(report.readers_run) or 'none'}",
        "- accuracy = readings matching gold under each field's match rule, "
        "out of all attempts (errors count as wrong)",
        "- gold boxes are used, so layout errors are excluded",
        "",
        "| Field type | Reader | n | Accuracy | Errors | Empty | Median ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report.stats:
        lines.append(
            f"| {c.field_type} | {c.reader} | {c.n} | {c.accuracy:.3f} | "
            f"{c.errors} | {c.empty} | {c.median_ms:.0f} | {c.p95_ms:.0f} |"
        )
    lines += ["", "## Proposed defaults (human approves)", ""]
    lines += ["| Field type | Primary | Second family | Note |", "|---|---|---|---|"]
    for p in report.proposals:
        lines.append(
            f"| {p.field_type} | {p.primary or '-'} | "
            f"{p.second_family or '-'} | {p.note} |"
        )
    lines += ["", f"Readers to enable: {', '.join(report.proposed_enabled) or 'none'}"]
    if report.readers_skipped:
        lines += ["", "## Skipped readers", ""]
        lines += [f"- {n}: {why}" for n, why in sorted(report.readers_skipped.items())]
    return "\n".join(lines) + "\n"


def write_report(report: BakeoffReport, markdown_path: Path) -> tuple[Path, Path]:
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = markdown_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return markdown_path, json_path


# --- the run ----------------------------------------------------------------


def _gold_value(case: ValidatedCase, column: str, row_index: int) -> Any:
    rows = case.expected.rows
    if row_index >= len(rows):
        return None
    return rows[row_index].get(column)


def bakeoff_case(
    case: ValidatedCase,
    template: TemplateConfig,
    pages: Sequence[Any],
    runner: ReadRunner,
    reader_names: Sequence[str],
) -> tuple[list[Outcome], int, int]:
    """Read every gold field of one case with every reader.

    Returns outcomes, the number of fields read, and the number skipped (no gold
    value, unknown column, or page out of range).
    """
    specs = {f.key: f for f in template.fields}
    outcomes: list[Outcome] = []
    skipped = 0
    attempted = 0
    for loc in case.locations.fields if case.locations is not None else []:
        spec = specs.get(loc.column_key)
        gold = _gold_value(case, loc.column_key, loc.row_index)
        if spec is None or is_empty(gold) or loc.page_index >= len(pages):
            skipped += 1
            continue
        region = Region(
            id=f"{case.case_id}:{loc.column_key}:{loc.row_index}",
            page_index=loc.page_index,
            bbox=loc.bbox,
            kind=loc.kind or RegionKind.FIELD_VALUE,
        )
        image = pages[loc.page_index].image
        hint = hint_for_field(spec)
        attempted += 1
        for name in reader_names:
            reading = runner.read(name, image, region, hint)
            outcomes.append(score_outcome(name, spec, reading, gold))
    return outcomes, attempted, skipped


def run_bakeoff(
    *,
    set_id: str,
    report_path: Path,
    reader_names: Sequence[str] | None = None,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
    split: Split | None = Split.DEV,
    template_id: str | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    runner: ReadRunner | None = None,
) -> tuple[BakeoffReport, tuple[Path, Path]]:
    """Run the bake-off and write ``report_path`` (+ JSON sidecar)."""
    from torus.ingest import ingest  # heavy-ish (pymupdf): only when running
    from torus.readers import ReaderRegistry, ReaderRunner, ReadingCache

    config = load_config(Path(config_dir) if config_dir else default_config_dir())
    cases = [
        c
        for c in validate_testset(set_id, data_dir=data_dir)
        if split is None or c.meta.split == split
    ]
    if not cases:
        raise EvalError(f"no cases in set {set_id!r}")
    ids = {c.meta.template_id for c in cases}
    chosen = template_id or (next(iter(ids)) if len(ids) == 1 else None)
    if chosen is None or chosen not in config.templates:
        raise EvalError("pass --template naming one known template id")
    template = config.templates[chosen]

    all_names = [r.name for r in config.readers.readers]
    wanted = list(reader_names) if reader_names else all_names
    unknown = [n for n in wanted if n not in all_names]
    if unknown:
        raise EvalError(f"unknown reader(s): {', '.join(unknown)}")

    # enabled is ignored on purpose: readers are enabled only after this runs.
    registry = ReaderRegistry(config.readers, require_enabled=False)
    skipped_readers = {
        n: registry.skip_reasons().get(n, "unavailable")
        for n in wanted
        if n in registry.skip_reasons()
    }
    names = [n for n in wanted if n not in skipped_readers]
    if runner is None:
        # No cache: every reading is timed for real.
        runner = ReaderRunner(registry, cache=ReadingCache(0))

    outcomes: list[Outcome] = []
    skipped_fields = 0
    read_fields = 0
    for case in cases:
        document = ingest(case.source_path, pipeline=config.pipeline).document
        case_outcomes, attempted, skipped = bakeoff_case(
            case, template, document.pages, runner, names
        )
        outcomes += case_outcomes
        read_fields += attempted
        skipped_fields += skipped

    # A reader that failed to load only shows up after its first read.
    for name in list(names):
        entry = registry.get_entry(name)
        if entry is not None and not entry.available:
            skipped_readers[name] = entry.skip_reason or "unavailable"
    ran = sorted({o.reader for o in outcomes} - set(skipped_readers))
    outcomes = [o for o in outcomes if o.reader not in skipped_readers]

    stats = aggregate(outcomes)
    families = {r.name: r.family for r in config.readers.readers}
    proposals, enabled = propose_defaults(stats, families, min_samples=min_samples)
    report = BakeoffReport(
        set_id=set_id,
        template_id=template.id,
        case_count=len(cases),
        field_samples=read_fields,
        skipped_fields=skipped_fields,
        min_samples=min_samples,
        readers_run=ran,
        readers_skipped=skipped_readers,
        stats=stats,
        proposals=proposals,
        proposed_enabled=enabled,
    )
    return report, write_report(report, report_path)
