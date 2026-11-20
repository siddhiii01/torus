"""Markdown and HTML evaluation report writer (P1.3).

Reports carry metrics, timings, and version metadata only.
They must never include gold/predicted field values, OCR text, or page images.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.core.errors import EvalError
from torus.core.types import FieldStatus
from torus.eval.classify_score import (
    ClassificationMetrics,
    ConfusionMatrix,
    MisclassificationByType,
    PageTypeMetrics,
)
from torus.eval.location_score import ColumnLocationMetrics, LocationMetrics
from torus.eval.metrics import CaseScore, FieldMetrics
from torus.eval.review_metrics import ReviewMetrics

# Re-export classification / location models for ``from torus.eval.report`` callers.
__all__ = [
    "ClassificationMetrics",
    "ColumnLocationMetrics",
    "ConfusionMatrix",
    "EvalRunReport",
    "LocationMetrics",
    "MisclassificationByType",
    "PageTypeMetrics",
    "RunVersions",
    "TimingSummary",
    "WrittenReportPaths",
    "build_run_report",
    "load_run_report",
    "merge_timings",
    "render_html",
    "render_markdown",
    "write_report",
    "write_run_report_json",
]


class TimingSummary(BaseModel):
    """Aggregated wall-clock timings for a run (milliseconds)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_ms: float = Field(ge=0.0)
    by_stage: dict[str, float] = Field(default_factory=dict)
    case_count: int = Field(ge=0)
    mean_case_ms: float | None = Field(default=None, ge=0.0)


class RunVersions(BaseModel):
    """Model and config versions that produced the scored outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_hash: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    model_versions: dict[str, str] = Field(default_factory=dict)


class EvalRunReport(BaseModel):
    """One evaluation run summary suitable for Markdown/HTML export.

    Field-level detail is metrics-only (keys, counts, rates). Case ids and
    set ids are identifiers, not document body text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_id: str = Field(min_length=1)
    split: str | None = None
    created_at: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    case_ids: list[str] = Field(default_factory=list)
    scores: CaseScore
    page_types: list[PageTypeMetrics] = Field(
        default_factory=list,
        description="Per-type rows (also mirrored under classification.per_type).",
    )
    classification: ClassificationMetrics | None = Field(
        default=None,
        description="Full page-type eval: P/R/F1, confusion matrix, misclass by type.",
    )
    location: LocationMetrics | None = Field(
        default=None,
        description="Field-location eval before reading (P4.8): IoU match per column.",
    )
    review: ReviewMetrics | None = Field(
        default=None,
        description="Human review effort (P9.5); absent when no bundles were given.",
    )
    timings: TimingSummary
    versions: RunVersions


@dataclass(frozen=True)
class WrittenReportPaths:
    """Paths written by :func:`write_report`."""

    markdown: Path
    html: Path
    json_path: Path | None = None


def merge_timings(
    per_case: Sequence[Mapping[str, float]],
) -> TimingSummary:
    """Sum stage timings across cases into one :class:`TimingSummary`."""
    by_stage: dict[str, float] = {}
    for mapping in per_case:
        for stage, ms in mapping.items():
            by_stage[stage] = by_stage.get(stage, 0.0) + float(ms)
    total = sum(by_stage.values())
    n = len(per_case)
    mean = (total / n) if n else None
    return TimingSummary(
        total_ms=total,
        by_stage=dict(sorted(by_stage.items())),
        case_count=n,
        mean_case_ms=mean,
    )


def build_run_report(
    *,
    set_id: str,
    aggregate: CaseScore,
    case_ids: Sequence[str],
    timings: TimingSummary,
    versions: RunVersions,
    page_types: Sequence[PageTypeMetrics] | None = None,
    classification: ClassificationMetrics | None = None,
    location: LocationMetrics | None = None,
    review: ReviewMetrics | None = None,
    split: str | None = None,
    created_at: str | None = None,
) -> EvalRunReport:
    """Assemble an :class:`EvalRunReport` from scored metrics and run metadata."""
    when = created_at or datetime.now(UTC).isoformat()
    ids = [cid for cid in case_ids if cid]
    per_type = list(page_types or ())
    if classification is not None and not per_type:
        per_type = list(classification.per_type)
    return EvalRunReport(
        set_id=set_id,
        split=split,
        created_at=when,
        case_count=len(ids),
        case_ids=list(ids),
        scores=aggregate,
        page_types=per_type,
        classification=classification,
        location=location,
        review=review,
        timings=timings,
        versions=versions,
    )


def write_run_report_json(report: EvalRunReport, path: Path | str) -> Path:
    """Write *report* as JSON (metrics/metadata only) for later ``diff``."""
    out = Path(path)
    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_run_report(path: Path | str) -> EvalRunReport:
    """Load an :class:`EvalRunReport` from a JSON file written by eval."""
    report_path = Path(path)
    if not report_path.is_file():
        raise EvalError(f"eval report not found: {report_path}")
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"cannot read eval report: {report_path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid JSON in eval report: {report_path}") from exc
    try:
        return EvalRunReport.model_validate(data)
    except ValidationError as exc:
        raise EvalError(f"invalid eval report schema in {report_path}:\n{exc}") from exc


def write_report(
    report: EvalRunReport, markdown_path: Path | str
) -> WrittenReportPaths:
    """Write Markdown, HTML, and a JSON sidecar next to *markdown_path*.

    The JSON file is the machine-readable input for ``torus.eval diff``.
    """
    md_path = Path(markdown_path)
    if md_path.suffix.lower() not in {".md", ".markdown"}:
        md_path = md_path.with_suffix(".md")
    html_path = md_path.with_suffix(".html")
    json_path = md_path.with_suffix(".json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    write_run_report_json(report, json_path)
    return WrittenReportPaths(markdown=md_path, html=html_path, json_path=json_path)


def render_markdown(report: EvalRunReport) -> str:
    """Render *report* as Markdown. Never includes field values."""
    lines: list[str] = [
        "# Torus evaluation report",
        "",
        f"- **Set:** `{report.set_id}`",
        f"- **Split:** `{report.split or '—'}`",
        f"- **Cases:** {report.case_count}",
        f"- **Template:** `{report.versions.template_id}`",
        f"- **Created:** {report.created_at}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Field accuracy (micro) | {_pct(report.scores.fields_micro.accuracy)} |",
        f"| Field F1 (micro) | {_pct(report.scores.fields_micro.f1)} |",
        f"| Silent errors | {report.scores.silent_errors} |",
        f"| Automatic rate | {_pct(report.scores.status_rates.automatic_rate)} |",
        f"| Review rate | {_pct(report.scores.status_rates.review_rate)} |",
        f"| Row alignment F1 | {_pct(report.scores.rows.f1)} |",
        "",
        "### Status rates",
        "",
        "| Status | Count | Rate |",
        "|---|---:|---:|",
    ]
    for status in FieldStatus:
        count = report.scores.status_rates.counts.get(status.value, 0)
        rate = report.scores.status_rates.rates.get(status.value, 0.0)
        lines.append(f"| `{status.value}` | {count} | {_pct(rate)} |")

    lines.extend(["", "## Per column", ""])
    lines.append(
        "| Column | Compared | Accuracy | Silent errors | "
        "confirmed | likely | needs_review | not_found |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(report.scores.per_field):
        fm = report.scores.per_field[key]
        lines.append(_md_field_row(fm))

    lines.extend(_markdown_classification_section(report))
    lines.extend(_markdown_location_section(report))
    lines.extend(_markdown_review_section(report))

    lines.extend(["", "## Timing", ""])
    lines.append(f"- **Total:** {_ms(report.timings.total_ms)}")
    if report.timings.mean_case_ms is not None:
        lines.append(f"- **Mean per case:** {_ms(report.timings.mean_case_ms)}")
    if report.timings.by_stage:
        lines.extend(["", "| Stage | ms |", "|---|---:|"])
        for stage, ms in report.timings.by_stage.items():
            lines.append(f"| `{stage}` | {_fmt_num(ms)} |")

    lines.extend(["", "## Versions", ""])
    lines.append(f"- **Config hash:** `{report.versions.config_hash}`")
    lines.append(f"- **Template:** `{report.versions.template_id}`")
    if report.versions.model_versions:
        lines.extend(["", "| Model | Version |", "|---|---|"])
        for name, ver in sorted(report.versions.model_versions.items()):
            lines.append(f"| `{name}` | `{ver}` |")
    else:
        lines.append("- **Models:** (none recorded)")

    if report.case_ids:
        lines.extend(
            [
                "",
                "## Cases",
                "",
                "Case identifiers only (no document content):",
                "",
            ]
        )
        for cid in report.case_ids:
            lines.append(f"- `{cid}`")

    lines.append("")
    return "\n".join(lines)


def render_html(report: EvalRunReport) -> str:
    """Render *report* as a self-contained HTML document. Escapes all text."""
    title = html.escape(f"Torus evaluation — {report.set_id}")
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4}",
        "table{border-collapse:collapse;margin:1rem 0;width:100%;max-width:960px}",
        "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left}",
        "th{background:#f4f4f4}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums}",
        "code{font-size:0.95em}",
        "h1,h2,h3{margin-top:1.6rem}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape('Torus evaluation report')}</h1>",
        "<ul>",
        f"<li><strong>Set:</strong> <code>{html.escape(report.set_id)}</code></li>",
        f"<li><strong>Split:</strong> <code>"
        f"{html.escape(report.split or '—')}</code></li>",
        f"<li><strong>Cases:</strong> {report.case_count}</li>",
        f"<li><strong>Template:</strong> <code>"
        f"{html.escape(report.versions.template_id)}</code></li>",
        f"<li><strong>Created:</strong> {html.escape(report.created_at)}</li>",
        "</ul>",
        "<h2>Summary</h2>",
        "<table>",
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>",
        "<tbody>",
        _html_kv("Field accuracy (micro)", _pct(report.scores.fields_micro.accuracy)),
        _html_kv("Field F1 (micro)", _pct(report.scores.fields_micro.f1)),
        _html_kv("Silent errors", str(report.scores.silent_errors)),
        _html_kv("Automatic rate", _pct(report.scores.status_rates.automatic_rate)),
        _html_kv("Review rate", _pct(report.scores.status_rates.review_rate)),
        _html_kv("Row alignment F1", _pct(report.scores.rows.f1)),
        "</tbody></table>",
        "<h3>Status rates</h3>",
        "<table><thead><tr><th>Status</th><th>Count</th><th>Rate</th></tr></thead>",
        "<tbody>",
    ]
    for status in FieldStatus:
        count = report.scores.status_rates.counts.get(status.value, 0)
        rate = report.scores.status_rates.rates.get(status.value, 0.0)
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(status.value)}</code></td>"
            f'<td class="num">{count}</td>'
            f'<td class="num">{html.escape(_pct(rate))}</td>'
            "</tr>"
        )
    parts.append("</tbody></table>")

    parts.extend(
        [
            "<h2>Per column</h2>",
            "<table><thead><tr>",
            "<th>Column</th><th>Compared</th><th>Accuracy</th><th>Silent errors</th>",
            "<th>confirmed</th><th>likely</th><th>needs_review</th><th>not_found</th>",
            "</tr></thead><tbody>",
        ]
    )
    for key in sorted(report.scores.per_field):
        parts.append(_html_field_row(report.scores.per_field[key]))
    parts.append("</tbody></table>")

    parts.extend(_html_classification_section(report))
    parts.extend(_html_location_section(report))
    parts.extend(_html_review_section(report))

    parts.extend(
        [
            "<h2>Timing</h2>",
            "<ul>",
            (
                "<li><strong>Total:</strong> "
                f"{html.escape(_ms(report.timings.total_ms))}</li>"
            ),
        ]
    )
    if report.timings.mean_case_ms is not None:
        parts.append(
            f"<li><strong>Mean per case:</strong> "
            f"{html.escape(_ms(report.timings.mean_case_ms))}</li>"
        )
    parts.append("</ul>")
    if report.timings.by_stage:
        parts.append("<table><thead><tr><th>Stage</th><th>ms</th></tr></thead><tbody>")
        for stage, ms in report.timings.by_stage.items():
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(stage)}</code></td>"
                f'<td class="num">{html.escape(_fmt_num(ms))}</td>'
                "</tr>"
            )
        parts.append("</tbody></table>")

    parts.extend(
        [
            "<h2>Versions</h2>",
            "<ul>",
            f"<li><strong>Config hash:</strong> "
            f"<code>{html.escape(report.versions.config_hash)}</code></li>",
            f"<li><strong>Template:</strong> "
            f"<code>{html.escape(report.versions.template_id)}</code></li>",
            "</ul>",
        ]
    )
    if report.versions.model_versions:
        parts.append(
            "<table><thead><tr><th>Model</th><th>Version</th></tr></thead><tbody>"
        )
        for name, ver in sorted(report.versions.model_versions.items()):
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(name)}</code></td>"
                f"<td><code>{html.escape(ver)}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append("<p>Models: (none recorded)</p>")

    if report.case_ids:
        parts.append("<h2>Cases</h2>")
        parts.append("<p>Case identifiers only (no document content):</p>")
        parts.append("<ul>")
        for cid in report.case_ids:
            parts.append(f"<li><code>{html.escape(cid)}</code></li>")
        parts.append("</ul>")

    parts.extend(["</body>", "</html>", ""])
    return "\n".join(parts)


def _markdown_classification_section(report: EvalRunReport) -> list[str]:
    lines: list[str] = ["", "## Per page type", ""]
    clf = report.classification
    rows = report.page_types
    if clf is None and not rows:
        lines.append("No page-type classification metrics in this run.")
        return lines

    if clf is not None:
        lines.append(
            f"- **Overall accuracy:** {_pct(clf.accuracy)} ({clf.correct}/{clf.total})"
        )
        if clf.unmatched_gold or clf.unmatched_pred:
            lines.append(
                f"- **Unmatched gold indices:** {clf.unmatched_gold}; "
                f"**unmatched pred indices:** {clf.unmatched_pred}"
            )
        lines.append("")

    lines.append(
        "| Page type | Support | Correct | Accuracy | Precision | Recall | F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pt in rows:
        lines.append(
            f"| `{pt.page_type}` | {pt.support} | {pt.correct} | "
            f"{_pct(pt.accuracy)} | {_pct(pt.precision)} | "
            f"{_pct(pt.recall)} | {_pct(pt.f1)} |"
        )

    if clf is not None and clf.confusion is not None:
        lines.extend(["", "### Confusion matrix", ""])
        lines.append("Rows = gold, columns = predicted (counts only).")
        lines.append("")
        labels = clf.confusion.labels
        header = "| gold \\ pred | " + " | ".join(f"`{lab}`" for lab in labels) + " |"
        sep = "|---|" + "|".join(["---:" for _ in labels]) + "|"
        lines.append(header)
        lines.append(sep)
        for i, gold in enumerate(labels):
            cells = " | ".join(
                str(clf.confusion.counts[i][j]) for j in range(len(labels))
            )
            lines.append(f"| `{gold}` | {cells} |")

    if clf is not None and clf.misclassifications:
        lines.extend(["", "### Misclassifications by gold type", ""])
        lines.append("| Gold type | Total wrong | Predicted as (counts) |")
        lines.append("|---|---:|---|")
        for row in clf.misclassifications:
            parts = ", ".join(
                f"`{pred}`×{count}" for pred, count in row.as_predicted.items()
            )
            lines.append(f"| `{row.gold_type}` | {row.total} | {parts or '—'} |")
    elif clf is not None and clf.total > 0 and clf.correct == clf.total:
        lines.extend(
            [
                "",
                "### Misclassifications by gold type",
                "",
                "None (all scored pages matched).",
            ]
        )

    return lines


def _html_classification_section(report: EvalRunReport) -> list[str]:
    parts: list[str] = ["<h2>Per page type</h2>"]
    clf = report.classification
    rows = report.page_types
    if clf is None and not rows:
        parts.append("<p>No page-type classification metrics in this run.</p>")
        return parts

    if clf is not None:
        parts.append("<ul>")
        parts.append(
            f"<li><strong>Overall accuracy:</strong> "
            f"{html.escape(_pct(clf.accuracy))} "
            f"({clf.correct}/{clf.total})</li>"
        )
        if clf.unmatched_gold or clf.unmatched_pred:
            parts.append(
                f"<li><strong>Unmatched gold indices:</strong> {clf.unmatched_gold}; "
                f"<strong>unmatched pred indices:</strong> {clf.unmatched_pred}</li>"
            )
        parts.append("</ul>")

    parts.append(
        "<table><thead><tr>"
        "<th>Page type</th><th>Support</th><th>Correct</th><th>Accuracy</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th>"
        "</tr></thead><tbody>"
    )
    for pt in rows:
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(pt.page_type)}</code></td>"
            f'<td class="num">{pt.support}</td>'
            f'<td class="num">{pt.correct}</td>'
            f'<td class="num">{html.escape(_pct(pt.accuracy))}</td>'
            f'<td class="num">{html.escape(_pct(pt.precision))}</td>'
            f'<td class="num">{html.escape(_pct(pt.recall))}</td>'
            f'<td class="num">{html.escape(_pct(pt.f1))}</td>'
            "</tr>"
        )
    parts.append("</tbody></table>")

    if clf is not None and clf.confusion is not None:
        labels = clf.confusion.labels
        parts.append("<h3>Confusion matrix</h3>")
        parts.append("<p>Rows = gold, columns = predicted (counts only).</p>")
        parts.append("<table><thead><tr><th>gold \\ pred</th>")
        for lab in labels:
            parts.append(f"<th><code>{html.escape(lab)}</code></th>")
        parts.append("</tr></thead><tbody>")
        for i, gold in enumerate(labels):
            parts.append(f"<tr><td><code>{html.escape(gold)}</code></td>")
            for j in range(len(labels)):
                parts.append(f'<td class="num">{clf.confusion.counts[i][j]}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")

    if clf is not None and clf.misclassifications:
        parts.append("<h3>Misclassifications by gold type</h3>")
        parts.append(
            "<table><thead><tr>"
            "<th>Gold type</th><th>Total wrong</th><th>Predicted as (counts)</th>"
            "</tr></thead><tbody>"
        )
        for row in clf.misclassifications:
            detail = ", ".join(
                f"<code>{html.escape(pred)}</code>×{count}"
                for pred, count in row.as_predicted.items()
            )
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(row.gold_type)}</code></td>"
                f'<td class="num">{row.total}</td>'
                f"<td>{detail or '—'}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    elif clf is not None and clf.total > 0 and clf.correct == clf.total:
        parts.append("<h3>Misclassifications by gold type</h3>")
        parts.append("<p>None (all scored pages matched).</p>")

    return parts


_REVIEW_TOP_N = 10


def _review_summary_rows(rm: ReviewMetrics) -> list[tuple[str, str]]:
    spf = "n/a" if rm.seconds_per_field is None else f"{rm.seconds_per_field:.1f} s"
    return [
        ("Packs reviewed", str(rm.packs)),
        ("Fields needing review", str(rm.fields_needing_review)),
        (
            "Fields needing review per pack",
            f"{rm.fields_needing_review_per_pack:.2f}",
        ),
        ("Accepted", str(rm.accepted)),
        ("Corrected", str(rm.corrected)),
        ("Skipped", str(rm.skipped)),
        ("Undecided", str(rm.undecided)),
        ("Correction rate", _pct(rm.correction_rate)),
        (
            f"Time per field (gaps <= {rm.max_gap_s:g} s, n={rm.timed_fields})",
            spf,
        ),
    ]


def _markdown_review_section(report: EvalRunReport) -> list[str]:
    lines: list[str] = ["", "## Review effort", ""]
    rm = report.review
    if rm is None:
        lines.append("No review metrics in this run.")
        return lines
    lines.extend(["| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in _review_summary_rows(rm))
    top = rm.most_corrected[:_REVIEW_TOP_N]
    if top:
        lines.extend(
            [
                "",
                "### Most-corrected columns",
                "",
                "| Column | Decided | Corrected | Correction rate |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| `{c.column_key}` | {c.decided} | {c.corrected} | "
            f"{_pct(c.correction_rate)} |"
            for c in top
        )
    return lines


def _html_review_section(report: EvalRunReport) -> list[str]:
    parts: list[str] = ["<h2>Review effort</h2>"]
    rm = report.review
    if rm is None:
        parts.append("<p>No review metrics in this run.</p>")
        return parts
    parts.extend(
        ["<table>", "<thead><tr><th>Metric</th><th>Value</th></tr></thead>", "<tbody>"]
    )
    parts.extend(_html_kv(k, v) for k, v in _review_summary_rows(rm))
    parts.append("</tbody></table>")
    top = rm.most_corrected[:_REVIEW_TOP_N]
    if top:
        parts.extend(
            [
                "<h3>Most-corrected columns</h3>",
                "<table>",
                "<thead><tr><th>Column</th><th>Decided</th><th>Corrected</th>"
                "<th>Correction rate</th></tr></thead>",
                "<tbody>",
            ]
        )
        parts.extend(
            f"<tr><td><code>{html.escape(c.column_key)}</code></td>"
            f'<td class="num">{c.decided}</td><td class="num">{c.corrected}</td>'
            f'<td class="num">{html.escape(_pct(c.correction_rate))}</td></tr>'
            for c in top
        )
        parts.append("</tbody></table>")
    return parts


def _markdown_location_section(report: EvalRunReport) -> list[str]:
    lines: list[str] = ["", "## Field location", ""]
    loc = report.location
    if loc is None:
        lines.append("No field-location metrics in this run.")
        return lines

    lines.append(
        f"- **Located rate:** {_pct(loc.located_rate)} ({loc.located}/{loc.expected})"
    )
    lines.append(f"- **Precision:** {_pct(loc.precision)}")
    lines.append(f"- **F1:** {_pct(loc.f1)}")
    lines.append(f"- **Mean IoU (matched):** {_pct(loc.mean_iou)}")
    lines.append(f"- **IoU threshold:** {loc.iou_threshold:.2f}")
    if loc.unmatched_pred:
        lines.append(f"- **Unmatched predictions:** {loc.unmatched_pred}")
    if loc.missed_columns:
        keys = ", ".join(f"`{k}`" for k in loc.missed_columns)
        lines.append(f"- **Columns with misses:** {keys}")
    lines.append("")
    lines.append("| Column | Expected | Located | Recall | Precision | F1 | Mean IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for col in loc.per_column:
        lines.append(
            f"| `{col.column_key}` | {col.expected} | {col.located} | "
            f"{_pct(col.recall)} | {_pct(col.precision)} | {_pct(col.f1)} | "
            f"{_pct(col.mean_iou)} |"
        )
    return lines


def _html_location_section(report: EvalRunReport) -> list[str]:
    parts: list[str] = ["<h2>Field location</h2>"]
    loc = report.location
    if loc is None:
        parts.append("<p>No field-location metrics in this run.</p>")
        return parts

    parts.append("<ul>")
    parts.append(
        f"<li><strong>Located rate:</strong> "
        f"{html.escape(_pct(loc.located_rate))} "
        f"({loc.located}/{loc.expected})</li>"
    )
    parts.append(
        f"<li><strong>Precision:</strong> {html.escape(_pct(loc.precision))}</li>"
    )
    parts.append(f"<li><strong>F1:</strong> {html.escape(_pct(loc.f1))}</li>")
    parts.append(
        f"<li><strong>Mean IoU (matched):</strong> "
        f"{html.escape(_pct(loc.mean_iou))}</li>"
    )
    parts.append(f"<li><strong>IoU threshold:</strong> {loc.iou_threshold:.2f}</li>")
    if loc.unmatched_pred:
        parts.append(
            f"<li><strong>Unmatched predictions:</strong> {loc.unmatched_pred}</li>"
        )
    if loc.missed_columns:
        keys = ", ".join(f"<code>{html.escape(k)}</code>" for k in loc.missed_columns)
        parts.append(f"<li><strong>Columns with misses:</strong> {keys}</li>")
    parts.append("</ul>")
    parts.append(
        "<table><thead><tr>"
        "<th>Column</th><th>Expected</th><th>Located</th>"
        "<th>Recall</th><th>Precision</th><th>F1</th><th>Mean IoU</th>"
        "</tr></thead><tbody>"
    )
    for col in loc.per_column:
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(col.column_key)}</code></td>"
            f'<td class="num">{col.expected}</td>'
            f'<td class="num">{col.located}</td>'
            f'<td class="num">{html.escape(_pct(col.recall))}</td>'
            f'<td class="num">{html.escape(_pct(col.precision))}</td>'
            f'<td class="num">{html.escape(_pct(col.f1))}</td>'
            f'<td class="num">{html.escape(_pct(col.mean_iou))}</td>'
            "</tr>"
        )
    parts.append("</tbody></table>")
    return parts


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _ms(value: float) -> str:
    return f"{_fmt_num(value)} ms"


def _fmt_num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _status_count(fm: FieldMetrics, status: FieldStatus) -> int:
    return fm.status_counts.get(status.value, 0)


def _md_field_row(fm: FieldMetrics) -> str:
    return (
        f"| `{fm.key}` | {fm.compared} | {_pct(fm.accuracy)} | {fm.silent_errors} | "
        f"{_status_count(fm, FieldStatus.CONFIRMED)} | "
        f"{_status_count(fm, FieldStatus.LIKELY)} | "
        f"{_status_count(fm, FieldStatus.NEEDS_REVIEW)} | "
        f"{_status_count(fm, FieldStatus.NOT_FOUND)} |"
    )


def _html_kv(label: str, value: str) -> str:
    return (
        f"<tr><td>{html.escape(label)}</td>"
        f'<td class="num">{html.escape(value)}</td></tr>'
    )


def _html_field_row(fm: FieldMetrics) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(fm.key)}</code></td>"
        f'<td class="num">{fm.compared}</td>'
        f'<td class="num">{html.escape(_pct(fm.accuracy))}</td>'
        f'<td class="num">{fm.silent_errors}</td>'
        f'<td class="num">{_status_count(fm, FieldStatus.CONFIRMED)}</td>'
        f'<td class="num">{_status_count(fm, FieldStatus.LIKELY)}</td>'
        f'<td class="num">{_status_count(fm, FieldStatus.NEEDS_REVIEW)}</td>'
        f'<td class="num">{_status_count(fm, FieldStatus.NOT_FOUND)}</td>'
        "</tr>"
    )
