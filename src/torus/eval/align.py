"""Row alignment for evaluation (configurable per template)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torus.config.schemas import RowKeyConfig, RowKeyPart
from torus.eval.normalize import apply_kind, is_empty
from torus.eval.schema import ExpectedCell

RowDict = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Gold/pred pairs plus rows that did not align."""

    pairs: list[tuple[RowDict, RowDict]]
    unmatched_gold: list[RowDict]
    unmatched_pred: list[RowDict]


def _first_nonempty(row: RowDict, sources: Sequence[str]) -> ExpectedCell | Any:
    for key in sources:
        if key not in row:
            continue
        value = row[key]
        # FieldValue-like objects expose .value
        if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
            value = value.value
        if not is_empty(value):
            return value
    return None


def part_value(
    row: RowDict,
    part: RowKeyPart,
) -> str:
    """Canonical value for one row-key part."""
    raw = _first_nonempty(row, part.sources)
    return apply_kind(part.normalize, raw, aliases=part.aliases or None)


def primary_key(row: RowDict, config: RowKeyConfig) -> tuple[str, ...]:
    return tuple(part_value(row, part) for part in config.primary)


def disambiguator_key(row: RowDict, config: RowKeyConfig) -> str:
    if config.disambiguator is None:
        return ""
    return part_value(row, config.disambiguator)


def align_by_index(
    gold_rows: Sequence[RowDict],
    pred_rows: Sequence[RowDict],
) -> AlignmentResult:
    """Pair rows by position; leftovers are unmatched."""
    n = min(len(gold_rows), len(pred_rows))
    pairs = [(gold_rows[i], pred_rows[i]) for i in range(n)]
    unmatched_gold = list(gold_rows[n:])
    unmatched_pred = list(pred_rows[n:])
    return AlignmentResult(
        pairs=pairs,
        unmatched_gold=unmatched_gold,
        unmatched_pred=unmatched_pred,
    )


def align_rows(
    gold_rows: Sequence[RowDict],
    pred_rows: Sequence[RowDict],
    row_key: RowKeyConfig | None,
) -> AlignmentResult:
    """Align predicted rows to gold.

    When *row_key* is set, match on the configured primary key (and optional
    disambiguator on collisions). Otherwise align by index.
    """
    if row_key is None:
        return align_by_index(gold_rows, pred_rows)

    gold_by_key: dict[tuple[str, ...], list[RowDict]] = defaultdict(list)
    for gold in gold_rows:
        gold_by_key[primary_key(gold, row_key)].append(gold)

    pairs: list[tuple[RowDict, RowDict]] = []
    unmatched_pred: list[RowDict] = []
    used_gold_ids: set[int] = set()

    for pred in pred_rows:
        key = primary_key(pred, row_key)
        candidates = [
            gold for gold in gold_by_key.get(key, []) if id(gold) not in used_gold_ids
        ]
        if not candidates:
            unmatched_pred.append(pred)
            continue
        if len(candidates) == 1:
            gold = candidates[0]
        else:
            pred_dis = disambiguator_key(pred, row_key)
            by_dis = [
                gold
                for gold in candidates
                if disambiguator_key(gold, row_key) == pred_dis
            ]
            if by_dis:
                gold = by_dis[0]
            else:
                unmatched_pred.append(pred)
                continue
        pairs.append((gold, pred))
        used_gold_ids.add(id(gold))

    unmatched_gold = [gold for gold in gold_rows if id(gold) not in used_gold_ids]
    return AlignmentResult(
        pairs=pairs,
        unmatched_gold=unmatched_gold,
        unmatched_pred=unmatched_pred,
    )
