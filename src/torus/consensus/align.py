"""ROVER-style character alignment of several readings of one region.

Readings are merged progressively into a column grid. Each column holds the
symbols seen so far (``None`` is a gap). The caller controls the merge order, so
results are deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

Row = list[str | None]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (unit costs)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align_texts(texts: Sequence[str]) -> list[Row]:
    """Align ``texts`` into equal-length rows of characters and gaps.

    Row ``i`` corresponds to ``texts[i]``; removing the gaps from it gives back
    ``texts[i]`` exactly. The first text seeds the grid.
    """
    if not texts:
        return []
    rows: list[Row] = [list(texts[0])]
    for text in texts[1:]:
        _merge(rows, text)
    return rows


def _merge(rows: list[Row], text: str) -> None:
    n_cols = len(rows[0])
    col_syms = [{row[c] for row in rows} for c in range(n_cols)]
    m = len(text)

    # cost[i][j]: best cost aligning the first i columns with the first j chars.
    cost = [[0] * (m + 1) for _ in range(n_cols + 1)]
    for i in range(1, n_cols + 1):
        cost[i][0] = cost[i - 1][0] + _skip_cost(col_syms[i - 1])
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n_cols + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + _sub_cost(col_syms[i - 1], text[j - 1]),
                cost[i - 1][j] + _skip_cost(col_syms[i - 1]),
                cost[i][j - 1] + 1,
            )

    # Backtrace; prefer diagonal, then skip, then insert, for stable output.
    new_row: Row = []
    i, j = n_cols, m
    # (char or None for a gap, existing column index or negative for an insertion)
    placed: list[tuple[str | None, int]] = []
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and cost[i][j]
            == cost[i - 1][j - 1] + _sub_cost(col_syms[i - 1], text[j - 1])
        ):
            placed.append((text[j - 1], i - 1))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + _skip_cost(col_syms[i - 1]):
            placed.append((None, i - 1))
            i -= 1
        else:
            placed.append((text[j - 1], -1))
            j -= 1
    placed.reverse()

    out_rows: list[Row] = [[] for _ in rows]
    for ch, tag in placed:
        if tag >= 0:
            for r, row in enumerate(rows):
                out_rows[r].append(row[tag])
        else:
            for out in out_rows:
                out.append(None)
        new_row.append(ch)
    for r, out in enumerate(out_rows):
        rows[r] = out
    rows.append(new_row)


def _sub_cost(symbols: set[str | None], ch: str) -> int:
    return 0 if ch in symbols else 1


def _skip_cost(symbols: set[str | None]) -> int:
    return 0 if None in symbols else 1
