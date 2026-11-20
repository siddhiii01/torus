#!/usr/bin/env python3
"""Pre-commit / CI guard: block sensitive artifacts under data paths.

Real customer documents and answer keys must live under TORUS_DATA_DIR,
outside the repository. This check fails the commit when staged paths
look like data artifacts (PDFs, images, spreadsheets, or answer-key JSON)
under ``data/``, ``samples/``, or ``**/fixtures/real/``.

No document content is read or printed — only path names.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

# Path prefixes (posix, relative to repo root) that must not hold real data.
_BLOCKED_PREFIXES: tuple[str, ...] = (
    "data/",
    "samples/",
    "fixtures/real/",
)

_BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".csv",
        ".zip",
        ".eml",
    }
)

# Answer keys and case metadata under data paths.
_SENSITIVE_JSON_NAMES: frozenset[str] = frozenset(
    {
        "gold.json",
        "expected.json",
        "answer.json",
        "answers.json",
        "labels.json",
        "meta.json",
    }
)


def _normalize(path: str) -> str:
    """Normalize to forward-slash, strip leading ./, lower-drive on Windows."""
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def is_under_blocked_prefix(path: str) -> bool:
    """Return True if *path* is under a blocked data prefix."""
    norm = _normalize(path)
    lowered = norm.lower()
    for prefix in _BLOCKED_PREFIXES:
        if lowered.startswith(prefix) or f"/{prefix}" in f"/{lowered}":
            return True
    # Also catch nested ``**/fixtures/real/`` without a hard-coded customer name.
    parts = PurePosixPath(lowered).parts
    for i in range(len(parts) - 1):
        if parts[i] == "fixtures" and parts[i + 1] == "real":
            return True
    return False


def is_blocked_path(path: str) -> bool:
    """Return True if staging *path* would violate data-safety rules."""
    if not is_under_blocked_prefix(path):
        return False
    norm = _normalize(path)
    name = PurePosixPath(norm).name.lower()
    suffix = PurePosixPath(norm).suffix.lower()
    if suffix in _BINARY_SUFFIXES:
        return True
    if name in _SENSITIVE_JSON_NAMES:
        return True
    # Any other JSON under blocked prefixes is treated as sensitive.
    return suffix == ".json"


def check_paths(paths: list[str]) -> list[str]:
    """Return the subset of *paths* that must not be committed."""
    return [p for p in paths if is_blocked_path(p)]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    blocked = check_paths(args)
    if not blocked:
        return 0
    print(
        "Data-safety check failed: real documents and answer keys must live "
        "under TORUS_DATA_DIR (outside the repo), not under data/ or samples/.",
        file=sys.stderr,
    )
    print("Blocked paths (names only):", file=sys.stderr)
    for path in blocked:
        # Metadata only — never print file contents.
        print(f"  {path}", file=sys.stderr)
    print(
        "See docs/SETUP.md (TORUS_DATA_DIR layout). "
        "History rewrite needs owner approval.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
