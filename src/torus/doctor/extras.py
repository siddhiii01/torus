"""Detect which optional extras from ``pyproject.toml`` are importable."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

# Extra name → marker modules. An extra is "installed" when every marker imports.
EXTRA_MARKERS: dict[str, tuple[str, ...]] = {
    "paddle": ("paddle", "paddleocr"),
    "doctr": ("doctr",),
    "vlm": ("torch", "transformers"),
    "train": ("accelerate", "datasets", "peft"),
    "review": ("streamlit",),
}

# Reader family → required pyproject extra (None = base install is enough).
FAMILY_EXTRA: dict[str, str | None] = {
    "stub": None,
    "paddle": "paddle",
    "doctr": "doctr",
    "vlm": "vlm",
    # Local-server document VLMs (P5.3); stdlib HTTP, no pip extra.
    "document_vlm": None,
    # General VLMs (P5.4); HTTP to local server, no pip extra for the client.
    "general_vlm": None,
    # Character-cell MLP (P5.5); NumPy inference, weights under TORUS_MODEL_DIR.
    "char_cell": None,
    "tesseract": None,  # system binary; checked separately later if wired
}


@dataclass(frozen=True)
class ExtraStatus:
    """Whether one optional extra is present in this environment."""

    name: str
    installed: bool
    missing_modules: tuple[str, ...]


def module_available(module_name: str) -> bool:
    """Return True if ``module_name`` can be found without importing it."""
    return importlib.util.find_spec(module_name) is not None


def detect_extras(
    markers: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, ExtraStatus]:
    """Probe each configured extra against its marker modules."""
    table = markers if markers is not None else EXTRA_MARKERS
    result: dict[str, ExtraStatus] = {}
    for name, mods in table.items():
        missing = tuple(m for m in mods if not module_available(m))
        result[name] = ExtraStatus(
            name=name,
            installed=not missing,
            missing_modules=missing,
        )
    return result


def family_extra_installed(
    family: str,
    extras: dict[str, ExtraStatus],
) -> tuple[bool, str | None]:
    """Return (ok, reason) for whether a reader family's extra is present.

    Unknown families and families mapped to ``None`` need no pip extra (base
    install only), so a new family can land in config before its probe mapping.
    """
    if family not in FAMILY_EXTRA:
        return True, None
    extra_name = FAMILY_EXTRA[family]
    if extra_name is None:
        return True, None
    status = extras.get(extra_name)
    if status is None:
        return False, f"extra {extra_name!r} not known to doctor"
    if status.installed:
        return True, None
    missing = ", ".join(status.missing_modules) or extra_name
    return False, f"extra {extra_name!r} not installed (missing: {missing})"
