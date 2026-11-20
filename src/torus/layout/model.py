"""Optional layout-model backend (P4.1).

No new model dependency is introduced here. When a human-approved backend is
wired later, register it via ``LayoutModelConfig.backend``. Missing or
disabled backends are skipped — never a crash.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from torus.config.schemas import LayoutModelConfig
from torus.core.models import Region


@runtime_checkable
class LayoutModelBackend(Protocol):
    """Plug-in layout model: image → regions (heavy imports stay inside)."""

    name: str

    def detect(self, image: Image.Image, *, page_index: int = 0) -> list[Region]: ...


class NullLayoutModel:
    """No-op backend used when the model path is disabled or unavailable."""

    name = "null"

    def detect(self, image: Image.Image, *, page_index: int = 0) -> list[Region]:
        _ = (image, page_index)
        return []


def load_layout_model(config: LayoutModelConfig | None = None) -> LayoutModelBackend:
    """Resolve a layout-model backend from config.

    Returns ``NullLayoutModel`` when disabled, when no backend is named, or
    when the named backend cannot be imported. Adding a real model stack is a
    human-approved dependency change (see task brief: stop and ask).
    """
    cfg = config if config is not None else LayoutModelConfig()
    if not cfg.enabled or not cfg.backend:
        return NullLayoutModel()

    # Intentionally no built-in heavy backends yet — avoid undeclared deps.
    # A future backend would be imported lazily by module path here.
    _ = cfg.backend
    return NullLayoutModel()
