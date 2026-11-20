"""Orchestrate perspective → deskew → crop (P2.3).

Each step is independently switchable via ``PreprocessConfig``. Steps that
do not fire leave the page unchanged for that stage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from torus.config.schemas import PreprocessConfig
from torus.core.models import Page
from torus.preprocess.crop import CropResult, apply_crop
from torus.preprocess.deskew import DeskewResult, apply_deskew
from torus.preprocess.perspective import PerspectiveResult, apply_perspective


class GeometryResult(BaseModel):
    """Combined geometry clean-up outcome for one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    perspective: PerspectiveResult
    deskew: DeskewResult
    crop: CropResult

    @property
    def any_applied(self) -> bool:
        return self.perspective.applied or self.deskew.applied or self.crop.applied


def correct_geometry(
    page: Page,
    config: PreprocessConfig | None = None,
) -> GeometryResult:
    """Run photo perspective, deskew, then border crop on ``page`` in place."""
    cfg = config if config is not None else PreprocessConfig()
    perspective = apply_perspective(page, config=cfg.perspective)
    deskew = apply_deskew(page, config=cfg.deskew)
    crop = apply_crop(page, config=cfg.crop)
    return GeometryResult(perspective=perspective, deskew=deskew, crop=crop)
