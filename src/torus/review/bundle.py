"""Review bundle: what one pipeline run hands to the reviewer (P9.1).

A bundle is the ``ExtractionResult``, every ``Reading`` behind it and the source
path (page images are re-rendered from it on demand, so no page pixels or crops
are stored). Bundles hold document content, so they live under
``TORUS_DATA_DIR`` and never in git.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.core.models import ExtractionResult, Reading
from torus.review.queue import ReviewError


class ReviewBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: ExtractionResult
    readings: list[Reading] = Field(default_factory=list)
    source_path: str = Field(min_length=1)


def save_bundle(bundle: ReviewBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")


def load_bundle(path: Path) -> ReviewBundle:
    """Load a bundle; errors name the file only, never its content."""
    if not path.is_file():
        raise ReviewError(f"bundle not found: {path.name}")
    try:
        return ReviewBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise ReviewError(f"invalid review bundle: {path.name}") from exc
