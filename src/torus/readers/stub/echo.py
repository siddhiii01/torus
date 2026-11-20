"""Minimal stub reader: returns empty text, never invents confidence.

Used by doctor / registry smoke paths. Real OCR/VLM readers land in later
P5.* tasks behind the same ``create_reader`` contract.
"""

from __future__ import annotations

from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.models import ReaderCaps, ReadHint, Reading, Region


class EchoReader:
    """Reader protocol stub. ``text`` stays empty; ``raw_score`` is None."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.device = device
        if spec is not None:
            self.name = spec.name
            self._version = spec.version or "0"
            self._caps = spec.capabilities
        else:
            self.name = name or "stub_echo"
            self._version = version or "0"
            self._caps = ReaderCaps(
                scripts=["latin"],
                handwriting=False,
                printed=True,
                layout=False,
                per_char_confidence=False,
            )

    def capabilities(self) -> ReaderCaps:
        return self._caps

    def read(self, image: Any, region: Region, hint: ReadHint) -> Reading:
        _ = image, hint
        return Reading(
            region_id=region.id,
            reader_name=self.name,
            reader_version=self._version,
            text="",
            raw_score=None,
            char_confidences=None,
        )


def create_reader(*, device: str, spec: ReaderSpec) -> EchoReader:
    """Factory used by ``torus.readers.load``."""
    return EchoReader(device=device, spec=spec)
