"""docTR printed-text reader (detection + recognition).

``doctr`` / ``torch`` are imported on first use only. docTR returns words with
a box and one score per word, but no per-character scores, so
``char_confidences`` stays ``None``.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReadHint, Region
from torus.readers.classic.common import (
    ClassicReaderBase,
    CropView,
    OcrWord,
    crop_region,
    local_box_to_bbox,
    unit_score,
)

DEFAULT_DET_ARCH = "fast_base"
DEFAULT_RECO_ARCH = "crnn_vgg16_bn"


def geometry_to_word(
    value: str,
    score: float | None,
    geometry: Any,
    view: CropView,
    line: int,
) -> OcrWord | None:
    """Build one ``OcrWord`` from a docTR word (geometry is relative to the crop)."""
    text = value.strip()
    (rx0, ry0), (rx1, ry1) = geometry
    crop_w, crop_h = view.image.size
    bbox = local_box_to_bbox(
        (rx0 * crop_w, ry0 * crop_h, rx1 * crop_w, ry1 * crop_h), view
    )
    if not text or bbox is None:
        return None
    return OcrWord(text=text, bbox=bbox, confidence=unit_score(score), line=line)


class PrintedReader(ClassicReaderBase):
    """docTR ``ocr_predictor``. Loads on first read, once per instance."""

    default_name = "doctr_printed"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        det_arch: str = DEFAULT_DET_ARCH,
        reco_arch: str = DEFAULT_RECO_ARCH,
        predictor: Any | None = None,
    ) -> None:
        super().__init__(device=device, spec=spec)
        if predictor is None:
            for module in ("doctr", "torch"):
                if importlib.util.find_spec(module) is None:
                    msg = f"{module} is not installed (pip install -e '.[doctr]')"
                    raise ReaderError(msg)
        self._det_arch = det_arch
        self._reco_arch = reco_arch
        self._predictor = predictor  # injectable for tests
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        with self._lock:
            if self._predictor is None:
                from doctr.models import ocr_predictor  # heavy: lazy by design

                predictor = ocr_predictor(
                    det_arch=self._det_arch,
                    reco_arch=self._reco_arch,
                    pretrained=True,
                )
                self._predictor = (
                    predictor.cuda() if self.device == "cuda" else predictor
                )
            return self._predictor

    def read_words(self, image: Any, region: Region, hint: ReadHint) -> list[OcrWord]:
        # docTR decoding cannot be constrained from a hint; see paddle reader.
        _ = hint
        import numpy as np

        view = crop_region(image, region)
        document = self._get_predictor()([np.asarray(view.image)])

        words: list[OcrWord] = []
        line_no = 0
        for page in document.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        item = geometry_to_word(
                            word.value, word.confidence, word.geometry, view, line_no
                        )
                        if item is not None:
                            words.append(item)
                    line_no += 1
        return words


def create_reader(*, device: str, spec: ReaderSpec) -> PrintedReader:
    """Factory used by ``torus.readers.load``."""
    return PrintedReader(device=device, spec=spec)
