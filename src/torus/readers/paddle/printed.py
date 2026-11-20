"""PaddleOCR (PP-OCRv5 detection + recognition) printed-text reader.

``paddle`` / ``paddleocr`` are imported on first use only. PaddleOCR segments
text into lines, not words, so each recognised line becomes one ``OcrWord``
(its text is never split here). It reports one score per line and no
per-character scores, so ``char_confidences`` stays ``None``.
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

DEFAULT_DET_MODEL = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL = "PP-OCRv5_server_rec"


def line_to_word(
    text: str,
    score: float | None,
    box: tuple[float, float, float, float],
    view: CropView,
    line: int,
) -> OcrWord | None:
    """Build one ``OcrWord`` from a recognised line; ``None`` if empty/degenerate."""
    text = text.strip()
    bbox = local_box_to_bbox(box, view)
    if not text or bbox is None:
        return None
    return OcrWord(text=text, bbox=bbox, confidence=unit_score(score), line=line)


class PrintedReader(ClassicReaderBase):
    """PP-OCRv5 via ``paddleocr``. Engine loads on first read, once per instance."""

    default_name = "paddle_printed"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        det_model: str = DEFAULT_DET_MODEL,
        rec_model: str = DEFAULT_REC_MODEL,
        engine: Any | None = None,
    ) -> None:
        super().__init__(device=device, spec=spec)
        if engine is None:
            for module in ("paddle", "paddleocr"):
                if importlib.util.find_spec(module) is None:
                    msg = f"{module} is not installed (pip install -e '.[paddle]')"
                    raise ReaderError(msg)
        self._det_model = det_model
        self._rec_model = rec_model
        self._engine = engine  # injectable for tests
        self._lock = threading.Lock()

    def _get_engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                from paddleocr import PaddleOCR  # heavy: lazy by design

                self._engine = PaddleOCR(
                    text_detection_model_name=self._det_model,
                    text_recognition_model_name=self._rec_model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="gpu:0" if self.device == "cuda" else "cpu",
                )
            return self._engine

    def read_words(self, image: Any, region: Region, hint: ReadHint) -> list[OcrWord]:
        # The hint cannot constrain PP-OCR decoding; the family reads unconstrained
        # and downstream validators/voting decide. No post-hoc character filtering.
        _ = hint
        import numpy as np

        view = crop_region(image, region)
        pixels = np.asarray(view.image)[:, :, ::-1]  # PaddleOCR expects BGR
        results = self._get_engine().predict(np.ascontiguousarray(pixels))

        words: list[OcrWord] = []
        for page in results:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            boxes = page["rec_boxes"]  # (N, 4) xyxy, crop pixels
            for idx, text in enumerate(texts):
                x0, y0, x1, y1 = (float(v) for v in boxes[idx])
                word = line_to_word(
                    str(text), float(scores[idx]), (x0, y0, x1, y1), view, idx
                )
                if word is not None:
                    words.append(word)
        return words


def create_reader(*, device: str, spec: ReaderSpec) -> PrintedReader:
    """Factory used by ``torus.readers.load``."""
    return PrintedReader(device=device, spec=spec)
