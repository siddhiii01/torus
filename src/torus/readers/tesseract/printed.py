"""Tesseract 5 reader. Runs the system binary; no Python package is needed.

The crop goes to the process over stdin and hOCR comes back over stdout, so no
document pixels or text touch the disk. hOCR with ``hocr_char_boxes`` gives word
boxes, word scores and per-character scores in one pass.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from typing import Any

from torus.config.schemas import ReaderSpec
from torus.core.errors import ReaderError
from torus.core.models import ReadHint, Region
from torus.core.types import Charset, RegionKind
from torus.readers.classic.common import (
    ClassicReaderBase,
    CropView,
    OcrWord,
    crop_region,
    local_box_to_bbox,
    unit_score,
)

_ENV_CMD = "TESSERACT_CMD"
_DEFAULT_LANG = "eng"
_DEFAULT_PROCESS_TIMEOUT_S = 60.0

# Page segmentation: single text line vs. uniform block of text.
_PSM_LINE = 7
_PSM_BLOCK = 6

_DIGITS = "0123456789"
_LATIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_BBOX_RE = re.compile(r"bbox\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
_CONF_RE = re.compile(r"x_(?:w)?conf\s+(-?[\d.]+)")
_LINE_CLASSES = frozenset(
    {"ocr_line", "ocr_header", "ocr_footer", "ocr_caption", "ocr_textfloat"}
)


def find_binary() -> str | None:
    """Locate the tesseract executable: ``TESSERACT_CMD`` first, then ``PATH``."""
    override = os.environ.get(_ENV_CMD)
    if override:
        return override if shutil.which(override) else None
    return shutil.which("tesseract")


def whitelist_for(hint: ReadHint) -> str | None:
    """Decoding constraint from the hint, or ``None`` when it cannot be expressed.

    Letters are only constrained when the script is explicitly Latin: an ASCII
    whitelist would silently exclude other scripts and accented letters.
    """
    latin = (hint.script or "").lower() == "latin"
    if hint.charset is Charset.DIGITS:
        return _DIGITS
    if hint.charset is Charset.LETTERS and latin:
        return _LATIN_LETTERS
    if hint.charset is Charset.ALPHANUMERIC and latin:
        return _DIGITS + _LATIN_LETTERS
    return None


class _HocrParser(HTMLParser):
    """Collect words (box, score, per-character scores) from Tesseract hOCR."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.words: list[dict[str, Any]] = []
        self._line = -1
        self._word: dict[str, Any] | None = None
        self._word_depth = 0
        self._cinfo: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            if self._word is not None and tag != "br":
                self._word_depth += 1
            return
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        title = attr.get("title") or ""
        if any(c in _LINE_CLASSES for c in classes):
            self._line += 1
        if "ocrx_word" in classes:
            box = _BBOX_RE.search(title)
            conf = _CONF_RE.search(title)
            self._word = {
                "box": tuple(float(v) for v in box.groups()) if box else None,
                "conf": float(conf.group(1)) if conf else None,
                "text": [],
                "chars": [],
                "line": max(self._line, 0),
            }
            self._word_depth = 1
            return
        if self._word is not None:
            self._word_depth += 1
            if "ocrx_cinfo" in classes:
                conf = _CONF_RE.search(title)
                self._cinfo = {
                    "conf": float(conf.group(1)) if conf else None,
                    "text": [],
                }

    def handle_endtag(self, tag: str) -> None:
        if self._word is None:
            return
        if self._cinfo is not None and tag == "span":
            text = "".join(self._cinfo["text"])
            self._word["chars"].append((text, self._cinfo["conf"]))
            self._cinfo = None
        self._word_depth -= 1
        if self._word_depth <= 0:
            self.words.append(self._word)
            self._word = None

    def handle_data(self, data: str) -> None:
        if self._word is None:
            return
        if self._cinfo is not None:
            self._cinfo["text"].append(data)
        else:
            self._word["text"].append(data)


def parse_hocr(hocr: str, view: CropView) -> list[OcrWord]:
    """Turn hOCR into ``OcrWord`` objects mapped to page coordinates."""
    parser = _HocrParser()
    parser.feed(hocr)
    parser.close()

    words: list[OcrWord] = []
    for raw in parser.words:
        chars: list[tuple[str, float | None]] = raw["chars"]
        text = "".join(c for c, _ in chars) if chars else "".join(raw["text"])
        text = text.strip()
        if not text or raw["box"] is None:
            continue
        bbox = local_box_to_bbox(raw["box"], view)
        if bbox is None:
            continue
        per_char: tuple[float, ...] | None = None
        if chars and len(chars) == len(text):
            scores = [unit_score(s, scale=100.0) for _, s in chars]
            if all(s is not None for s in scores):
                per_char = tuple(s for s in scores if s is not None)
        words.append(
            OcrWord(
                text=text,
                bbox=bbox,
                confidence=unit_score(raw["conf"], scale=100.0),
                char_confidences=per_char,
                line=raw["line"],
            )
        )
    return words


class PrintedReader(ClassicReaderBase):
    """Tesseract via subprocess.

    ``spec.model`` is a weights path for the doctor check, so it is not used for
    the language. The language comes from ``ReadHint.language`` when it names an
    installed pack, otherwise the default.
    """

    default_name = "tesseract_printed"

    def __init__(
        self,
        *,
        device: str = "cpu",
        spec: ReaderSpec | None = None,
        process_timeout_s: float = _DEFAULT_PROCESS_TIMEOUT_S,
    ) -> None:
        super().__init__(device="cpu", spec=spec)  # Tesseract is CPU-only.
        binary = find_binary()
        if binary is None:
            msg = f"tesseract executable not found (set {_ENV_CMD} or add to PATH)"
            raise ReaderError(msg)
        self._binary = binary
        self._process_timeout_s = process_timeout_s
        self._installed: frozenset[str] | None = None

    def _installed_langs(self) -> frozenset[str]:
        if self._installed is None:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [self._binary, "--list-langs"],
                capture_output=True,
                timeout=self._process_timeout_s,
                check=False,
            )
            lines = proc.stdout.decode("utf-8", "replace").splitlines()[1:]
            self._installed = frozenset(line.strip() for line in lines if line.strip())
        return self._installed

    def _language(self, hint: ReadHint) -> str:
        if hint.language and hint.language in self._installed_langs():
            return hint.language
        return _DEFAULT_LANG

    def _command(self, hint: ReadHint, region: Region) -> list[str]:
        psm = _PSM_LINE if region.kind is RegionKind.TEXT_LINE else _PSM_BLOCK
        cmd = [
            self._binary,
            "stdin",
            "stdout",
            "-l",
            self._language(hint),
            "--psm",
            str(psm),
            "-c",
            "hocr_char_boxes=1",
        ]
        allowed = whitelist_for(hint)
        if allowed is not None:
            cmd += ["-c", f"tessedit_char_whitelist={allowed}"]
        cmd.append("hocr")
        return cmd

    def read_words(self, image: Any, region: Region, hint: ReadHint) -> list[OcrWord]:
        view = crop_region(image, region)
        buffer = io.BytesIO()
        view.image.save(buffer, format="PNG")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                self._command(hint, region),
                input=buffer.getvalue(),
                capture_output=True,
                timeout=self._process_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"tesseract exceeded {self._process_timeout_s}s"
            raise TimeoutError(msg) from exc
        if proc.returncode != 0:
            # stderr may echo engine messages, never document text; keep it short.
            detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
            msg = f"tesseract exited with code {proc.returncode}: {detail}"
            raise ReaderError(msg)
        return parse_hocr(proc.stdout.decode("utf-8", "replace"), view)


def create_reader(*, device: str, spec: ReaderSpec) -> PrintedReader:
    """Factory used by ``torus.readers.load``."""
    return PrintedReader(device=device, spec=spec)
