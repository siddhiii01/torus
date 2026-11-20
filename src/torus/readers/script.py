"""Script detection per region and routing to readers that support the script (P5.7).

Detection reads the Unicode block of the letters a probe reader produced, so it
needs no extra model: any enabled reader can serve as the probe. The block
table is a public standard, not document knowledge. The "confidence" is the
measured share of letters in the dominant script, never a made-up formula, and
detection returns ``None`` when there is too little text to say.

Routing only filters by ``ReaderCaps.scripts`` from ``config/readers.yaml``.
An unknown script does not filter (fewer assumptions, same voters); a script no
available reader supports yields an empty route so the caller can send the
region to review instead of guessing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from torus.core.models import ReaderCaps, ReadHint, Region
from torus.core.protocols import Reader
from torus.readers.registry import ReaderRegistry

# (first, last, script) code-point ranges; script names are lower-case and match
# the entries in ``ReaderCaps.scripts``.
_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x024F, "latin"),
    (0x1E00, 0x1EFF, "latin"),
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x052F, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0xA8E0, 0xA8FF, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "odia"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x30FF, "japanese"),
    (0x4E00, 0x9FFF, "han"),
    (0xAC00, 0xD7AF, "hangul"),
)

# Fewer letters than this is not evidence of a script (a lone mark or digit).
DEFAULT_MIN_LETTERS = 2


@dataclass(frozen=True)
class ScriptGuess:
    """Dominant script of a region, or ``script=None`` when it cannot be said."""

    script: str | None
    share: float | None = None  # fraction of letters in ``script``, in [0, 1]
    letters: int = 0


def script_of_char(ch: str) -> str | None:
    """Script of one letter; ``None`` for digits, punctuation, spaces, symbols."""
    if not ch.isalpha():
        return None
    code = ord(ch)
    for first, last, script in _BLOCKS:
        if first <= code <= last:
            return script
    return None


def guess_script(text: str, *, min_letters: int = DEFAULT_MIN_LETTERS) -> ScriptGuess:
    """Dominant script by letter count.

    Ties and too-few-letters return ``script=None`` rather than picking one.
    """
    counts = Counter(s for s in map(script_of_char, text) if s is not None)
    total = sum(counts.values())
    if total < max(1, min_letters):
        return ScriptGuess(script=None, letters=total)
    ranked = counts.most_common(2)
    if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
        return ScriptGuess(script=None, letters=total)
    script, top = ranked[0]
    return ScriptGuess(script=script, share=top / total, letters=total)


class ScriptDetector(Protocol):
    def detect(self, image: Any, region: Region) -> ScriptGuess: ...


class ProbeScriptDetector:
    """Detects script from the text a probe reader returns for the region.

    A probe that fails or returns nothing gives ``script=None``: unknown, never
    a default. The probe should be a reader that tolerates unfamiliar scripts.
    """

    def __init__(self, probe: Reader, *, min_letters: int = DEFAULT_MIN_LETTERS):
        self._probe = probe
        self._min_letters = min_letters

    def detect(self, image: Any, region: Region) -> ScriptGuess:
        reading = self._probe.read(image, region, ReadHint())
        if reading.error is not None or not reading.text:
            return ScriptGuess(script=None)
        return guess_script(reading.text, min_letters=self._min_letters)


@dataclass(frozen=True)
class Route:
    """Which readers to run for a region, and why."""

    script: str | None
    readers: tuple[str, ...]
    reason: str  # metadata only; never document text


def supports_script(caps: ReaderCaps, script: str) -> bool:
    return script.lower() in {s.lower() for s in caps.scripts}


def select_by_script(
    candidates: Sequence[tuple[str, ReaderCaps]], script: str | None
) -> Route:
    """Pure routing rule over ``(name, caps)`` pairs, preserving their order."""
    names = tuple(name for name, _ in candidates)
    if script is None:
        return Route(None, names, "script unknown: no filtering")
    key = script.lower()
    chosen = tuple(name for name, caps in candidates if supports_script(caps, key))
    if not chosen:
        return Route(key, (), f"no available reader supports script {key!r}")
    return Route(key, chosen, f"readers supporting script {key!r}")


def route_readers(registry: ReaderRegistry, script: str | None) -> Route:
    """Route among the registry's available readers (no model is loaded)."""
    candidates: list[tuple[str, ReaderCaps]] = []
    for name in registry.available_names():
        entry = registry.get_entry(name)
        if entry is not None:
            candidates.append((name, entry.spec.capabilities))
    return select_by_script(candidates, script)


def hint_for_script(hint: ReadHint, script: str | None) -> ReadHint:
    """``hint`` with ``script`` filled in; an explicit field-spec script wins."""
    if script is None or hint.script is not None:
        return hint
    return hint.model_copy(update={"script": script.lower()})
