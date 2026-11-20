"""Character classes and ReadHint charset masking for the cell classifier."""

from __future__ import annotations

from torus.core.types import Charset

# Digits then uppercase Latin letters. Lowercase EMNIST labels map up here.
DIGITS: tuple[str, ...] = tuple(str(d) for d in range(10))
LETTERS: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26))
DEFAULT_CLASSES: tuple[str, ...] = DIGITS + LETTERS

_DIGIT_SET = frozenset(DIGITS)
_LETTER_SET = frozenset(LETTERS)
_ALNUM_SET = frozenset(DEFAULT_CLASSES)


def allowed_indices(
    classes: tuple[str, ...] | list[str], charset: Charset
) -> list[int]:
    """Return indices into ``classes`` that are legal for ``charset``.

    ``Charset.ANY`` keeps every class. Unknown charset values fall back to all.
    """
    if charset is Charset.DIGITS:
        allowed = _DIGIT_SET
    elif charset is Charset.LETTERS:
        allowed = _LETTER_SET
    elif charset is Charset.ALPHANUMERIC:
        allowed = _ALNUM_SET
    else:
        return list(range(len(classes)))

    return [i for i, ch in enumerate(classes) if ch in allowed]


def normalize_label(raw: str) -> str | None:
    """Map a single-character label into ``DEFAULT_CLASSES``, or ``None``."""
    if len(raw) != 1:
        return None
    ch = raw.upper()
    if ch in _ALNUM_SET:
        return ch
    return None
