"""LRU cache of readings keyed by crop hash + reader version (+ hint)."""

from __future__ import annotations

from collections import OrderedDict

from torus.core.models import Reading


class ReadingCache:
    """In-process LRU. ``max_entries=0`` disables storage."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 0:
            msg = f"max_entries must be >= 0, got {max_entries}"
            raise ValueError(msg)
        self._max = max_entries
        self._entries: OrderedDict[str, Reading] = OrderedDict()

    @property
    def max_entries(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def get(self, key: str) -> Reading | None:
        if self._max == 0:
            return None
        reading = self._entries.get(key)
        if reading is None:
            return None
        self._entries.move_to_end(key)
        return reading

    def put(self, key: str, reading: Reading) -> None:
        if self._max == 0:
            return
        # Do not cache soft failures: retries / different loads may succeed.
        if reading.error is not None:
            return
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = reading
            return
        self._entries[key] = reading
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
