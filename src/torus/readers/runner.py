"""Run readers with timeout, crop-hash cache, and optional batching."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from torus.config.schemas import ReadersConfig
from torus.core.models import ReadHint, Reading, Region
from torus.core.protocols import Reader
from torus.logging import get_logger
from torus.readers.cache import ReadingCache
from torus.readers.crop_hash import cache_key, crop_hash
from torus.readers.registry import ReaderRegistry
from torus.readers.timeout import run_with_timeout

log = get_logger(__name__)


class ReaderRunner:
    """Orchestrates registry lookup → cache → timed read → optional batch."""

    def __init__(
        self,
        registry: ReaderRegistry,
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
        cache: ReadingCache | None = None,
    ) -> None:
        cfg: ReadersConfig = registry.config
        self._registry = registry
        self._timeout_s = float(timeout_s) if timeout_s is not None else cfg.timeout_s
        self._batch_size = int(batch_size) if batch_size is not None else cfg.batch_size
        if cache is not None:
            self._cache = cache
        else:
            self._cache = ReadingCache(cfg.cache_max_entries)

    @property
    def registry(self) -> ReaderRegistry:
        return self._registry

    @property
    def cache(self) -> ReadingCache:
        return self._cache

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def read(
        self,
        reader_name: str,
        image: Any,
        region: Region,
        hint: ReadHint | None = None,
    ) -> Reading:
        """Read one region. Soft-failures become ``Reading.error``."""
        rh = hint if hint is not None else ReadHint()
        entry = self._registry.get_entry(reader_name)
        if entry is None:
            return _error_reading(
                region_id=region.id,
                reader_name=reader_name,
                reader_version="unknown",
                error=f"unknown reader {reader_name!r}",
            )
        version = entry.spec.version or "0"
        if not entry.available:
            return _error_reading(
                region_id=region.id,
                reader_name=reader_name,
                reader_version=version,
                error=entry.skip_reason or "reader unavailable",
            )

        digest = crop_hash(image, region)
        key = cache_key(
            crop_digest=digest,
            reader_name=reader_name,
            reader_version=version,
            hint=rh,
        )
        cached = self._cache.get(key)
        if cached is not None:
            log.info(
                "reader_cache_hit",
                reader_name=reader_name,
                reader_version=version,
                region_id=region.id,
                status_code="cache_hit",
            )
            return cached.model_copy(update={"region_id": region.id})

        reader = entry.get()
        if reader is None:
            return _error_reading(
                region_id=region.id,
                reader_name=reader_name,
                reader_version=version,
                error=entry.skip_reason or "reader failed to load",
            )

        reading = self._timed_read(reader, image, region, rh, version=version)
        self._cache.put(key, reading)
        return reading

    def read_many(
        self,
        reader_name: str,
        image: Any,
        regions: Sequence[Region],
        hint: ReadHint | None = None,
    ) -> list[Reading]:
        """Read many regions; use ``read_batch`` when the plugin exposes it."""
        rh = hint if hint is not None else ReadHint()
        if not regions:
            return []

        entry = self._registry.get_entry(reader_name)
        if entry is None or not entry.available:
            version = entry.spec.version if entry is not None else "unknown"
            version = version or "0"
            err = (
                f"unknown reader {reader_name!r}"
                if entry is None
                else (entry.skip_reason or "reader unavailable")
            )
            return [
                _error_reading(
                    region_id=r.id,
                    reader_name=reader_name,
                    reader_version=version,
                    error=err,
                )
                for r in regions
            ]

        reader = entry.get()
        version = entry.spec.version or "0"
        if reader is None:
            err = entry.skip_reason or "reader failed to load"
            return [
                _error_reading(
                    region_id=r.id,
                    reader_name=reader_name,
                    reader_version=version,
                    error=err,
                )
                for r in regions
            ]

        # Fill from cache first; collect misses in order.
        out: list[Reading | None] = [None] * len(regions)
        misses: list[tuple[int, Region]] = []
        for idx, region in enumerate(regions):
            digest = crop_hash(image, region)
            key = cache_key(
                crop_digest=digest,
                reader_name=reader_name,
                reader_version=version,
                hint=rh,
            )
            cached = self._cache.get(key)
            if cached is not None:
                out[idx] = cached.model_copy(update={"region_id": region.id})
            else:
                misses.append((idx, region))

        if not misses:
            return [r for r in out if r is not None]

        batch_fn = getattr(reader, "read_batch", None)
        if callable(batch_fn):
            self._fill_via_batch(
                reader,
                batch_fn,
                image,
                misses,
                rh,
                version=version,
                reader_name=reader_name,
                out=out,
            )
        else:
            for idx, region in misses:
                reading = self._timed_read(reader, image, region, rh, version=version)
                key = cache_key(
                    crop_digest=crop_hash(image, region),
                    reader_name=reader_name,
                    reader_version=version,
                    hint=rh,
                )
                self._cache.put(key, reading)
                out[idx] = reading

        return [
            r
            if r is not None
            else _error_reading(
                region_id=regions[i].id,
                reader_name=reader_name,
                reader_version=version,
                error="internal: missing reading",
            )
            for i, r in enumerate(out)
        ]

    def _fill_via_batch(
        self,
        reader: Reader,
        batch_fn: Any,
        image: Any,
        misses: list[tuple[int, Region]],
        hint: ReadHint,
        *,
        version: str,
        reader_name: str,
        out: list[Reading | None],
    ) -> None:
        size = self._batch_size
        for start in range(0, len(misses), size):
            chunk = misses[start : start + size]
            chunk_regions = [r for _, r in chunk]

            def _call(
                regions: list[Region] = chunk_regions,
            ) -> list[Reading]:
                result = batch_fn(image, regions, hint)
                if not isinstance(result, list):
                    msg = "read_batch must return list[Reading]"
                    raise TypeError(msg)
                return result

            t0 = time.perf_counter()
            try:
                readings = run_with_timeout(_call, timeout_s=self._timeout_s)
            except TimeoutError as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                for idx, region in chunk:
                    out[idx] = _error_reading(
                        region_id=region.id,
                        reader_name=reader_name,
                        reader_version=version,
                        error=str(exc),
                        time_ms=elapsed,
                    )
                continue
            except Exception as exc:  # noqa: BLE001 — soft fail
                elapsed = (time.perf_counter() - t0) * 1000.0
                for idx, region in chunk:
                    out[idx] = _error_reading(
                        region_id=region.id,
                        reader_name=reader_name,
                        reader_version=version,
                        error=f"{type(exc).__name__}: {exc}",
                        time_ms=elapsed,
                    )
                continue

            elapsed = (time.perf_counter() - t0) * 1000.0
            if len(readings) != len(chunk):
                for idx, region in chunk:
                    out[idx] = _error_reading(
                        region_id=region.id,
                        reader_name=reader_name,
                        reader_version=version,
                        error=(
                            "read_batch length mismatch: "
                            f"got {len(readings)} for {len(chunk)} regions"
                        ),
                        time_ms=elapsed,
                    )
                continue

            for (idx, region), reading in zip(chunk, readings, strict=True):
                if reading.time_ms <= 0:
                    reading = reading.model_copy(update={"time_ms": elapsed})
                key = cache_key(
                    crop_digest=crop_hash(image, region),
                    reader_name=reader_name,
                    reader_version=version,
                    hint=hint,
                )
                self._cache.put(key, reading)
                out[idx] = reading

    def _timed_read(
        self,
        reader: Reader,
        image: Any,
        region: Region,
        hint: ReadHint,
        *,
        version: str,
    ) -> Reading:
        name = getattr(reader, "name", "unknown")

        def _call() -> Reading:
            return reader.read(image, region, hint)

        t0 = time.perf_counter()
        try:
            reading = run_with_timeout(_call, timeout_s=self._timeout_s)
        except TimeoutError as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            log.info(
                "reader_timeout",
                reader_name=name,
                reader_version=version,
                region_id=region.id,
                duration_ms=elapsed,
                status_code="timeout",
            )
            return _error_reading(
                region_id=region.id,
                reader_name=name,
                reader_version=version,
                error=str(exc),
                time_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 — soft fail
            elapsed = (time.perf_counter() - t0) * 1000.0
            log.info(
                "reader_error",
                reader_name=name,
                reader_version=version,
                region_id=region.id,
                duration_ms=elapsed,
                error_type=type(exc).__name__,
                status_code="error",
            )
            return _error_reading(
                region_id=region.id,
                reader_name=name,
                reader_version=version,
                error=f"{type(exc).__name__}: {exc}",
                time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        if reading.time_ms <= 0:
            reading = reading.model_copy(update={"time_ms": elapsed})
        log.info(
            "reader_finished",
            reader_name=name,
            reader_version=version,
            region_id=region.id,
            duration_ms=reading.time_ms,
            status_code="ok" if reading.error is None else "soft_error",
        )
        return reading


def _error_reading(
    *,
    region_id: str,
    reader_name: str,
    reader_version: str,
    error: str,
    time_ms: float = 0.0,
) -> Reading:
    return Reading(
        region_id=region_id,
        reader_name=reader_name,
        reader_version=reader_version,
        text="",
        raw_score=None,
        char_confidences=None,
        time_ms=time_ms,
        error=error,
    )
