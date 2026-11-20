"""Wall-clock timeout around a reader call (soft-fail, never crash)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout


def run_with_timeout[T](
    fn: Callable[[], T],
    *,
    timeout_s: float,
) -> T:
    """Run ``fn`` in a worker thread; raise ``TimeoutError`` on deadline.

    Stuck native code may keep the worker alive after the timeout; the caller
    should treat the call as failed and continue the pipeline.
    """
    if timeout_s <= 0:
        msg = f"timeout_s must be > 0, got {timeout_s}"
        raise ValueError(msg)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(f"reader call exceeded {timeout_s:.3f}s") from exc
