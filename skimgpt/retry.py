"""Reusable retry helper with configurable backoff and error handling.

Zero project dependencies — this module relies only on the Python stdlib.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float | None = None,
    non_retryable: tuple[type[BaseException], ...] = (),
    on_non_retryable: Callable[[BaseException, int], None] | None = None,
    on_retryable: Callable[[BaseException, int], None] | None = None,
    default: T = None,
) -> T:
    """Call *fn* up to *max_retries* times, returning the first successful result.

    Args:
        fn: Zero-argument callable to invoke on each attempt.
        max_retries: Total number of attempts (not *additional* retries).
        delay: Base sleep duration between retries (seconds).
        backoff_factor: If provided, sleep = delay * backoff_factor ** (attempt - 1)
            (exponential backoff).  If *None*, sleep is constant at *delay*.
        non_retryable: Exception types that should abort the loop immediately.
        on_non_retryable: Optional callback invoked with ``(exc, attempt)`` when a
            non-retryable exception is caught.  For logging or cleanup.
        on_retryable: Optional callback invoked with ``(exc, attempt)`` on every
            retryable failure.  The callback may raise ``StopIteration`` to abort
            the retry loop early (e.g. for specific status codes).
        default: Value returned when all attempts are exhausted.

    Returns:
        The return value of *fn*, or *default* if all attempts fail.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()

        except non_retryable as exc:
            if on_non_retryable is not None:
                on_non_retryable(exc, attempt)
            return default

        except Exception as exc:
            last_exc = exc
            if on_retryable is not None:
                try:
                    on_retryable(exc, attempt)
                except StopIteration:
                    return default

            if attempt < max_retries:
                if backoff_factor is not None:
                    sleep_time = delay * (backoff_factor ** (attempt - 1))
                else:
                    sleep_time = delay
                logger.debug(
                    "retry_call: attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt, max_retries, exc, sleep_time,
                )
                time.sleep(sleep_time)

    if last_exc is not None:
        logger.warning(
            "retry_call: all %d attempts exhausted, last error: %s",
            max_retries, last_exc,
        )
    return default
