"""Provider-neutral request engine: retry driver, failover loop, stream bridge.

No provider imports (no DeepSeek client/protocol/PoW), no ``deepseaport.server``
imports, no ``_server_override`` lookups. Providers supply small adapters
(predicates, recover callbacks, run-once closures); the engine owns only the
generic mechanics: sequential retry tables, account-slot release/acquire
failover, cancellation checks, and producer-thread bridging.

DeepSeek keeps session/PoW/SSE/ban + token-refresh in
``providers.deepseek.completion_service`` and delegates its loops here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger("deepseaport.core.engine")


def drive_retries(
    max_retries: int,
    is_cancelled: Callable[[], bool],
    get_error: Callable[[], str | None],
    handlers: Sequence[tuple[str, Callable[[str | None], bool], Callable[[], None]]],
) -> None:
    """Run an ordered retry table to fixpoint (one pass per kind, in order).

    Each handler is ``(name, should_retry, recover)``. For each handler in
    order, while not cancelled and ``should_retry(current_error)`` holds,
    call ``recover()`` (which must refresh the error via a new attempt), up
    to ``max_retries`` times. Mirrors the historical five sequential string
    loops (pow -> invalid-session -> invalid-token -> waf -> overload) with
    identical ``max_retries``-per-kind semantics; a later handler sees the
    error left by the earlier one.
    """
    try:
        budget = max(0, int(max_retries))
    except Exception:
        budget = 0
    for _name, should_retry, recover in handlers:
        for _ in range(budget):
            try:
                if is_cancelled():
                    return
            except Exception:
                return
            try:
                err = get_error()
            except Exception:
                return
            try:
                want = bool(should_retry(err))
            except Exception:
                return
            if not want:
                break
            recover()


def run_with_failover(
    pool,
    first_item,
    run_once: Callable[[object], object],
    is_retryable_exc: Callable[[BaseException], bool],
    on_before_retry: Callable[[BaseException, object, int, int], None] | None = None,
    acquire_timeout: float = 10,
    is_cancelled: Callable[[], bool] | None = None,
    allow_failover: bool = True,
    release: Callable[[object], None] | None = None,
    acquire: Callable[[float], object] | None = None,
) -> object:
    """Run ``run_once(item)`` on ``first_item``, failing over on retryable errors.

    Owns each attempt's slot: releases exactly once per attempt (success,
    retryable, or fatal), acquires the next account before retrying, and
    re-raises the first error when no healthy account remains. ``run_once``
    performs one attempt (including any per-kind retries); the engine only
    handles cross-account failover, cancellation, and slot lifecycle.
    """
    if release is None:
        def _default_release(item) -> None:
            try:
                from ..accounts import AccountPool as _Pool

                _Pool.release(item)
            except Exception:
                try:
                    rel = getattr(item, "lock", None)
                    if rel is not None:
                        rel.release()
                except Exception:
                    pass

        release = _default_release
    if acquire is None:
        def _default_acquire(timeout: float):
            return pool.acquire(timeout=timeout)

        acquire = _default_acquire

    if not allow_failover:
        try:
            return run_once(first_item)
        finally:
            try:
                release(first_item)
            except Exception:
                pass

    try:
        max_tries = max(1, len(pool))
    except Exception:
        max_tries = 1
    cur = first_item
    first_exc: BaseException | None = None
    tries = 0
    while True:
        try:
            result = run_once(cur)
        except Exception as exc:
            try:
                release(cur)
            except Exception:
                pass
            retryable = False
            try:
                retryable = bool(is_retryable_exc(exc))
            except Exception:
                retryable = False
            cancelled = False
            try:
                cancelled = bool(is_cancelled()) if is_cancelled is not None else False
            except Exception:
                cancelled = False
            if not retryable or cancelled or tries + 1 >= max_tries:
                raise
            if first_exc is None:
                first_exc = exc
            tries += 1
            if on_before_retry is not None:
                try:
                    on_before_retry(exc, cur, tries, max_tries)
                except Exception:
                    pass
            try:
                cur = acquire(float(acquire_timeout))
            except Exception:
                assert first_exc is not None
                raise first_exc
            continue
        try:
            release(cur)
        except Exception:
            pass
        return result
