"""Server concurrency primitives and process lifecycle.

Pure moves from deepseaport.server: the shared thread-pool executors, the
queued-slot acquisition / client-disconnect helpers, and the FastAPI
lifespan plus startup warmup. No DeepSeek completion logic lives here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .accounts import pool_provider_label
from .config import Settings
from .core.obscura_bridge import ObscuraBridge, register_default_bridge

logger = logging.getLogger("deepseaport.server")


class _DaemonExecutor:
    """``submit() -> Future`` over a bounded set of DAEMON threads.

    ``ThreadPoolExecutor`` cannot be a daemon pool, and the three executors
    below are exactly the workers that must not be able to hold the process:

    * it has no daemon option at all. ``initializer`` runs *inside* the worker
      after it is started, and ``Thread.daemon`` refuses to change once a thread
      is alive, so there is no hook that works.
    * ``Thread.__init__`` otherwise inherits ``current_thread().daemon``, so a
      worker's daemon-ness depended on which thread submitted first -- a policy
      decided by an accident, and the reason ``lifespan``'s "Threads exit at
      process end" was not a statement anyone could rely on.

    Reach past the public API to fix that and you override
    ``_adjust_thread_count`` and import ``concurrent.futures.thread._worker``,
    ``_threads_queues`` and ``_shutdown`` -- stdlib privates that break silently
    on a CPython change. This is the same shape built from two public pieces
    (``concurrent.futures.Future`` and ``queue.SimpleQueue``) plus the one
    behaviour that is wanted. ``submit`` is all the three call sites use.

    What the daemon flag does NOT do, so nobody re-derives this wrongly:
    ``concurrent.futures.thread._python_exit`` (registered with
    ``threading._register_atexit``) joins every worker it created, daemon or
    not, so the flag alone does not rescue a worker PARKED inside an
    uninterruptible C call. What ends that is ``deepseek.client``'s bounded
    read handing the producer its thread back within a slice; this class makes
    the flag deterministic and keeps ``lifespan``'s comment true for every
    worker that is not parked.

    ponytail: no ``shutdown()``, no ``map``, no cancel-futures. Add one if a
    caller ever needs it -- and note the module comment in ``lifespan`` already
    says these pools must outlive a lifespan.
    """

    def __init__(self, max_workers: int, thread_name_prefix: str = "") -> None:
        # ``_max_workers`` is the historical ThreadPoolExecutor attribute name
        # and tests/test_reliability.py asserts on it.
        self._max_workers = int(max_workers)
        self._thread_name_prefix = str(thread_name_prefix or "worker")
        self._queue: "_queue.SimpleQueue[tuple]" = _queue.SimpleQueue()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._pending = 0

    def submit(self, fn, *args, **kwargs) -> "Future":
        """Schedule ``fn(*args, **kwargs)``; returns a Future for its result.

        Grows the pool only while work is genuinely queued and unhandled, so a
        parallel-subagent burst still bounds at ``max_workers`` instead of
        piling one raw thread per submit -- the property these executors were
        introduced for.
        """
        fut: Future = Future()
        with self._lock:
            self._pending += 1
            if (len(self._threads) < self._max_workers
                    and self._pending > len(self._threads)):
                t = threading.Thread(
                    target=self._work, daemon=True,
                    name=f"{self._thread_name_prefix}_{len(self._threads)}")
                self._threads.append(t)
                t.start()
        self._queue.put((fut, fn, args, kwargs))
        return fut

    def _work(self) -> None:
        while True:
            fut, fn, args, kwargs = self._queue.get()
            with self._lock:
                self._pending -= 1
            if not fut.set_running_or_notify_cancel():
                continue  # cancelled before pickup
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - Future's own contract
                fut.set_exception(exc)


# Shared executors: replacing per-request ThreadPoolExecutor/thread churn.
# Challenge work (create_session + fetch_pow) is I/O-bound; sized generously
# so concurrent completions never serialize (previously one 2-worker executor
# per request). Cleanup (delete_session) is fire-and-forget best-effort.
_DEFAULT_POOL_SIZE = min(32, (os.cpu_count() or 1) + 4)
_CHALLENGE_EXECUTOR = _DaemonExecutor(
    max_workers=2 * _DEFAULT_POOL_SIZE, thread_name_prefix="ds-challenge")
_CLEANUP_EXECUTOR = _DaemonExecutor(
    max_workers=_DEFAULT_POOL_SIZE, thread_name_prefix="ds-cleanup")
# Stream producers (one per streamed request, each holding an account slot +
# PoW CPU): bounded so a parallel-subagent burst queues instead of piling
# unbounded raw threads.
_STREAM_EXECUTOR = _DaemonExecutor(
    max_workers=_DEFAULT_POOL_SIZE, thread_name_prefix="ds-stream")

# Consumer wait (seconds) for a producer event, on top of the curl stream
# timeout. Keeps the async reader from timing out before the blocking call.
_CONSUMER_GRACE_SECONDS = 10

# Failover wait (seconds) for a free account when the current one turns out
# banned mid-request. Short: healthy accounts are normally idle; long waits
# would stall the request that already paid for a failed ban attempt.
FAILOVER_ACQUIRE_TIMEOUT = 10
# Poll interval for _await_cancelable: how long one fut.result() wait lasts
# before re-checking cancel_event. Short enough to abort promptly on client
# disconnect, long enough to avoid a hot spin on the challenge executor.
_CANCEL_POLL_SECONDS = 0.05
# A 401 means either a missing token or a browser refresh that could not
# produce a usable one.  Cool that account briefly so failover can move to a
# healthy account instead of immediately reacquiring the same broken one.
INVALID_TOKEN_COOLDOWN_SECONDS = 60


class _StreamCancelled(Exception):
    """Internal: client disconnected, producer should stop at the next event."""


async def _watch_disconnect(request: Request, cancel_event: threading.Event) -> None:
    """Poll Request.is_disconnected() into cancel_event (non-stream path).

    The stream path sets cancel_event from the response generator's finally;
    non-stream has no generator, so a background poller watches the client
    socket. When the client goes away the blocking completion aborts at the
    next setup checkpoint, and the SSE read notices within one read slice (see
    ``deepseek.client``'s bounded line pump) rather than at the next event, so
    the account slot comes back instead of being held to the end of the socket
    budget while the next request queues.
    """
    try:
        while not cancel_event.is_set():
            try:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
            except Exception:
                break
            await asyncio.sleep(0.15)
    except asyncio.CancelledError:
        pass


async def _acquire_slot_or_499(pool, request: Request,
                               cancel_event: threading.Event,
                               timeout: float, allow_failover: bool,
                               provider_label: str | None = None):
    """Acquire an account slot, aborting the queue wait on disconnect.

    aacquire() alone would park the endpoint task for the full 90s even
    after the client gave up. Chunk the wait so a disconnect surfaces as
    499 within ~2s without ever holding a slot.

    ``provider_label`` is optional and names the upstream in the 503/429
    detail ("no Qwen accounts in pool" instead of telling a Qwen-only
    operator to debug DeepSeek). Omitted -- the way every current call site
    calls it -- it is resolved from the pool itself
    (``AccountPool.provider_label``, derived from the member accounts'
    ``provider`` field), falling back to "DeepSeek". Status codes, ordering
    and the exported signature are unchanged.
    """
    from fastapi import HTTPException as _HTTPException

    label = str(provider_label or "").strip() or pool_provider_label(pool)
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        chunk = min(2.0, max(0.05, deadline - time.monotonic()))
        try:
            return await pool.aacquire(chunk, allow_failover=allow_failover)
        except ValueError:
            raise _HTTPException(status_code=503,
                                 detail=f"no {label} accounts in pool")
        except TimeoutError:
            if cancel_event.is_set():
                raise _HTTPException(status_code=499, detail="client disconnected")
            try:
                if await request.is_disconnected():
                    raise _HTTPException(status_code=499, detail="client disconnected")
            except _HTTPException:
                raise
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise _HTTPException(status_code=429,
                                     detail=f"all {label} accounts busy, retry later")
            continue


def _await_cancelable(fut, cancel_event: threading.Event | None):
    """Wait for a challenge-executor future, aborting promptly on disconnect.

    No cancel token: plain blocking wait (no 50ms poll). With a token:
    polls with a short timeout so a client disconnect (cancel_event set by
    the Request watcher) raises _StreamCancelled instead of holding the
    account slot through the full session→PoW setup.
    """
    import concurrent.futures as _fut

    if cancel_event is None:
        return fut.result()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                fut.cancel()
            except Exception:
                pass
            raise _StreamCancelled()
        try:
            return fut.result(timeout=_CANCEL_POLL_SECONDS)
        except _fut.TimeoutError:
            continue


def apply_log_level(level: str) -> None:
    """Apply log level to deepseaport loggers (settings-driven, no handler reset)."""
    try:
        numeric = getattr(logging, str(level or "INFO").upper(), logging.INFO)
        # The parent, not a name list: every deepseaport logger is a child of
        # "deepseaport" and is created NOTSET, so getEffectiveLevel() walks up
        # to it. A hand-maintained tuple of module names silently misses any
        # logger added later, and those records then ignore the setting.
        logging.getLogger("deepseaport").setLevel(numeric)
    except Exception:
        pass


def _effective_log_level(settings) -> str:
    """Runtime override first, then the saved field (chat mode pins the former)."""
    return (getattr(settings, "_runtime_log_level", None)
            or getattr(settings, "log_level", "INFO") or "INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    apply_log_level(_effective_log_level(settings))
    bridge = ObscuraBridge(settings.obscura_bin, settings.obscura_profile)
    app.state.bridge = bridge
    register_default_bridge(bridge)
    if getattr(settings, "warmup_on_startup", True):
        threading.Thread(target=_startup_warm, args=(bridge, settings),
                         daemon=True).start()
    yield
    # NOTE: module-level executors intentionally NOT shut down here: lifespan
    # can run multiple times per process (tests, reload) and a shutdown
    # executor raises RuntimeError on submit. Their workers are daemon threads
    # (see _DaemonExecutor), so the process is not held by one that is still
    # parked on an upstream socket.


def _startup_warm(bridge: ObscuraBridge, settings=None) -> None:
    # Local-only mode (opencode CLI) needs no WAF bridge: skip the obscura
    # warmup (and the DeepSeek PoW fallback below) when no active provider
    # consumes the bridge's harvested jar -- a qwen-only setup would otherwise
    # boot a browser against the DeepSeek host and plant cookies qwen must
    # never send. Provider warmups still run.
    try:
        from .core import pools as _pools_w
        from .core import registry as _reg_w
        if settings is not None:
            _w_ids = _pools_w.active_provider_ids(settings)
        else:
            _w_ids = [_reg_w.default().id]
        # Resolve the ids ONCE and drop the unknown ones. getattr(None,
        # "needs_waf_jar", True) defaulted to True, so a provider the registry
        # does not have counted as "consumes the jar" and fetched the DeepSeek
        # wasm for a provider that needs neither a browser nor an account.
        _w_provs = [p for p in (_reg_w.get(pid) for pid in _w_ids)
                    if p is not None]
        _needs_waf_jar = any(bool(getattr(p, "needs_waf_jar", True))
                             for p in _w_provs)
    except Exception as exc:
        logger.debug("bridge-need check unavailable, warming by default: %s", exc)
        _needs_waf_jar = True
    if _needs_waf_jar:
        try:
            bridge.warmup()
        except Exception as exc:
            logger.warning("startup WAF warmup failed: %s", exc)
    # Provider-driven warmup: warm every active provider (deepseek only in
    # single mode). It is authoritative once it answers -- the inline PoW path
    # below exists for the registry being unavailable, not for a provider that
    # simply has nothing to warm.
    try:
        from .core import pools, registry
        if settings is not None:
            provider_ids = pools.active_provider_ids(settings)
        else:
            provider_ids = [registry.default().id]
        for pid in provider_ids:
            provider = registry.get(pid)
            if provider is None:
                # An unknown id is not a reason to warm DeepSeek. `warmed`
                # stayed False here, so this used to run on into the inline PoW
                # path: a wasted wasm fetch plus an account prompt for a
                # provider that is not serving anything.
                logger.warning("startup warmup skipped: unknown provider %r", pid)
                continue
            provider.warmup(bridge)
        return
    except Exception as exc:
        logger.debug("provider warmup unavailable, using inline PoW: %s", exc)
    if not _needs_waf_jar:
        # Local-only mode: no DeepSeek PoW fallback (nothing uses it).
        return
    # Inline PoW fallback is DeepSeek-specific and lazy-imported so this
    # module has no top-level providers.deepseek import. Provider.warmup
    # above is the preferred path.
    try:
        from .providers.deepseek import pow as PoW
        PoW.ensure_wasm(cookies=bridge.cookie_header(), user_agent=bridge.state.user_agent)
    except Exception as exc:
        logger.warning("PoW wasm fetch failed (will retry on demand): %s", exc)
        return
    # Pre-compile wasmtime Engine/Module so the first completion skips the
    # 100-300ms compile on the hot path (Store/instance stay per-solve).
    try:
        PoW.prewarm()
    except Exception as exc:
        logger.debug("PoW prewarm failed (lazy compile on demand): %s", exc)
