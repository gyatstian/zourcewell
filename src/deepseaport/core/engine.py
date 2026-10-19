"""Provider-neutral request engine: retry driver, failover loop, stream bridge.

No provider imports (no DeepSeek client/protocol/PoW), no ``deepseaport.server``
imports, no ``_server_override`` lookups. Providers supply small adapters
(predicates, recover callbacks, run-once closures); the engine owns only the
generic mechanics: sequential retry tables, account-slot release/acquire
failover, cancellation checks, and producer-thread bridging.

DeepSeek keeps session/PoW/SSE/ban + token-refresh in
``providers.deepseek.completion_service`` and Qwen keeps chat lifecycle in
``providers.qwen.completion_service``; both delegate their loops here.
Provider-specific parts that stay in the provider: ``classify_error``,
``_headers``, session/chat setup, PoW solve, WAF warmup body, and the
invalid-token terminal (DeepSeek probes + browser note, Qwen is static).
Everything else -- slot lifecycle, failover wrapper, attempt drain,
ban-403 marking, CURRENT switch, live-emit guard, terminal messages,
stream consumer -- lives here exactly once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager

from fastapi import HTTPException

from ..server_concurrency import _STREAM_EXECUTOR, _StreamCancelled

logger = logging.getLogger("deepseaport.core.engine")


# Exception types a recovery closure must never have contained.
#
# A recovery can raise two kinds of exception and only one of them is a
# malfunction:
#
# * ``HTTPException`` is a deliberate status DECISION -- the ban 403 an
#   invalid-token recovery raises after a positive users/current probe, the
#   401 the session path raises for a dead token. That status is the answer
#   the client must see, so it propagates untouched (otherwise a 401 that
#   was really a ban would be downgraded to the generic upstream status).
# * ``_StreamCancelled`` means the client already left; it has to win, or a
#   disconnect during recovery would be reported as an upstream 401/502
#   instead of 499.
#
# Everything else is a failure OF the recovery itself -- ``DS.solve_pow``'s
# HTTP ``fetch_pow`` blowing up, ``RuntimeError("PoW solver found no
# answer")``, a browser warmup error -- and must not become the client's
# error.
_MUST_PROPAGATE: tuple[type[BaseException], ...] = (HTTPException, _StreamCancelled)


def _run_recovery(name: str, recover: Callable[[], None]) -> bool:
    """Run one recovery closure; return False when it failed (contained).

    A recovery closure calls real work (PoW solve + HTTP challenge fetch,
    session recreate, browser token refresh, Obscura warmup) and that work can
    fail. Its failure is NOT the client's actionable error -- the upstream
    reason is -- and letting it escape replaced the decided status (503
    overload / 502 WAF / 401 token) with an unhandled 500 on the non-stream
    path. On failure we log the scrubbed reason and stop escalating this kind;
    the caller hands the unchanged error to the next handler and the terminal
    ladder raises the ORIGINAL status. Not swallowed: the request still fails.
    """
    try:
        recover()
        return True
    except _MUST_PROPAGATE:
        raise
    except Exception as exc:
        # No ``exc_info``: a recovery failure is very often the provider's own
        # HTTP raise, whose message embeds a slice of the raw upstream body,
        # and the traceback prints that message again at the bottom. Both were
        # a second unscrubbed copy of the body this file exists to de-tokenize,
        # and a recovery that fails is not worth a stack trace anyway.
        logger.warning(
            "retry recovery %r failed (%s: %s); keeping the original upstream "
            "error and escalating no further on this kind",
            name, type(exc).__name__, scrub_upstream_text(exc, 500))
        return False


def drive_retries(
    max_retries: int,
    is_cancelled: Callable[[], bool],
    get_error: Callable[[], str | None],
    handlers: Sequence[tuple[str, Callable[[str | None], bool], Callable[[], None]]],
) -> None:
    """Run an ordered retry table to fixpoint (one pass per kind, in order).

    Each handler is ``(name, should_retry, recover)``: while not cancelled and
    ``should_retry(current_error)`` holds, call ``recover()`` (which must
    refresh the error via a new attempt), up to ``max_retries`` times; a later
    handler sees the error the earlier one left. A raising ``recover()`` is
    contained by ``_run_recovery`` -- log, stop escalating this kind, hand the
    unchanged error on -- so a failed recovery cannot replace the decided
    status (503 overload / 502 WAF / 401 token) with a 500. ``HTTPException``
    and the cancellation sentinel are exempt: decisions, not malfunctions (see
    ``_run_recovery`` and core/AGENTS.md).
    """
    try:
        budget = max(0, int(max_retries))
    except Exception:
        budget = 0
    for name, should_retry, recover in handlers:
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
            if not _run_recovery(name, recover):
                break


def run_with_failover(
    pool,
    first_item,
    run_once: Callable[[object], object],
    is_retryable_exc: Callable[[BaseException], bool],
    acquire: Callable[[float], object],
    on_before_retry: Callable[[BaseException, object, int, int], None] | None = None,
    acquire_timeout: float = 10,
    is_cancelled: Callable[[], bool] | None = None,
    allow_failover: bool = True,
    *,
    release: Callable[[object], None],
) -> object:
    """Run ``run_once(item)`` on ``first_item``, failing over on retryable errors.

    Owns each attempt's slot: releases it exactly once per attempt (success,
    retryable, or fatal) unless ``slot_releaser`` already gave it up inside
    that attempt -- the give-up is published thread-locally and the release is
    then skipped, because the lock belongs to whichever request acquired the
    account meanwhile (see ``slot_releaser``). Acquires the next account
    before retrying, and re-raises the first error when no healthy account
    remains. ``run_once`` performs one attempt (including any per-kind
    retries); the engine only handles cross-account failover, cancellation,
    and slot lifecycle.

    ``acquire`` and ``release`` are both required: the only caller binds the
    pool's own ``acquire`` and ``AccountPool.release``, so a default could only
    duplicate the former or shadow the latter.
    """
    def _release_slot(item) -> None:
        # Not ``release(item)``: see the docstring. Consuming the give-up
        # always runs, so it cannot leak into the next attempt on this thread.
        # The skip also drops ``AccountPool.release``'s ``uses`` bump and its
        # waiter wake -- stats-only counter, and acquire() re-polls anyway.
        if not consume_slot_abandoned(item):
            release(item)

    if not allow_failover:
        try:
            return run_once(first_item)
        finally:
            try:
                _release_slot(first_item)
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
                _release_slot(cur)
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
            _release_slot(cur)
        except Exception:
            pass
        return result


def error_text(code: str | None, message: str | None) -> str:
    """Single "CODE: message" formatter (the string classify_error parses)."""
    code = (code or "").strip()
    message = (message or "").strip()
    return f"{code}: {message}" if code else message


# Shared error vocabularies for the providers' ``classify_error`` tables.
# ``transport.http_error_code`` renders every non-200 as "HTTP_502: <body>",
# so a bare status shape must classify the same way on every provider: the
# word lists were forked into each classifier and the numeric fallbacks were
# never back-propagated (DeepSeek retried "bad gateway" but answered a bare
# "502" terminally). A provider composes the neutral tuple with its own extras
# (``_engine.TRANSIENT_OVERLOAD_MARKERS + (...)``) and keeps its own check order:
# rate-limit before overload wherever a throttle body also says "try again".
TRANSIENT_OVERLOAD_MARKERS = (
    "overload", "overloaded", "server busy", "server_busy",
    "service unavailable", "temporarily unavailable", "try again",
    "timed out", "timeout", "bad gateway", "gateway timeout",
    "500", "502", "503", "504",
)
RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "too many")
UPSTREAM_INVALID_MARKERS = (
    "http_422", "http 422", "http_400", "http 400", "unprocessable")


# An "opaque run" is a long unbroken stretch of token characters -- exactly
# what a credential looks like (bearer token, session id, base64 blob, hex
# nonce) and never what a diagnostic sentence looks like. 32 is comfortably
# above the longest thing an operator reads in an error ("generation_timeout",
# "Authorization Failed", a biz code, a ban phrase) and comfortably below a
# real secret.
_OPAQUE_RUN_RE = re.compile(r"[A-Za-z0-9_\-]{32,}")

# The second shape a secret takes and the run rule structurally cannot see: a
# JWT, i.e. dot-framed base64url segments. Its segments are each SHORT (a
# typical header is 20 chars, a signature 27), so `[A-Za-z0-9_-]{32,}` matches
# nothing and the whole token -- header and payload included -- travelled
# verbatim. A future base64 (`+`/`/`) credential has the same blind spot.
#
# The pattern is deliberately anchored to a whole token-shaped string: the
# lookarounds stop it matching the middle of a longer run, and requiring every
# segment to be 4+ chars of token characters (dots directly adjacent, no
# spaces) is what keeps prose intact -- "Server busy. Try again." has spaces
# around its dots, "opencode 1.18.32" has 1-2 char segments, and
# "docs.example.com" ends in a 3-char segment. Both patterns are needed: the
# run rule catches the single-segment secrets (DeepSeek's 64-char userToken),
# this one catches the framed ones.
_SEGMENT_FRAMED_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])[A-Za-z0-9_\-]{4,}(?:\.[A-Za-z0-9_\-]{4,}){1,3}"
    r"(?![A-Za-z0-9_.\-])")


def scrub_upstream_text(text, limit: int = 800) -> str:
    """Clamp and de-tokenize an upstream body before it leaves the process.

    Runs on every path that can echo an upstream body, always before any
    truncation (the sink list is core/AGENTS.md's). The body can hold an
    Authorization header, a session id, a captcha token, or the account's own
    64-char bearer token -- the failing request is the one carrying it.

    Two redaction shapes, one per credential shape (see ``_SEGMENT_FRAMED_RE``
    and ``_OPAQUE_RUN_RE``); the framed rule runs FIRST so a JWT is redacted
    whole rather than partially -- its payload can be long enough to trip the
    run rule and leak the header next to a ``[redacted]``.

    Otherwise the redaction is deliberately narrow: operator-diagnosis words
    survive verbatim (status code, biz_code, the ban/error wording), the
    length is hard-clamped so a hostile or broken upstream cannot use the
    detail as a megabyte of reflected anything, and it never raises -- this
    runs on the terminal error path, where raising would replace the real
    status with a 500.
    """
    try:
        out = "" if text is None else str(text)
    except Exception:
        return ""
    try:
        out = _SEGMENT_FRAMED_RE.sub("[redacted]", out)
    except Exception:
        pass
    try:
        out = _OPAQUE_RUN_RE.sub("[redacted]", out)
    except Exception:
        pass
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = 800
    try:
        if cap and len(out) > cap:
            out = out[:cap] + f"... [+{len(out) - cap} chars]"
    except Exception:
        pass
    return out


class AttemptOutcome(tuple):
    """4-tuple (content, thinking, error, usage) + ban_until attr.

    Subclass so existing unpacking ``a, b, c, d = _attempt(...)`` keeps
    working (tests + callers), while ``_run_completion_core`` reads
    ``.ban_until``. Plain-tuple mocks simply have no attr -> getattr fallback.
    """

    def __new__(cls, content: str, think: str, error: str | None,
                usage: int, ban_until: float | None = None):
        obj = super().__new__(cls, (content, think, error, usage))
        obj.ban_until = ban_until
        return obj


# The cancel token of the attempt this thread is currently draining.
# Thread-local, not a parameter: ``drain_attempt`` receives ``stream_fn`` as
# ``provider.client.stream_completion`` and every provider's signature is
# ``(headers, payload)`` -- adding a third argument here would mean passing a
# kwarg every provider does not accept, and passing it positionally would have
# silently landed in Qwen's ``timeout``. ``drain_attempt`` and the generator it
# creates run on the same producer thread, so thread-local is the exact scope
# that matters: no cross-talk between concurrent requests, and no
# re-entrancy problem (the previous value is restored, not cleared).
_CURRENT_CANCEL = threading.local()


def current_cancel_event() -> threading.Event | None:
    """The cancel token in force on this thread, or None outside a drain.

    Read by a stream function that sits inside a blocking socket read, where
    returning control to ``drain_attempt``'s per-event check is the only way to
    notice a disconnect.
    """
    return getattr(_CURRENT_CANCEL, "event", None)


def drain_attempt(stream_fn: Callable[[dict, dict], object],
                  headers: dict, payload: dict,
                  on_content=None, on_thinking=None,
                  cancel_event: threading.Event | None = None,
                  ) -> tuple[str, str, str | None, int]:
    """Drain one SSE stream into (content, thinking, error, usage).

    ``stream_fn(headers, payload)`` yields events with ``.kind`` in
    {content, thinking, usage, error, finished}, ``.text`` / ``.code`` /
    ``.message`` / ``.ban_until``. Shared by every web2api provider; the only
    per-provider input is the stream function itself. Explicit ``gen.close()``
    so the HTTP connection returns to the pool instead of lingering through
    retries. Cancel is checked per event so a disconnected client frees the
    account slot promptly, and is published for the duration of the drain (see
    ``current_cancel_event``) so a stream function blocked inside the socket
    read can notice it too.
    """
    fmt = error_text
    content: list[str] = []
    think: list[str] = []
    error: str | None = None
    usage_total = 0
    ban_until: float | None = None
    _outer_cancel = getattr(_CURRENT_CANCEL, "event", None)
    _CURRENT_CANCEL.event = cancel_event
    gen = stream_fn(headers, payload)
    try:
        for event in gen:
            if cancel_event is not None and cancel_event.is_set():
                break
            kind = getattr(event, "kind", None)
            if kind == "content":
                text = getattr(event, "text", "") or ""
                content.append(text)
                if on_content is not None:
                    try:
                        on_content(text)
                    except Exception:
                        pass
            elif kind == "thinking":
                text = getattr(event, "text", "") or ""
                think.append(text)
                if on_thinking is not None:
                    try:
                        on_thinking(text)
                    except Exception:
                        pass
            elif kind == "usage":
                try:
                    usage_total = max(usage_total, int(getattr(event, "text", 0)))
                except (ValueError, TypeError):
                    pass
            elif kind == "error":
                try:
                    error = fmt(getattr(event, "code", None),
                                getattr(event, "message", None))
                except Exception:
                    error = str(getattr(event, "message", "") or "upstream error")
                try:
                    ban_until = getattr(event, "ban_until", None)
                except Exception:
                    ban_until = None
                break
            elif kind == "finished":
                break
    finally:
        _CURRENT_CANCEL.event = _outer_cancel
        try:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    return AttemptOutcome("".join(content), "".join(think), error,
                          usage_total, ban_until)


# How long ``slot_releaser`` waits to hand the account slot back before
# giving up (see that docstring). Sized for the realistic contention, not
# for the worst case: the slot was released so OTHER work could use the
# account, and the honest owner of it is another request whose own attempt
# can legitimately run for a full stream (a ~90s socket budget plus PoW).
# One minute covers that for a single-account deployment; a WAF warmup or a
# browser login that outlives it just means this attempt proceeds without
# the slot, which is the safe direction *and* is published to the release
# sites via ``consume_slot_abandoned``.
SLOT_REACQUIRE_TIMEOUT = 60.0

# The account whose slot this thread gave up and did not get back. Thread-local
# for the same reason as ``_CURRENT_CANCEL``: the thread that released the
# slot is the thread whose ``run_with_failover`` release runs later. It is
# needed because ``threading.Lock.release()`` has NO owner check -- releasing
# a lock this thread does not hold SUCCEEDS -- so a blind release after a
# give-up unlocked the request that had acquired the account meanwhile, and a
# third request could then start a stream on an account already in flight.
_SLOT_ABANDONED = threading.local()


def _slot_given_up(item) -> bool:
    """True when THIS thread already gave up ``item``'s slot (read-only)."""
    return getattr(_SLOT_ABANDONED, "item", None) is item


def consume_slot_abandoned(item) -> bool:
    """True once when THIS thread gave up ``item``'s slot; consumes it.

    Identity-matched (``is``), not a bare flag: a give-up belongs to one
    account for the rest of one attempt, never to whatever account the thread
    touches next. Otherwise a stale entry would suppress a real release --
    leaking a lock that stays held until the process restarts.
    """
    given_up = _slot_given_up(item)
    _SLOT_ABANDONED.item = None
    return given_up


@contextmanager
def slot_releaser(item, cancel_event: threading.Event | None = None):
    """Release the per-account slot during global (browser/WAF) waits.

    Releases ``item.lock`` on entry (no-op when not held, e.g. unit tests
    calling the engine with a fresh account), reacquires on exit. The
    reacquire polls so a disconnect aborts promptly instead of blocking
    indefinitely under contention (raises ``_StreamCancelled``).

    Exiting WITHOUT the slot back (cancel, deadline, acquire raising) leaves
    the lock to whoever took the account meanwhile, so the give-up is
    published for this thread and ``run_with_failover``'s release sites skip
    it: ``Lock.release()`` from a non-holder succeeds and steals the new
    holder's lock. A second ``slot_releaser`` on the same item later in the
    same attempt (token refresh after a WAF give-up) must not release either,
    which is why the entry release is guarded too.

    The reacquire is DEADLINE-bounded (``SLOT_REACQUIRE_TIMEOUT``) because
    ``cancel_event=None`` is legal (it is ``Provider.complete``'s documented
    default, and the direct-call/unit-test path), so every wait in this module
    has to be bounded. On give-up the attempt continues without the slot
    rather than block forever; the pool still rate-limits it.
    """
    released = False
    if not _slot_given_up(item):
        try:
            item.lock.release()
            released = True
        except Exception:
            # RuntimeError when nobody holds it (a fresh account in a unit
            # test, or an attempt whose give-up already released it): not held
            # is the answer, not a malfunction.
            released = False
    try:
        yield released
    finally:
        if released:
            reacquired = False
            try:
                deadline = time.monotonic() + SLOT_REACQUIRE_TIMEOUT
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise _StreamCancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # Bound hit, not a cancel: the account is busy
                        # elsewhere (another request, or a failover to this
                        # very account). Continue WITHOUT the slot rather than
                        # block forever -- the attempt is still rate-limited by
                        # the pool, and the outer release skips a slot this
                        # thread no longer owns.
                        logger.warning(
                            "account slot still held elsewhere after %.0fs; "
                            "continuing without re-acquiring it for this "
                            "attempt",
                            SLOT_REACQUIRE_TIMEOUT)
                        break
                    try:
                        if item.lock.acquire(timeout=min(0.5, remaining)):
                            reacquired = True
                            break
                    except Exception:
                        break
            finally:
                if not reacquired:
                    _SLOT_ABANDONED.item = item


def app_settings(app):
    """Return app settings, tolerating app=None and embedded test apps.

    The freshly loaded fallback is deliberately NOT memoized onto
    ``app.state``. A hand-built app (test/tool embedding) has no settings of
    its own, and ``load_settings()`` returns a Settings bound to the operator's
    on-disk ``config.json``; writing it back meant every later
    ``app.state.settings`` read got that same operator-bound object, and the
    provider save paths (qwen token expiry, ban persisting) then rewrote the
    real file. The caller still gets the object to use for THIS call -- nothing
    is retained, so the next call re-reads and a provider save stays the
    caller's own decision.
    """
    if app is None:
        from ..config import load_settings

        return load_settings()
    settings = getattr(getattr(app, "state", None), "settings", None)
    if settings is None:
        from ..config import load_settings

        return load_settings()
    return settings


def ensure_token(item, no_token_detail: str) -> str:
    """Return the account token, or a fast 401 that failover can retry on.

    Normalizes JSON-wrapped pastes via ``extract_token``. ``no_token_detail``
    is provider-specific (DeepSeek points at ``deepseaport login``, Qwen at
    the chat.qwen.ai paste); the mechanics (normalize-or-401) are shared.
    """
    from ..accounts import extract_token

    if getattr(getattr(item, "cfg", None), "token", None):
        token = extract_token(item.cfg.token)
        if token != item.cfg.token:
            item.cfg.token = token
        return token
    raise _http_401(no_token_detail)


def _http_401(detail: str):
    return HTTPException(status_code=401, detail=detail)


def ban_403_for_item(app, item, ban_until: float | None,
                     error_text_: str = "",
                     ban_message_fn: Callable[[float | None], str] | None = None):
    """Mark + persist an account ban and build the 403.

    Persisting ``cfg.banned``/``cfg.banned_until`` is what makes the account
    show up as ``(BANNED ...)`` in the CLI/TUI even after restart.
    ``ban_until=None`` still marks the account instead of losing the verdict.
    ``ban_message_fn`` is provider-specific (DeepSeek ``P.ban_message``,
    Qwen local ``_ban_message``); everything else is shared. Never raises
    for the marking itself; always returns the 403 to raise.

    Persist is gated on the app OWNING its settings -- the same guard
    ``providers/qwen/completion_service`` applies. ``app_settings(app)``
    falls back to ``load_settings()``, which is a Settings bound to the
    operator's real ``config.json``; saving that from a hand-built app (a
    test or a tool embedding the engine with no ``app.state.settings``) would
    rewrite the operator's file with a ban they never hit. The fallback is
    also no longer memoized (see ``app_settings``), so the write would not
    even stick as a side effect -- it just corrupted the file. When the app
    owns its settings we save exactly as before; otherwise the in-memory
    marker stands alone and a warning says so.
    """
    try:
        from ..accounts import AccountPool

        AccountPool.mark_banned(item, until=ban_until)
        _ident = getattr(getattr(item, "cfg", None), "identifier", "?")
        own = getattr(getattr(app, "state", None), "settings", None)
        if own is not None:
            try:
                own.save()
            except Exception as exc:
                logger.warning("ban for account %s could not be persisted "
                               "(save failed: %s: %s); the marker is "
                               "in-memory only and a restart forgets it",
                               _ident, type(exc).__name__, exc)
        else:
            logger.warning(
                "ban for account %s NOT persisted: this app owns no "
                "settings (app.state.settings is unset), so there is nothing "
                "safe to write -- the operator's config.json was left "
                "untouched on purpose. The ban marker is in-memory only; "
                "a restart forgets it.", _ident)
    except Exception:
        pass
    try:
        base = ban_message_fn(ban_until) if ban_message_fn is not None else ""
        ident = getattr(getattr(item, "cfg", None), "identifier", "?")
        if not base:
            # Scrubbed: the caller-facing text here is the RAW upstream/login
            # body (the Obscura path passes up to 200 chars of scraped login
            # page DOM), which lands in the 403 detail the client sees and in
            # nothing the operator can act on beyond the ban verdict itself.
            # ``base`` is our own ban_message, not upstream text, and is kept
            # verbatim.
            detail = scrub_upstream_text(error_text_ or "account banned", 300)
            if "banned" not in detail.lower() and "muted" not in detail.lower():
                detail = f"Account {ident} banned. {detail}"
        else:
            detail = f"Account {ident} banned. {base}"
    except Exception:
        try:
            ident = getattr(getattr(item, "cfg", None), "identifier", "?")
        except Exception:
            ident = "?"
        detail = scrub_upstream_text(error_text_ or f"Account {ident} banned", 300)
    return HTTPException(status_code=403, detail=detail)


def is_ban_http_exception(exc: BaseException,
                          is_ban_text: Callable[[str], bool]) -> bool:
    """True when exc is the ban 403 raised by ``_run_completion_core``."""
    try:
        return (isinstance(exc, HTTPException) and exc.status_code == 403
                and bool(is_ban_text(str(getattr(exc, "detail", "") or ""))))
    except Exception:
        return False


def switch_current_after_ban_failover(pool, settings, used, failed,
                                      reason: str = "unusable",
                                      persist: bool | None = None) -> None:
    """Move CURRENT to the working account after failover.

    Only fires when CURRENT pointed at the failed account: the request
    already proved it unusable (ban/auth cooldown), so future requests should
    prefer the healthy one first. Best-effort, never raises.

    ``persist`` decides whether the moved selection is written to disk.
    ``None`` means "persist" (the historical behaviour, kept for the
    provider aliases that call this with a settings object they were handed).
    ``False`` marks the switch in-memory only. The reason it has to be
    selectable: ``complete_with_failover_sync`` gets ``settings`` from
    ``get_settings``, whose default is ``engine.app_settings`` -- and that
    falls back to ``load_settings()``, a Settings bound to the OPERATOR's real
    ``config.json``. Saving that from an app that owns no settings (a
    hand-built app, a test, an embedder) rewrote the operator's file with a
    CURRENT switch they never made -- the same defect
    ``ban_403_for_item`` was fixed for. So the caller passes the ownership
    verdict it already computed, and a skip says so in the log.
    """
    try:
        from ..accounts import _matches as _m
        cur = pool.current
        if not cur or not _m(failed.cfg, cur):
            return  # auto-failover mode or CURRENT wasn't the failed one
        if _m(used.cfg, cur):
            return  # same account (shouldn't happen after mark_bad)
        if pool.set_current(used.cfg.identifier):
            try:
                settings.active_account = pool.current
                if persist is None or bool(persist):
                    settings.save()
                else:
                    logger.warning(
                        "CURRENT %s -> %s switched but NOT persisted: this "
                        "app owns no settings (app.state.settings is unset), "
                        "so there is nothing safe to write -- the operator's "
                        "config.json was left untouched on purpose. The "
                        "switch is in-memory only; a restart picks the old "
                        "CURRENT again.", failed.cfg.identifier,
                        used.cfg.identifier)
            except Exception:
                pass
            logger.warning("account %s %s, CURRENT auto-switched -> %s",
                           failed.cfg.identifier, reason, used.cfg.identifier)
    except Exception:
        pass


def complete_with_failover_sync(app, pool, prep: dict, first_item,
                                run_core: Callable[[object], dict],
                                is_ban_exc: Callable[[BaseException], bool],
                                cancel_event: threading.Event | None = None,
                                on_content=None, on_thinking=None,
                                allow_failover: bool | None = None,
                                get_settings: Callable[[object], object] | None = None,
                                ) -> dict:
    """Run one completion, auto-failing over on ban or broken credentials.

    The provider supplies ``run_core(account)`` (its ``_run_completion_core``
    bound to app/prep/callbacks) and the ban-403 predicate; the generic slot
    lifecycle is ``run_with_failover``'s (see there). ``allow_failover=False``
    surfaces the first error directly; None reads ``use_multiple_accounts``
    from settings. Bans and 401 auth failures are account-specific; other
    failures belong to the caller/current request.

    ``prep``, ``on_content`` and ``on_thinking`` are ACCEPTED AND IGNORED: the
    ``run_core`` closure already captured all three (both adapters pass them
    by keyword), so wiring them up here would apply every delta a second time
    -- duplicated SSE text and a doubled tool-call JSON payload, which is
    worse than a crash because the client renders a plausible wrong answer.
    They are ``del``-ed at the top of the body so the next maintainer reads a
    ``NameError`` risk instead of a plausible-looking dead parameter; the
    parameters stay in the signature purely for call compatibility. Do NOT
    reintroduce a use for them: add a new, explicitly named parameter if the
    engine ever needs to drive emission itself.
    """
    # Historical call-compatibility only -- see the docstring. Do not use.
    del prep, on_content, on_thinking
    _get_settings = get_settings or app_settings
    if allow_failover is None:
        try:
            allow_failover = bool(getattr(_get_settings(app),
                                          "use_multiple_accounts", True))
        except Exception:
            allow_failover = True

    try:
        from ..server_concurrency import (
            FAILOVER_ACQUIRE_TIMEOUT as _FAILOVER_TIMEOUT,
        )
        from ..server_concurrency import (
            INVALID_TOKEN_COOLDOWN_SECONDS as _TOKEN_COOLDOWN,
        )
    except Exception:
        _FAILOVER_TIMEOUT, _TOKEN_COOLDOWN = 10, 60

    first_exc = None
    failed_item = None
    used_item = None

    def _run_once(account) -> dict:
        nonlocal used_item
        result = run_core(account)
        used_item = account
        return result

    def _is_retryable(exc: BaseException) -> bool:
        try:
            return (isinstance(exc, HTTPException)
                    and (bool(is_ban_exc(exc)) or exc.status_code == 401))
        except Exception:
            return False

    def _on_before_retry(exc, account, tries: int, max_tries: int) -> None:
        nonlocal first_exc, failed_item
        if first_exc is None:
            first_exc, failed_item = exc, account
        try:
            from ..accounts import AccountPool

            if isinstance(exc, HTTPException) and exc.status_code == 401:
                AccountPool.mark_bad(account, seconds=_TOKEN_COOLDOWN)
                logger.info("account %s has no usable token, failing over "
                            "(%d/%d)", account.cfg.identifier, tries + 1, max_tries)
            else:
                logger.info("account %s banned, failing over to next account "
                            "(%d/%d)", account.cfg.identifier, tries + 1, max_tries)
        except Exception:
            pass

    from ..accounts import AccountPool as _Pool
    _release = _Pool.release

    result = run_with_failover(
        pool,
        first_item,
        _run_once,
        _is_retryable,
        acquire=lambda timeout: pool.acquire(timeout=timeout),
        on_before_retry=_on_before_retry,
        acquire_timeout=_FAILOVER_TIMEOUT,
        is_cancelled=lambda: cancel_event is not None and cancel_event.is_set(),
        allow_failover=allow_failover,
        release=_release,
    )
    if failed_item is not None and used_item is not None:
        try:
            reason = "banned" if bool(is_ban_exc(first_exc)) else "has no usable token"
        except Exception:
            reason = "unusable"
        try:
            settings = _get_settings(app)
            # Ownership verdict, same rule as ban_403_for_item: only the app's
            # OWN settings may be written back. The fallback Settings from
            # app_settings() is the operator's on-disk config.json, and
            # saving it from a settings-less app rewrote their file with a
            # CURRENT switch this process invented.
            own = getattr(getattr(app, "state", None), "settings", None)
            persist = own is not None and own is settings
        except Exception:
            settings, persist = None, False
        try:
            if settings is not None:
                switch_current_after_ban_failover(
                    pool, settings, used_item, failed_item,
                    reason=reason, persist=persist)
        except Exception:
            pass
    return result  # type: ignore[return-value]


class LiveEmitTracker:
    """Flag partial live emits so EVERY retry row skips once output left.

    Live mode already ``_put`` partial deltas to the client; a retry after a
    partial emit would re-POST a payload the upstream already turned into a
    turn, then re-stream from 0 -- duplicated/interleaved text and a corrupt
    tool-call JSON object inside a 200 stream that still ends in a clean
    ``[DONE]``. Wrap each live callback once; every retry predicate checks
    ``tracker.emitted``. Shared by every provider.
    """

    def __init__(self) -> None:
        self.emitted = False

    def wrap(self, cb: Callable[[str], None] | None) -> Callable[[str], None]:
        def _emit_tracked(t: str) -> None:
            if t:
                self.emitted = True
            if cb is not None:
                try:
                    cb(t)
                except Exception:
                    pass

        return _emit_tracked


def raise_rate_limit(provider_label: str, error: str | None) -> None:
    """Global throttle: 429, never account-marked. Always raises."""
    raise HTTPException(status_code=429,
                        detail=f"{provider_label} rate limited: {scrub_upstream_text(error, 200)}. "
                               "Wait 2-20 min, space requests, avoid parallel loops.")


def raise_overload(provider_label: str, error: str | None) -> None:
    """Transient upstream overload: 503 (retryable), never a 500. Always raises."""
    raise HTTPException(status_code=503,
                        detail=f"{provider_label} server busy: {scrub_upstream_text(error, 200)}. "
                               "Retry shortly (transient overload).")


def raise_upstream_invalid(provider_label: str, prep: dict, error: str | None) -> None:
    """Upstream payload rejection (422/400 family): 502. Always raises."""
    logger.warning("upstream rejected completion payload (%s); "
                   "prompt_len=%d model=%s",
                   scrub_upstream_text(error, 500), len(str((prep or {}).get("prompt", ""))),
                   (prep or {}).get("model"))
    raise HTTPException(
        status_code=502,
        detail=f"{provider_label} rejected the request as invalid ({scrub_upstream_text(error, 800)}). "
               "The prompt/payload failed upstream validation, not a ban or "
               "rate limit. Retry with a non-empty prompt; if it persists, "
               "the upstream API schema likely changed.")


def raise_upstream(provider_label: str, error: str | None) -> None:
    """Generic upstream error: 502. Always raises."""
    logger.warning("upstream completion error: %s", scrub_upstream_text(error, 500))
    raise HTTPException(
        status_code=502,
        detail=f"{provider_label} upstream error: {scrub_upstream_text(error, 800)}")


def empty_response_error(provider_label: str):
    """Upstream closed without content/thinking/error: 502, never silent 200."""
    return HTTPException(status_code=502,
                         detail=f"empty upstream response from {provider_label} "
                                "(no content/thinking/error events; often "
                                "throttling — wait then retry; if it persists "
                                "for hours, the upstream stream format likely "
                                "changed — update deepseaport)")


def generic_terminal_handlers(kind, label: str, *, upstream=None) -> dict:
    """Terminal handlers that only differ by provider label.

    RATE_LIMIT / OVERLOAD / UPSTREAM_INVALID / UPSTREAM carry no provider
    state -- they are the raise_* helpers above with the provider's display
    name substituted. Every web2api adapter was re-declaring those four
    one-line wrappers plus a hand-written dispatch table, so a fix to any
    raise_* message had to be copied into each provider. Only INVALID_TOKEN
    and BAN are genuinely provider-specific (they read the stored credential
    and the account health table), so those stay in the adapter and are
    merged in by the caller.

    ``kind`` is the provider's own RetryKind enum (the enums are deliberately
    per-provider; members differ), so the returned dict is keyed the way
    ``dispatch_terminal`` looks up. ``upstream`` overrides just the
    UPSTREAM entry for the one provider that needs a distinguishable
    message there.

    ponytail: one table, keyed per provider. A provider with new terminal
    states adds its own kind + handler; it does not fork this.
    """
    def _rate_limit(app, item, prep, token, error, login_note, ban_until) -> None:
        raise_rate_limit(label, error)

    def _overload(app, item, prep, token, error, login_note, ban_until) -> None:
        raise_overload(label, error)

    def _upstream_invalid(app, item, prep, token, error, login_note, ban_until) -> None:
        raise_upstream_invalid(label, prep, error)

    def _upstream(app, item, prep, token, error, login_note, ban_until) -> None:
        raise_upstream(label, error)

    return {kind.RATE_LIMIT: _rate_limit,
            kind.OVERLOAD: _overload,
            kind.UPSTREAM_INVALID: _upstream_invalid,
            kind.UPSTREAM: upstream or _upstream}


def dispatch_terminal(classify_fn: Callable[[str | None], object],
                      handlers: dict, app, item, prep: dict,
                      token: str | None, error: str,
                      login_note, ban_until) -> None:
    """Map a terminal completion error to its HTTP status. Always raises.

    Single dispatch: ``classify_fn`` (the provider's ``classify_error``)
    produces one kind, then the handler for that kind raises. No string
    matching lives at call sites.

    Uniform terminal contract (all providers): handlers take
    ``(app, item, prep, token, error, login_note, ban_until)``. Providers
    without a browser note pass ``login_note=None``; providers without a
    ban envelope pass ``ban_until=None``.

    No fallback handler: every provider's ``_TERMINAL_HANDLERS`` is total
    over its own ``RetryKind``, and this is only reached under ``if error:``
    (``classify_error`` returns None solely for a falsy error). A missing key
    is therefore a provider bug, and ``KeyError`` names it at once instead of
    quietly answering a wrong status.
    """
    handlers[classify_fn(error)](app, item, prep, token, error, login_note,
                                 ban_until)


# --- Stream bridge (producer/consumer shared by every provider) -------------

# Read-budgets covered by the consumer watchdog's first-payload wait (see
# ``drain_stream_queue``) and by the buffered watchdog sizing (see
# ``new_stream_context``): one for the attempt itself, the rest covering
# session create + PoW solve + one recovery pass (browser login refresh /
# Obscura warmup). Deliberately a fixed multiple, not a derived budget: the
# engine has no view of ``max_retries`` and the worst case is unbounded anyway.
_BUFFERED_PRODUCER_READS = 3

# A producer running silently is normal: session create, PoW solve, and one
# recovery pass (a ~180s browser login, an Obscura warmup) all emit nothing
# until the reply lands. A proxy in front does not know that -- nginx drops an
# idle response body at 60s (proxy_read_timeout), Cloudflare at 100s -- so a
# healthy slow reply died without even an error frame. The consumer therefore
# waits in slices this long and writes one SSE comment frame per slice: the
# one line every SSE client ignores.
_KEEPALIVE_INTERVAL = 15.0
_KEEPALIVE_FRAME = b": ping\n\n"


def stream_chunk(cid: str, created: int, model: str,
                 delta: dict, finish: str | None = None,
                 extra: dict | None = None) -> bytes:
    """One OpenAI-style ``chat.completion.chunk`` SSE frame.

    ``extra`` merges top-level keys into the payload (dry run stamps
    ``{"dry_run": True}``); absent/empty keeps the frame byte-identical.
    """
    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
               "model": model, "choices": [{"index": 0, "delta": delta,
                                            "finish_reason": finish}]}
    if extra:
        payload.update(extra)
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")


def tool_finish_chunks(chunk_fn: Callable[..., bytes], calls,
                       truncated: bool = False):
    """Tool-call deltas + finish chunk; ``length`` when a call was cut off.

    Every delta carries its own ``index`` -- its position in ``calls``, in
    emission order -- because that is the only field that tells a consumer
    which of N parallel calls a delta belongs to (each delta is otherwise a
    complete, standalone call). OpenAI's streaming contract makes ``index``
    mandatory on ``tool_calls`` deltas: ``openai-python`` validates each
    chunk against its pydantic model and raises when it is absent, i.e.
    inside the stream, mid agent-loop turn; accumulators keyed by ``index``
    cannot merge parallel calls at all. The non-stream path
    (``core.openai._openai_response``) needs no index: there the calls
    arrive as one complete list, so position is implicit.
    """
    if calls:
        for i, call in enumerate(calls):
            yield chunk_fn({"tool_calls": [{
                "index": i, "id": call["id"], "type": "function",
                "function": {"name": call["function"]["name"],
                             "arguments": call["function"]["arguments"]}}]})
        yield chunk_fn({}, "tool_calls")
    else:
        yield chunk_fn({}, "length" if truncated else "stop")


def usage_chunk(cid: str, created: int, model: str, prep: dict,
                content: str, thinking: str,
                usage_total: int | None = None,
                extra: dict | None = None) -> bytes:
    """Final OpenAI-style usage chunk (stream_options.include_usage).

    Emitted with empty choices before [DONE] so cost-tracking harnesses see
    the same prompt/completion totals as the non-stream response. Uses the
    shared ``core.openai.build_usage`` estimate (upstream never reports
    prompt tokens); ``usage_total`` wins for completion tokens when seen.
    ``extra`` merges top-level keys into the payload (dry run stamps
    ``{"dry_run": True}``).
    """
    try:
        from .openai import build_usage as _build_usage
        usage = _build_usage(prep, content or "", thinking or "", usage_total)
    except Exception:  # pragma: no cover - build_usage is pure and local
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
               "model": model, "choices": [], "usage": usage}
    if extra:
        payload.update(extra)
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")


# In-band SSE error frames (OpenAI ``{"error": {message, type, code}}``)
# must be built in ONE place: three call sites in ``drain_stream_queue``
# (watchdog timeout, propagated HTTPException, any other queued failure)
# previously hand-rolled the dict and drifted -- some had no ``type`` at
# all, and ``HTTPException.detail`` is untyped in FastAPI, so a dict detail
# serialized as a nested JSON object and failed strict SDK deserialization
# of ``error.message`` (``str`` in the OpenAI schema).
#
# ``type`` scheme (chosen so LiteLLM / SDK retry classifiers bucket these
# the way they bucket api.openai.com):
#   * 4xx -> ``invalid_request_error``: the caller/bridge rejected the
#     request (auth, unknown model, rate limit) -- not worth retrying as-is.
#   * 5xx, and frames raised locally with no HTTP status (the watchdog
#     timeout, a generic consumer failure) -> ``upstream_error``: the web
#     upstream broke or stalled, which is a "retry later" bucket. OpenAI's
#     own 5xx slug is ``server_error``; ``upstream_error`` is the honest
#     name here because every 5xx this bridge raises originates upstream.
# ``code`` stays the numeric HTTP status when one is known and ``null``
# when the failure never had one (OpenAI types it as optional).
def sse_error_frame(message, status_code=None) -> bytes:
    """One OpenAI-shaped SSE error frame; ``message`` always a JSON string.

    Clamped and de-tokenized here, not at the call sites: the response has no
    Content-Length yet, so an unbounded frame is both a reflected megabyte and
    the one path a locally-raised error takes upstream text into the client.
    A frame raised on the consumer side (the watchdog timeout, a generic
    failure) carries whatever ``str(exc)`` held, and the producer tails hand us
    exactly the upstream body text.
    """
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        status = 0
    kind = "invalid_request_error" if 400 <= status < 500 else "upstream_error"
    return ("data: " + json.dumps(
        {"error": {"message": scrub_upstream_text(message, 800),
                    "type": kind, "code": status or None}},
        ensure_ascii=False) + "\n\n").encode()


def wants_usage(prep: dict) -> bool:
    """True when the client passed stream_options.include_usage=true."""
    try:
        opts = (prep or {}).get("stream_options") or {}
        return bool(isinstance(opts, dict) and opts.get("include_usage"))
    except Exception:
        return False


async def parse_stream_calls(prep: dict, content: str,
                             parse_fn: Callable | None = None,
                             truncated_fn: Callable | None = None):
    """Tool-parse buffered content (tools requested); log when none found.

    Misses log at warning with the parse_tool_calls_ex diagnostics trail
    (which format almost matched, unknown names, oversize drops) instead of
    a bare debug preview — a stop-with-raw-markup reply is otherwise
    indistinguishable from a plain refusal. The ``parse_fn`` seam stays for
    tests; diagnostics always come from the real parser for logging only.
    No core->server import: the default resolves via the shared
    ``core.openai._resolve_parse_fn`` soft lookup (server patch wins when
    loaded, else the real parser).
    """
    from .openai import _resolve_parse_fn
    parse_fn, _real_parse, _real_ex = _resolve_parse_fn(parse_fn)
    if truncated_fn is None:
        from .tools import looks_truncated_tool_attempt as truncated_fn  # type: ignore[no-redef]
    limit = prep.get("tool_args_max_chars")
    forgiving = bool(prep.get("forgiving_toolcalls", False))
    calls: list[dict] | None = None
    remaining: str = content
    dropped: list[int] | None = None
    diagnostics: list[dict] | None = None
    kwargs: dict = {"forgiving": forgiving}
    if limit:
        kwargs["max_args_chars"] = limit
    if parse_fn is _real_parse and _real_ex is not None:
        res = await asyncio.to_thread(
            _real_ex, content, prep["tools"], **kwargs)
        calls, remaining = res.calls, res.rest
        dropped, diagnostics = res.dropped, res.diagnostics
    else:
        calls, remaining = await asyncio.to_thread(
            parse_fn, content, prep["tools"], **kwargs)
    truncated = False
    if not calls:
        try:
            truncated = bool(truncated_fn(content, prep["tools"], forgiving=forgiving))
        except Exception:
            truncated = False
        if truncated:
            logger.warning("stream tool call looks truncated model=%s preview=%.200s",
                           prep.get("model"), (content or "")[:200])
        else:
            # Rich miss log, single home in core.openai._log_tool_miss: the
            # diagnostics trail says unknown-name vs oversize vs format drift.
            from .openai import _log_tool_miss
            await asyncio.to_thread(_log_tool_miss, prep, content,
                                    diagnostics, where="stream")
    elif dropped:
        from .openai import _log_tool_dropped
        _log_tool_dropped(prep, dropped, where="stream")
    return calls, remaining, truncated


async def drain_stream_queue(out: asyncio.Queue, prep: dict, live: bool,
                             cid: str, created: int, model: str,
                             consumer_timeout: float,
                             parse_fn: Callable | None = None,
                             truncated_fn: Callable | None = None):
    """Shared consumer: buffered and live differ only in chunk mapping.

    Buffered emits thinking + content once the whole reply lands; live
    already forwarded the deltas and only needs the close/tool tail.
    Error/timeout handling is identical. Yields SSE ``bytes`` frames.

    The FIRST get is waited on with the producer-sized budget
    (``consumer_timeout * _BUFFERED_PRODUCER_READS``): no mode has a payload
    yet, and the producer is doing the same session + PoW + recovery work in
    both, which can outlast a single read. From the first payload on the plain
    ``consumer_timeout`` applies -- an event resets the clock, so it measures
    genuine silence only.

    Each wait is sliced at ``_KEEPALIVE_INTERVAL`` and every slice that
    expires with budget left emits a comment frame, so the client and the
    proxies between us see bytes through a silent producer stretch. A slice
    never crosses the budget, so a sub-interval budget (every hermetic test)
    produces exactly the frames it did before the slices existed.
    """
    chunk = lambda delta, finish=None: stream_chunk(cid, created, model, delta, finish)  # noqa: E731
    yield chunk({"role": "assistant"})
    got_first = False
    loop = asyncio.get_running_loop()
    while True:
        budget = consumer_timeout if got_first else (
            consumer_timeout * _BUFFERED_PRODUCER_READS)
        deadline = loop.time() + budget
        timed_out = False
        while True:
            slice_left = min(_KEEPALIVE_INTERVAL, deadline - loop.time())
            try:
                kind, payload = await asyncio.wait_for(out.get(), timeout=slice_left)
                break
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    timed_out = True
                    break
                yield _KEEPALIVE_FRAME
        if timed_out:
            yield sse_error_frame("upstream timed out")
            yield b"data: [DONE]\n\n"
            return
        got_first = True
        if kind == "thinking":
            yield chunk({"reasoning_content": payload})
            continue
        if kind == "content":
            yield chunk({"content": payload})
            continue
        if kind == "done":
            if not live:
                thinking, content = payload["thinking"], payload["content"]
                truncated = False
                if prep["tools"]:
                    calls, remaining, truncated = await parse_stream_calls(
                        prep, content, parse_fn, truncated_fn)
                else:
                    calls, remaining = None, content
                if thinking:
                    yield chunk({"reasoning_content": thinking})
                if remaining:
                    yield chunk({"content": remaining})
                for piece in tool_finish_chunks(chunk, calls, truncated):
                    yield piece
                if wants_usage(prep):
                    # payload["content"]/["thinking"] already unpacked at 1127.
                    yield usage_chunk(cid, created, model, prep, content,
                                      thinking, payload.get("usage_total"))
            elif prep.get("tools"):
                calls, remaining, truncated = await parse_stream_calls(
                    prep, payload["content"], parse_fn, truncated_fn)
                if remaining:
                    yield chunk({"content": remaining})
                for piece in tool_finish_chunks(chunk, calls, truncated):
                    yield piece
                if wants_usage(prep):
                    yield usage_chunk(cid, created, model, prep,
                                      payload.get("content", ""),
                                      payload.get("thinking", ""),
                                      payload.get("usage_total"))
            else:
                yield chunk({}, "stop")
                if wants_usage(prep):
                    yield usage_chunk(cid, created, model, prep,
                                      payload.get("content", ""),
                                      payload.get("thinking", ""),
                                      payload.get("usage_total"))
            yield b"data: [DONE]\n\n"
            return
        if kind == "http_error":
            err = payload
            yield sse_error_frame(err.detail, err.status_code)
            yield b"data: [DONE]\n\n"
            return
        yield sse_error_frame(payload)
        yield b"data: [DONE]\n\n"
        return


def new_stream_context(stream_timeout: float,
                       consumer_grace: float = 10.0, buffered: bool = False):
    """Allocate one stream's (cid, created, queue, cancel_event, timeouts).

    Thin constructor so providers share the exact queue/cancel/timeout shape:
    the consumer wait outlasts the blocking stream timeout so the async reader
    never times out while the producer is still on a valid long reply.

    ``stream_timeout`` is ONE socket read, which is the right budget for live
    mode (every delta resets the per-``get`` clock, so it only measures
    genuine silence) but far too small for buffered mode, where the producer
    emits nothing until the whole reply lands: the timer then covers the ENTIRE
    producer -- session create (30s) + PoW solve + the reply + up to
    ``max_retries`` recovery passes, any of which can include a ~180s browser
    login refresh and an Obscura warmup. Sizing it from one read killed a
    healthy-but-slow upstream with a client-visible
    ``{"error": {"message": "upstream timed out"}}`` and threw away a valid
    answer. ``buffered=True`` therefore multiplies the budget by
    ``_BUFFERED_PRODUCER_READS`` (setup + one full recovery pass of headroom);
    live mode passes ``False`` and keeps the single-read budget for every get
    AFTER the first payload. Before any payload the consumer widens the wait
    in every mode (see ``drain_stream_queue``): live has no delta yet either.
    """
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    out: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    consumer_timeout = float(stream_timeout) + float(consumer_grace)
    if buffered:
        consumer_timeout *= _BUFFERED_PRODUCER_READS
    return cid, created, out, cancel_event, consumer_timeout


def emit_to_loop(loop: asyncio.AbstractEventLoop, out: asyncio.Queue, item: tuple) -> None:
    """Thread-safe queue put; swallows RuntimeError after loop close."""
    try:
        loop.call_soon_threadsafe(out.put_nowait, item)
    except RuntimeError:
        pass


def live_emit_pair(on_content, on_thinking):
    """One LiveEmitTracker + wrapped callbacks (shared _run_completion_core prelude).

    Returns ``(tracker, wrapped_content, wrapped_thinking)``. Wrapping is
    unconditional so the partial-emit guard sees live deltas even when a
    caller passes callbacks selectively; call sites pass
    ``wrapped if original is not None else None`` into ``_do_attempt``.
    """
    tracker = LiveEmitTracker()
    return tracker, tracker.wrap(on_content), tracker.wrap(on_thinking)


def live_retry_predicate(classify_fn, kind, tracker):
    """One predicate factory so NO row retries after a live emit.

    The guard used to sit on the overload row alone, on the assumption that
    the other kinds fail before any content. ``drain_attempt`` makes
    content-then-error a legal ordering, and the completion POST is not
    idempotent (the upstream already appended a turn), so a recovery fired
    after a delta re-POSTs the same payload on the same session/chat and
    re-streams from delta 0 into a stream the client already read: duplicated
    text, or two interleaved halves of one ``tool_calls`` JSON object, then a
    clean ``[DONE]`` and no error to stop the harness executing the corrupt
    call. Every provider row goes through its local ``_retryable(kind)``,
    which returns this.
    """
    def _ok(e):
        return classify_fn(e) == kind and not tracker.emitted

    return _ok


def ban_until_of(outcome) -> float | None:
    """Best-effort ban-expiry from an attempt outcome (None when absent)."""
    try:
        return getattr(outcome, "ban_until", None)
    except Exception:
        return None


def new_producer_context(prep: dict, client_timeout: float,
                         consumer_grace: float = 10.0, buffered: bool = False):
    """Shared _stream_completion prelude: context + model + emit closures.

    Owns ``new_stream_context`` + ``prep["model"]`` + the running loop +
    ``_emit``/``_put`` pair so the three providers share one queue/cancel/
    emit shape; ``stream_producer`` is its only caller. ``buffered``
    is forwarded to the watchdog sizing (see ``new_stream_context``).
    """
    cid, created, out, cancel_event, consumer_timeout = new_stream_context(
        client_timeout, consumer_grace, buffered=buffered)
    model = prep["model"]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def _emit(item: tuple) -> None:
        if loop is not None:
            emit_to_loop(loop, out, item)
        else:  # pragma: no cover - defensive (no running loop in tests)
            try:
                out.put_nowait(item)
            except Exception:
                pass

    def _put(kind: str, text: str) -> None:
        if text:
            _emit((kind, text))

    return cid, created, out, cancel_event, consumer_timeout, model, _emit, _put


async def consume_stream(out: asyncio.Queue, prep: dict, live: bool,
                         cid: str, created: int, model: str,
                         consumer_timeout: float,
                         parse_fn: Callable | None = None,
                         truncated_fn: Callable | None = None,
                         cancel_event: threading.Event | None = None):
    """Shared _stream_completion tail: drain + cancel on close.

    Owns the ``drain_stream_queue`` consumer and the ``cancel_event.set()``
    that frees the account slot on disconnect/completion. Yields SSE bytes.
    """
    try:
        async for frame in drain_stream_queue(
                out, prep, live, cid, created, model, consumer_timeout,
                parse_fn, truncated_fn):
            yield frame
    finally:
        try:
            if cancel_event is not None:
                cancel_event.set()
        except Exception:
            pass


async def stream_producer(run, *, prep: dict, timeout: float, live: bool,
                          grace: float = 10.0, parse_fn: Callable | None = None,
                          truncated_fn: Callable | None = None):
    """Shared _stream_completion skeleton the three adapters copied.

    Owns context allocation (``new_producer_context``, with the watchdog sized
    by ``buffered=not live``), the producer exception ladder, the bounded
    submit (``_STREAM_EXECUTOR.submit`` plus the raw-thread fallback when the
    executor is already shut down) and the ``consume_stream`` tail.
    ``run(live, put, cancel_event)`` performs ONE completion on the producer
    thread and returns the result dict; ``put(kind, text)`` publishes one live
    delta. The provider keeps its failover entrypoint, its timeout global and
    its grace global (read at generator-body time, so tests can patch them per
    module).
    """
    (cid, created, out, cancel_event, consumer_timeout, model,
     _emit, _put) = new_producer_context(prep, timeout, grace, buffered=not live)

    def produce() -> None:
        try:
            result = run(live, _put, cancel_event)
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            # Scrubbed because this is the one emit path that does NOT go
            # through the terminal ladder: the client helpers raise with a raw
            # slice of the upstream body in the message, and that body is the
            # one most likely to echo the Authorization header it was given.
            # 800 is the scrubber's default and a superset of that 200-char
            # slice, so no diagnostic text is lost relative to today.
            _emit(("error", scrub_upstream_text(exc, 800)))

    # Bounded producer: one account-holding thread per stream, capped by the
    # shared executor so a parallel-subagent burst queues instead of piling
    # unbounded raw threads (each with account + PoW CPU).
    try:
        _STREAM_EXECUTOR.submit(produce)
    except RuntimeError:
        # Executor shut down (interpreter exit): fall back to a raw thread.
        threading.Thread(target=produce, daemon=True).start()
    async for frame in consume_stream(
            out, prep, live, cid, created, model, consumer_timeout,
            parse_fn, truncated_fn, cancel_event):
        yield frame
