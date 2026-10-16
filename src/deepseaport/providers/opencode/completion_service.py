"""Opencode completion engine: thin adapter over ``deepseaport.core.engine``.

No accounts, no tokens, no PoW, no WAF: one completion is create-session ->
one turn -> best-effort delete-session. Retries are limited to transient
overload/empty turns (fresh session each); serve-down, auth, and validation
errors are terminal with an actionable message. Streaming is live
token-by-token off the serve's ``GET /api/event`` SSE deltas (forwarded as
OpenAI chunks as they arrive); when the live tail cannot start, the producer
falls back to the blocking turn fanned out as SSE chunks. Sessions are pinned
to the caller's workdir via the create body (``location.directory``) so file
tools run where the harness expects. The canonical monkeypatch surface is
this module.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from . import client as OC
from . import protocol as P
from ...core import engine as _engine
from ...core.engine import drive_retries
# Shared shim re-export for compat (non-stream route uses server's copy).
from ...core.openai import _openai_response  # noqa: F401
from ...core.tools import (
    looks_truncated_tool_attempt,
    parse_tool_calls as _parse_tool_calls_local,
)
from ...server_concurrency import (
    _CONSUMER_GRACE_SECONDS,
    _STREAM_EXECUTOR,
    _StreamCancelled,
)

logger = logging.getLogger("deepseaport.server")


def _ensure_managed_serve() -> str | None:
    """Boot/reuse the isolated managed serve; None when unavailable.

    Lazy import keeps this module the canonical monkeypatch surface and
    avoids import cycles (managed_serve only needs stdlib + config).
    """
    try:
        from . import managed_serve as _ms
        return _ms.ensure()
    except Exception:
        return None


class RetryKind(enum.Enum):
    """Single classifier for attempt errors (retry + terminal share it)."""

    OVERLOAD = "overload"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_INVALID = "upstream_invalid"
    UPSTREAM = "upstream"


_error_text = _engine.error_text


def classify_error(error: str | None) -> RetryKind | None:
    """Map an attempt error string to its retry/terminal kind."""
    if not error:
        return None
    low = error.lower()
    if P.is_serve_down_error(error) or "opencode_serve_down" in low:
        return RetryKind.UPSTREAM
    if any(k in low for k in ("rate_limit", "rate limit", "429", "too many",
                              "quota", "freeusagelimit", "free_usage")):
        return RetryKind.RATE_LIMIT
    if any(k in low for k in ("http_422", "http 422", "http_400", "http 400",
                              "unprocessable", "validation")):
        return RetryKind.UPSTREAM_INVALID
    if any(k in low for k in ("overload", "overloaded", "server busy",
                              "server_busy", "try again", "timed out", "timeout",
                              "500", "502", "503", "504", "bad gateway",
                              "service unavailable", "temporarily unavailable",
                              "fetch failed", "failed to fetch", "network error",
                              "socket hang up", "econnreset", "etimedout",
                              "empty completion", "opencode_empty")):
        return RetryKind.OVERLOAD
    return RetryKind.UPSTREAM


def _app_settings(app: FastAPI):
    return _engine.app_settings(app)


def _terminal_upstream(app, item, prep, token, error, login_note, stream_ban_until) -> None:
    # login_note ignored (dispatch uniformity). Serve-down stays a distinct
    # 502 with an actionable message (managed-serve boot), not generic 502.
    if P.is_serve_down_error(error):
        from fastapi import HTTPException as _HTTP

        raise _HTTP(status_code=502, detail=f"opencode serve down: {error[:800]}")
    _engine.raise_upstream("Opencode", error)


# RATE_LIMIT/OVERLOAD/UPSTREAM_INVALID are the label-only handlers every
# provider shares (core.engine); UPSTREAM is overridden because serve-down
# needs its own actionable message.
_TERMINAL_HANDLERS: dict[RetryKind, Callable[..., None]] = _engine.generic_terminal_handlers(
    RetryKind, "Opencode", upstream=_terminal_upstream)


def _raise_for_terminal_error(app, item, prep, token, error, login_note, stream_ban_until) -> None:
    """Map a terminal completion error to its HTTP status. Always raises."""
    _engine.dispatch_terminal(classify_error, _TERMINAL_HANDLERS,
                              app, item, prep, token,
                              error, login_note, stream_ban_until)


def _attempt(prompt: str, model: str, timeout: float,
             on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             variant: str | None = None,
             ) -> tuple[str, str, str | None, int]:
    """One session lifecycle -> (content, thinking, error, usage).

    Never raises transport errors: they become the ``error`` string the
    retry table classifies. ``cancel_event`` aborts between checkpoints.
    ``variant`` is the one-shot reasoning effort; when the serve rejects
    it as unknown the turn is resent once without it.
    """
    provider_id, model_id = P.split_model(model)
    agent = P.agent_name()
    session_id: str | None = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            return "", "", _error_text("CANCELLED", "client disconnected"), 0
        try:
            session_id = OC.create_session(timeout=min(30.0, timeout))
        except _StreamCancelled:
            raise
        except Exception as exc:
            err = _error_text("", str(exc))
            if P.is_serve_down_error(err) and _ensure_managed_serve():
                try:
                    session_id = OC.create_session(timeout=min(30.0, timeout))
                except _StreamCancelled:
                    raise
                except Exception as exc2:
                    return "", "", _error_text("", str(exc2)), 0
            else:
                return "", "", err, 0
        if cancel_event is not None and cancel_event.is_set():
            return "", "", _error_text("CANCELLED", "client disconnected"), 0
        try:
            result = OC.send_message(session_id, provider_id, model_id,
                                     agent, prompt, timeout=timeout,
                                     variant=variant)
        except _StreamCancelled:
            raise
        except Exception as exc:
            err = _error_text("", str(exc))
            if variant and "variant" in err.lower():
                logger.info("opencode variant %r rejected, retrying without",
                            variant)
                try:
                    result = OC.send_message(
                        session_id, provider_id, model_id,
                        agent, prompt, timeout=timeout)
                except _StreamCancelled:
                    raise
                except Exception as exc2:
                    return "", "", _error_text("", str(exc2)), 0
            else:
                return "", "", err, 0
        content = result.get("content") or ""
        thinking = result.get("thinking") or ""
        if on_content is not None and content:
            try:
                on_content(content)
            except Exception:
                pass
        if on_thinking is not None and thinking:
            try:
                on_thinking(thinking)
            except Exception:
                pass
        return content, thinking, None, int(result.get("usage_total") or 0)
    finally:
        try:
            OC.delete_session(session_id)
        except Exception:
            pass


def _attempt_live(prompt: str, model: str, timeout: float,
                   on_content=None, on_thinking=None,
                   cancel_event: threading.Event | None = None,
                   variant: str | None = None,
                   ) -> tuple[str, str, str | None, int]:
    """Live session lifecycle via /api/event -> (content, ...).

    Deltas ride ``on_content``/``on_thinking`` as they arrive (true
    token-by-token streaming); the returned content is the authoritative
    final assembled from the session timeline. When the live tail
    fails before emitting anything, falls back to the blocking
    :func:`_attempt` on the same session shape (new session) so a transient
    SSE failure does not fail the request. Same error-string contract as
    :func:`_attempt` (never raises transport errors except cancellation).
    """
    provider_id, model_id = P.split_model(model)
    agent = P.agent_name()
    session_id: str | None = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            return "", "", _error_text("CANCELLED", "client disconnected"), 0
        try:
            session_id = OC.create_session(timeout=min(30.0, timeout))
        except _StreamCancelled:
            raise
        except Exception as exc:
            err = _error_text("", str(exc))
            if P.is_serve_down_error(err) and _ensure_managed_serve():
                try:
                    session_id = OC.create_session(timeout=min(30.0, timeout))
                except _StreamCancelled:
                    raise
                except Exception as exc2:
                    return "", "", _error_text("", str(exc2)), 0
            else:
                return "", "", err, 0
        if cancel_event is not None and cancel_event.is_set():
            return "", "", _error_text("CANCELLED", "client disconnected"), 0
        emitted = {"content": False, "thinking": False}

        def _track_content(t: str) -> None:
            if t:
                emitted["content"] = True
            if on_content is not None:
                try:
                    on_content(t)
                except Exception:
                    pass

        def _track_thinking(t: str) -> None:
            if t:
                emitted["thinking"] = True
            if on_thinking is not None:
                try:
                    on_thinking(t)
                except Exception:
                    pass

        try:
            result = OC.stream_message(
                session_id, provider_id, model_id, agent, prompt,
                timeout=timeout, variant=variant,
                on_content=(_track_content if on_content is not None else None),
                on_thinking=(_track_thinking if on_thinking is not None else None),
                cancel_event=cancel_event)
        except _StreamCancelled:
            raise
        except Exception as exc:
            err = _error_text("", str(exc))
            # Variant rejection: resend once without it (live path).
            if variant and "variant" in err.lower():
                logger.info("opencode variant %r rejected, retrying without",
                            variant)
                try:
                    result = OC.stream_message(
                        session_id, provider_id, model_id, agent, prompt,
                        timeout=timeout,
                        on_content=(_track_content if on_content is not None else None),
                        on_thinking=(_track_thinking if on_thinking is not None else None),
                        cancel_event=cancel_event)
                except _StreamCancelled:
                    raise
                except Exception as exc2:
                    return "", "", _error_text("", str(exc2)), 0
            elif not emitted["content"] and not emitted["thinking"]:
                # Live tail never started (SSE/connect failure): fall back to
                # the blocking POST on a fresh session so the request still
                # succeeds with buffered (chunked) SSE downstream.
                # Scrubbed before the 100-char slice: a slice is not a redactor,
                # and `err` is a raw upstream message. serve is localhost with
                # no account credential, but it is still a process whose prose
                # reaches the log file and any shipper reading it.
                logger.info("opencode live tail failed (%s), falling back to "
                            "blocking", _engine.scrub_upstream_text(err, 100))
                try:
                    OC.delete_session(session_id)
                except Exception:
                    pass
                session_id = None
                return _attempt(prompt, model, timeout,
                                on_content=on_content, on_thinking=on_thinking,
                                cancel_event=cancel_event, variant=variant)
            else:
                return "", "", err, 0
        content = result.get("content") or ""
        thinking = result.get("thinking") or ""
        # Live deltas already rode the callbacks; do not re-emit the full
        # text here (would duplicate). The tracker in _run_completion_core
        # already marked emitted via the wrapped callbacks.
        return content, thinking, None, int(result.get("usage_total") or 0)
    finally:
        try:
            OC.delete_session(session_id)
        except Exception:
            pass


def _run_completion_core(
    app: FastAPI,
    item,
    prep: dict,
    on_content=None,
    on_thinking=None,
    cancel_event: threading.Event | None = None,
    live: bool = False,
) -> dict:
    """Shared retry engine for buffered + live streaming (item/pool unused).

    ``item`` is accepted for route-signature compatibility (the opencode
    provider needs no accounts) and ignored. One stale-session recreate is
    attempted inline before the overload retry table runs. ``live=True``
    tails the serve's ``/api/event`` deltas (token-by-token); False uses the
    blocking prompt-then-poll turn (buffered).
    """
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    timeout = P.opencode_timeout(OC.DEFAULT_TIMEOUT)
    prompt = str(prep.get("prompt") or "")
    model = str(prep.get("model") or "")
    variant = str(prep.get("reasoning_effort") or "") or None

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if _cancelled():
        raise _StreamCancelled()
    if not prompt.strip():
        _engine.raise_upstream_invalid("Opencode", prep, "empty prompt")

    content = think = ""
    error: str | None = None
    usage_total = 0
    _overload_waits = 0
    _emit_tracker, _wrapped_content, _wrapped_thinking = _engine.live_emit_pair(
        on_content, on_thinking)

    def _do_attempt() -> None:
        nonlocal content, think, error, usage_total
        if live:
            outcome = _attempt_live(
                prompt, model, timeout,
                on_content=(_wrapped_content if on_content is not None else None),
                on_thinking=(_wrapped_thinking if on_thinking is not None else None),
                cancel_event=cancel_event, variant=variant)
        else:
            outcome = _attempt(
                prompt, model, timeout,
                on_content=(_wrapped_content if on_content is not None else None),
                on_thinking=(_wrapped_thinking if on_thinking is not None else None),
                cancel_event=cancel_event, variant=variant)
        content, think, error, usage_total = outcome

    _do_attempt()
    # Stale-session race: one recreate-and-resend outside the retry budget.
    if error is not None and P.is_session_error(error) and not _cancelled():
        # Scrubbed before the 80-char slice: a slice is a length guard, not a
        # secret guard, and `error` is a raw upstream message.
        logger.info("opencode session race (%s), recreating once",
                    _engine.scrub_upstream_text(error, 80))
        _do_attempt()

    def _recover_overload() -> None:
        # Back off before the fresh-session retry: the free tier recovers
        # seconds later, immediate hammering does not. Cancel-aware sleep.
        nonlocal _overload_waits
        _overload_waits += 1
        wait = min(2.0 * _overload_waits, 8.0)
        logger.info("opencode overloaded (%s), retrying in %.0fs "
                    "(retries left %s)",
                    _engine.scrub_upstream_text(error, 80), wait, max_retries)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if _cancelled():
                return
            time.sleep(0.1)
        if _cancelled():
            return
        _do_attempt()

    # Same shape the two web providers use, so the one-row table here reads the
    # same way and a second row cannot be added without the guard.
    def _retryable(kind) -> Callable[[str | None], bool]:
        def _ok(e: str | None) -> bool:
            return (classify_error(e) == kind
                    and not _emit_tracker.emitted)

        return _ok

    drive_retries(
        max_retries,
        _cancelled,
        lambda: error,
        (
            ("overload", _retryable(RetryKind.OVERLOAD), _recover_overload),
        ),
    )
    if _cancelled():
        raise _StreamCancelled()
    if error:
        _raise_for_terminal_error(app, item, prep, None, error, None, None)
    if not content and not think:
        raise _engine.empty_response_error("Opencode")
    return {"content": content, "thinking": think, "usage_total": usage_total}


def _run_completion(app: FastAPI, item, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion (no failover: no accounts to fail over to)."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


def _complete_with_failover_sync(app: FastAPI, pool, prep: dict,
                                 first_item,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Route-compatible entrypoint; pool/failover are no-ops (no accounts)."""
    return _run_completion_core(
        app, first_item, prep, on_content=on_content, on_thinking=on_thinking,
        cancel_event=cancel_event)


async def _stream_completion(app: FastAPI, pool, item, prep: dict):
    """Async generator: live token-by-token deltas, tools parsed at end.

    The producer runs the session lifecycle in the shared stream executor;
    the consumer is engine-owned (chunk framing, tool tail, timeout/errors
    identical to every provider). Live mode streams ``/event`` deltas as
    they arrive; content streams live only when no tools were requested
    (otherwise a tool_calls reply would also leak as user-visible text) —
    thinking always streams. ``stream_mode=buffered`` keeps the historical
    blocking fan-out; live falls back to it when the SSE tail cannot start.
    Queue/cancel/emit shape + consumer are engine-owned
    (``new_producer_context`` / ``consume_stream``); the live vs buffered
    session lifecycle stays here (no accounts, stale-session recreate +
    overload sleep preserved).
    """
    try:
        settings = _app_settings(app)
        live = str(getattr(settings, "stream_mode", "live")).lower() == "live"
    except Exception:
        live = True
    # `buffered=not live` is the cross-provider contract, not a detail
    # (ARCHITECTURE.md, "Provider architecture": "stream_mode: buffered must
    # call new_stream_context(..., buffered=True) so the consumer watchdog
    # covers the whole producer budget, not one read"); DeepSeek passes the
    # same argument at providers/deepseek/completion_service.py:1156.
    #
    # Without it the consumer waits 180 + 10 s while this producer's own
    # deadline is create(30) + turn(180) + one overload retry(180) = 390 s, so
    # a healthy-but-slow serve was answered with
    # {"error": {"message": "upstream timed out"}} and a valid, fully
    # assembled reply was thrown away. Live mode keeps the single-read budget
    # exactly: every delta resets the per-get clock.
    (cid, created, out, cancel_event, consumer_timeout, model,
     _emit, _put) = _engine.new_producer_context(
        prep, OC.DEFAULT_TIMEOUT, _CONSUMER_GRACE_SECONDS, buffered=not live)

    def _produce(*, live_produce: bool) -> None:
        try:
            if live_produce:
                result = _run_completion_core(
                    app, item, prep, cancel_event=cancel_event, live=True,
                    on_content=(None if prep.get("tools") else lambda t: _put("content", t)),
                    on_thinking=lambda t: _put("thinking", t))
            else:
                result = _run_completion_core(
                    app, item, prep, cancel_event=cancel_event, live=False)
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
            _emit(("error", _engine.scrub_upstream_text(exc, 800)))

    def produce_buffered():
        _produce(live_produce=False)

    def produce_live():
        _produce(live_produce=True)

    try:
        _STREAM_EXECUTOR.submit(produce_live if live else produce_buffered)
    except RuntimeError:
        threading.Thread(
            target=produce_live if live else produce_buffered,
            daemon=True).start()
    async for frame in _engine.consume_stream(
            out, prep, live, cid, created, model, consumer_timeout,
            _parse_tool_calls_local, looks_truncated_tool_attempt,
            cancel_event):
        yield frame
