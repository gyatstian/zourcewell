"""Qwen completion engine: failover, retries, chat lifecycle.

Qwen-specific adapter over the provider-neutral mechanics in
``deepseaport.core.engine``. Unlike DeepSeek there is no PoW, no wasm, no
browser token refresh and no chat-list management: a completion is
create-chat -> POST /api/v2/chat/completions -> best-effort delete-chat.
WAF recovery is a bridge warmup (Aliyun anti-bot), not a challenge solve.
The canonical monkeypatch surface is this module
(``providers.qwen.completion_service``).
"""

from __future__ import annotations

import asyncio
import datetime
import enum
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from . import client as QC
from . import protocol as P
from ...accounts import AccountPool, PooledAccount
from ...config import Settings, load_settings
from ...core import engine as _engine
from ...core.engine import (
    AttemptOutcome as _AttemptOutcome,
    drive_retries,
    slot_releaser as _slot_releaser,
)
from ...core.obscura_bridge import ObscuraBridge
# Shared implementation: _openai_response lives in core.openai so every
# web2api provider reuses one response shim (re-exported here for compat).
from ...core.openai import _openai_response  # noqa: F401
from ...core.tools import (
    looks_truncated_tool_attempt,
    parse_tool_calls as _parse_tool_calls_local,
)
from ...server_concurrency import (
    FAILOVER_ACQUIRE_TIMEOUT,
    INVALID_TOKEN_COOLDOWN_SECONDS,
    _CLEANUP_EXECUTOR,
    _CONSUMER_GRACE_SECONDS,
    _STREAM_EXECUTOR,
    _StreamCancelled,
)

logger = logging.getLogger("deepseaport.server")


class RetryKind(enum.Enum):
    """Single classifier for attempt errors (retry + terminal share it).

    Retryable: WAF, INVALID_TOKEN, OVERLOAD.
    Terminal: BAN, RATE_LIMIT, UPSTREAM_INVALID, UPSTREAM (generic 502);
    INVALID_TOKEN is retryable but its terminal handler (401) is also
    table-driven. One ``classify_error`` maps text -> kind; retry and
    terminal tables consume the same value.
    """

    WAF = "waf"
    INVALID_TOKEN = "invalid_token"
    OVERLOAD = "overload"
    BAN = "ban"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_INVALID = "upstream_invalid"
    UPSTREAM = "upstream"


_error_text = _engine.error_text


def classify_error(error: str | None) -> RetryKind | None:
    """Map an attempt error string to its retry/terminal kind.

    String families map per the Qwen recon (see README error table):
    WAF/challenge -> WAF; 401/unauthorized -> INVALID_TOKEN; rate limits ->
    RATE_LIMIT; 422/400 -> UPSTREAM_INVALID; overload/busy/timeout/502 ->
    OVERLOAD; generic ban words -> BAN; anything else -> UPSTREAM. The retry
    table additionally guards OVERLOAD with the "partial live emit" flag.
    """
    if not error:
        return None
    low = error.lower()
    if "waf" in low or "403" in low or "challenge" in low:
        return RetryKind.WAF
    if any(k in low for k in ("401", "unauthorized", "invalid", "authorization")):
        return RetryKind.INVALID_TOKEN
    if any(k in low for k in ("rate_limit", "rate limit", "429", "too many",
                               "quota", "quota_limit")):
        return RetryKind.RATE_LIMIT
    if any(k in low for k in ("http_422", "http 422", "http_400", "http 400",
                              "unprocessable")):
        return RetryKind.UPSTREAM_INVALID
    if any(k in low for k in ("overload", "overloaded", "server busy",
                              "server_busy", "try again", "timed out", "timeout",
                              "502", "bad gateway", "service unavailable",
                              "temporarily unavailable")):
        # Partial live emit: do not retry (would duplicate), but the
        # terminal ladder still surfaces it as 503. Signal that by returning
        # OVERLOAD anyway; the retry recover checks the guard.
        return RetryKind.OVERLOAD
    if P.is_ban_error_text(error) or "restricted" in low:
        return RetryKind.BAN
    return RetryKind.UPSTREAM


def _log_cookies(where: str, headers: dict) -> None:
    """Cookie-names-only diagnostic (never values/secrets)."""
    try:
        desc = P.describe_cookies(headers)
        logger.warning("qwen %s cookie_names=%s has_token=%s has_acw_tc=%s",
                       where, desc["names"], desc["has_token"], desc["has_acw_tc"])
    except Exception:
        pass


def _headers(app: FastAPI, token: str) -> dict:
    # getattr guard: embedded/test apps may have no lifespan bridge. Missing
    # bridge must not mask the real error with AttributeError (notably from
    # the delete-chat cleanup finally, which must never replace the exception).
    # Prefer app.state.bridge cookies directly: P.bridge_headers() reads the
    # process default bridge, which diverges when the app bridge differs
    # (tests, embedded, re-created lifespan). Fall back to the default bridge
    # for out-of-server callers, else plain base (which always merges token=).
    # Qwen-host acw_tc harvested via prime/fetch (despite discard_cookies)
    # is merged too so Cookie carries token= + acw_tc (Bearer kept as well).
    def _with_cached(waf_cookies: str, ua: str) -> dict:
        try:
            waf_cookies = QC._merged_waf_cookies(waf_cookies or "")
        except Exception:
            pass
        return P.base_headers(user_agent=ua, waf_cookies=waf_cookies,
                              bearer=token)
    bridge: ObscuraBridge | None = None
    if app is not None:
        try:
            bridge = getattr(app.state, "bridge", None)
        except Exception:
            bridge = None
    if bridge is not None:
        try:
            try:
                ua = bridge.state.user_agent
                cookies = bridge.cookie_header()
            except OSError as exc:
                # WinError5 on profile dir: graceful, names-only, keep token.
                try:
                    _log_cookies("headers_winerror5", {"Cookie": ""})
                except Exception:
                    pass
                try:
                    logger.warning("qwen headers WinError5 (%s), "
                                   "continuing with token only", exc)
                except Exception:
                    pass
                ua, cookies = "", ""
            return _with_cached(cookies, ua)
        except Exception:
            pass
    try:
        # Default-bridge path already merges token=; add cached acw_tc.
        h = P.bridge_headers(bearer=token)
        try:
            cached = QC.qwen_waf_cookie_header()
            if cached and "acw_tc=" not in str(h.get("Cookie") or "").lower():
                base = str(h.get("Cookie") or "").strip().rstrip(";").strip()
                h["Cookie"] = f"{base}; {cached}" if base else cached
                QC._ensure_token_cookie(h)
        except Exception:
            pass
        return h
    except Exception:
        pass
    try:
        return _with_cached("", "")
    except Exception:
        return P.base_headers(bearer=token)


def _ensure_token(app: FastAPI, item: PooledAccount) -> str:
    """Return the account token, or a fast 401 that callers can fail over on.

    A missing token is a provisioning state; paste ``localStorage.token`` from
    chat.qwen.ai (or the qwen_token/token cookies). The failover helper can
    move to another account that already has a token.
    """
    from ...accounts import token_help_for

    return _engine.ensure_token(
        item, f"account {item.cfg.identifier} has no token. "
              + token_help_for("qwen"))


def _app_settings(app: FastAPI) -> Settings:
    return _engine.app_settings(app)


def _ban_message(mute_until: float | None) -> str:
    """Human 403 text for a marked ban (Qwen envelope unverified).

    # TODO(unverified): real Qwen ban messages carry unknown wording; mirror
    # DeepSeek's UI-style message as a stand-in until a live payload lands.
    """
    base = "Due to violation of user policies, your account has been suspended"
    if mute_until is None:
        return f"{base} (expiry unknown). If you have any questions, please contact us."
    dt = datetime.datetime.fromtimestamp(float(mute_until))
    return (f"{base} until {dt.strftime('%B')} {dt.day} {dt.year} "
            f"{dt.strftime('%H:%M')}. If you have any questions, please contact us.")


def _ban_403_for_item(app: FastAPI, item: PooledAccount, ban_until: float | None,
                      error_text: str = "") -> HTTPException:
    """Mark + persist an account ban and build the 403 (delegates to engine)."""
    return _engine.ban_403_for_item(
        app, item, ban_until, error_text,
        ban_message_fn=(lambda u: _ban_message(u) if u else ""))


def _is_ban_http_exception(exc: BaseException) -> bool:
    """True when exc is the ban 403 raised by _run_completion_core."""
    return _engine.is_ban_http_exception(exc, P.is_ban_error_text)


def _switch_current_after_ban_failover(pool: AccountPool, settings: Settings,
                                       used: PooledAccount,
                                       failed: PooledAccount,
                                       reason: str = "unusable") -> None:
    """Move CURRENT to the working account after failover (engine-owned)."""
    _engine.switch_current_after_ban_failover(pool, settings, used, failed,
                                              reason=reason)


def _complete_with_failover_sync(app: FastAPI, pool: AccountPool, prep: dict,
                                 first_item: PooledAccount,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Run one completion, auto-failing over on ban or broken credentials.

    Qwen supplies its ``_run_completion_core`` and the ban-403 predicate via
    module globals (late-bound so monkeypatching keeps working); the engine
    owns the generic slot lifecycle.
    """
    def _run_core(account: PooledAccount) -> dict:
        return _run_completion_core(
            app, account, prep, on_content=on_content, on_thinking=on_thinking,
            cancel_event=cancel_event)

    def _is_ban(exc: BaseException) -> bool:
        return _is_ban_http_exception(exc)

    return _engine.complete_with_failover_sync(
        app, pool, prep, first_item, _run_core, _is_ban,
        cancel_event=cancel_event, on_content=on_content,
        on_thinking=on_thinking, allow_failover=allow_failover,
        get_settings=_app_settings)


def _refresh_waf_without_account_lock(app: FastAPI, item: PooledAccount,
                                       cancel_event: threading.Event | None = None) -> None:
    """Global WAF refresh without wasting the per-account slot.

    Aliyun anti-bot blocks mutating chat endpoints. DeepSeek-host warmup is
    skipped for qwen (never plants qwen cookies + WinError5 noise on Windows);
    instead best-effort solved-jar fetch to chat.qwen.ai via obscura (full
    acw_tc 64-char + x-ap/sca/cna/tfstk/isg), then curl prime GET models/
    (bare acw_tc fallback). All graceful, names-only, never raises. Uses the
    account token when available so retry headers carry token=.
    Never raises.
    """
    with _slot_releaser(item, cancel_event):
        try:
            bridge = None
            try:
                bridge = getattr(app.state, "bridge", None) if app is not None else None
            except Exception:
                bridge = None
            if bridge is None:
                try:
                    from ...core.obscura_bridge import get_default_bridge
                    bridge = get_default_bridge()
                except Exception:
                    bridge = None
            if bridge is not None:
                try:
                    # Qwen-host solved fetch (existing chat.qwen.ai/ only, no
                    # invented endpoints). Skips DeepSeek warmup entirely.
                    QC.prime_qwen_bridge(bridge)
                except OSError as exc:
                    try:
                        _log_cookies("refresh_qwen_bridge_winerror", {})
                    except Exception:
                        pass
                    if QC._is_winerror5(exc):
                        logger.warning("qwen bridge WinError5 "
                                       "(profile dir locked), priming qwen host")
                    else:
                        logger.warning("qwen bridge OSError (%s), "
                                       "priming qwen host", exc)
                except Exception as exc:
                    logger.debug("qwen bridge fetch ignored: %s", exc)
            # Qwen-host priming (existing endpoint only): GET models/ is NOT
            # WAF-gated and warms the qwen edge path on chat.qwen.ai.
            # Best-effort; uses the account token when available so
            # the retry headers already carry token= + Bearer.
            try:
                tok = ""
                try:
                    from ...accounts import extract_token
                    tok = extract_token(
                        getattr(getattr(item, "cfg", None), "token", "") or "")
                except Exception:
                    tok = ""
                try:
                    hdrs = _headers(app, tok)
                except OSError as exc:
                    if QC._is_winerror5(exc):
                        logger.warning("qwen prime headers WinError5, "
                                       "priming with token only")
                    hdrs = P.base_headers(bearer=tok)
                except Exception:
                    hdrs = P.base_headers(bearer=tok)
                QC.prime_waf(hdrs)
            except OSError as exc:
                if QC._is_winerror5(exc):
                    logger.warning("qwen prime WinError5, continuing")
                pass
            except Exception:
                pass
        except Exception as exc:
            logger.warning("WAF warmup failed: %s", exc)


def _delete_chat_bg(headers: dict, chat_id: str | None, enabled: bool = True) -> None:
    """Best-effort chat cleanup without blocking the response."""
    if not enabled or not chat_id:
        return

    def _run() -> None:
        try:
            QC.delete_chat(headers, chat_id)
        except Exception:
            pass

    try:
        _CLEANUP_EXECUTOR.submit(_run)
    except Exception:
        pass


def _terminal_invalid_token(app: FastAPI, item: PooledAccount, prep: dict,
                            token: str | None, error: str,
                            stream_ban_until: float | None) -> None:
    # No browser refresh for Qwen: a 401 means the pasted token is stale.
    raise HTTPException(
        status_code=401,
        detail=(f"token invalid for {item.cfg.identifier} ({error[:200]}). "
                "Paste a fresh localStorage.token from https://chat.qwen.ai "
                "(or the qwen_token/token cookies) and set-token the account. "
                "If the web UI shows suspension/violation, treat it as BANNED."))


def _terminal_ban(app: FastAPI, item: PooledAccount, prep: dict,
                  token: str | None, error: str,
                  stream_ban_until: float | None) -> None:
    ban_until: float | None = stream_ban_until
    raise _ban_403_for_item(app, item, ban_until, error)


def _terminal_rate_limit(app: FastAPI, item: PooledAccount, prep: dict,
                         token: str | None, error: str,
                         stream_ban_until: float | None) -> None:
    # Global throttle: do NOT poison a specific account with a cooldown.
    _engine.raise_rate_limit("Qwen", error)


def _terminal_overload(app: FastAPI, item: PooledAccount, prep: dict,
                       token: str | None, error: str,
                       stream_ban_until: float | None) -> None:
    _engine.raise_overload("Qwen", error)


def _terminal_upstream_invalid(app: FastAPI, item: PooledAccount, prep: dict,
                               token: str | None, error: str,
                               stream_ban_until: float | None) -> None:
    _engine.raise_upstream_invalid("Qwen", prep, error)


def _terminal_upstream(app: FastAPI, item: PooledAccount, prep: dict,
                       token: str | None, error: str,
                       stream_ban_until: float | None) -> None:
    _engine.raise_upstream("Qwen", error)


#: Terminal dispatch table: one RetryKind (from the shared classifier) per
#: HTTP outcome. WAF is a retryable kind that only reaches here after
#: exhausting the retry budget -> generic bad-gateway.
_TERMINAL_HANDLERS: dict[RetryKind, Callable[..., None]] = {
    RetryKind.INVALID_TOKEN: _terminal_invalid_token,
    RetryKind.BAN: _terminal_ban,
    RetryKind.RATE_LIMIT: _terminal_rate_limit,
    RetryKind.OVERLOAD: _terminal_overload,
    RetryKind.UPSTREAM_INVALID: _terminal_upstream_invalid,
    RetryKind.WAF: _terminal_upstream,
    RetryKind.UPSTREAM: _terminal_upstream,
}


def _raise_for_terminal_error(app: FastAPI, item: PooledAccount, prep: dict,
                              token: str | None, error: str,
                              stream_ban_until: float | None) -> None:
    """Map a terminal completion error event to its HTTP status. Always raises."""
    _engine.dispatch_terminal(classify_error, _TERMINAL_HANDLERS,
                              _terminal_upstream, app, item, prep, token,
                              error, stream_ban_until)


def _run_completion_core(
    app: FastAPI,
    item: PooledAccount,
    prep: dict,
    on_content=None,
    on_thinking=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Shared retry engine for buffered + live streaming.

    on_content/on_thinking are called incrementally (producer thread) for the
    live stream mode. Overload retries are skipped once partial deltas were
    already emitted (re-stream would duplicate text). WAF retries refresh the
    bridge and recreate the chat; INVALID_TOKEN retries recreate the chat once
    with the same token. Buffered callers pass no callbacks and get identical
    behaviour. Retry dispatch is table-driven through
    ``core.engine.drive_retries`` with ``classify_error`` as the single source
    of truth. cancel_event, when set, aborts at the next SSE event.
    """
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    auto_delete = bool(getattr(settings, "auto_delete_session", True))
    token = _ensure_token(app, item)
    chat_id: str | None = None
    headers: dict = {}
    payload: dict = {}

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    try:
        if _cancelled():
            raise _StreamCancelled()
        content = think = ""
        error: str | None = None
        usage_total = 0
        stream_ban_until: float | None = None
        _emit_tracker = _engine.LiveEmitTracker()
        _wrapped_content = _emit_tracker.wrap(on_content)
        _wrapped_thinking = _emit_tracker.wrap(on_thinking)

        def _do_attempt(h: dict, p: dict) -> None:
            nonlocal content, think, error, usage_total, stream_ban_until
            outcome = _attempt(
                h, p,
                on_content=(_wrapped_content if on_content is not None else None),
                on_thinking=(_wrapped_thinking if on_thinking is not None else None),
                cancel_event=cancel_event)
            content, think, error, usage_total = outcome
            try:
                cur_ban = getattr(outcome, "ban_until", None)
            except Exception:
                cur_ban = None
            if cur_ban is not None:
                stream_ban_until = cur_ban

        def _prepare_chat() -> bool:
            """Create a fresh chat + bind payload; False + error on failure."""
            nonlocal chat_id, headers, payload, error
            try:
                hdrs = _headers(app, token)
                try:
                    # Defensive: token= must ride Cookie (Bearer alone is
                    # WAF-gated). Names-only diagnostic on miss.
                    if token and "token" not in [
                            n.lower() for n in P.cookie_names(
                                str(hdrs.get("Cookie") or ""))]:
                        _log_cookies("prepare_chat_missing_token", hdrs)
                except Exception:
                    pass
                sid = QC.create_chat(hdrs, prep["model"])
            except _StreamCancelled:
                raise
            except Exception as exc:
                error = _error_text("", str(exc))
                try:
                    if classify_error(error) == RetryKind.WAF:
                        _log_cookies("prepare_chat_waf",
                                     hdrs if "hdrs" in dir() else headers)
                except Exception:
                    pass
                return False
            chat_id = sid
            headers = hdrs
            payload = P.completion_payload(
                chat_id, prep["prompt"], prep["model"],
                prep["thinking"], prep["search"])
            return True

        def _recover_waf() -> None:
            """Refresh WAF cookies (slot released during the global wait)."""
            logger.info("possible WAF block, refreshing cookies (retries left %s)",
                        max_retries)
            try:
                _log_cookies("recover_waf_before", headers)
            except Exception:
                pass
            _refresh_waf_without_account_lock(app, item, cancel_event=cancel_event)
            if _cancelled():
                return
            if _prepare_chat():
                _do_attempt(headers, payload)

        def _recover_invalid_token() -> None:
            """Mid-stream 401: recreate the chat once with the same token."""
            logger.info("token invalid, recreating chat (retries left %s)", max_retries)
            if _prepare_chat():
                _do_attempt(headers, payload)

        def _recover_overload() -> None:
            """Retry transient overload with fresh headers (no stale reuse)."""
            nonlocal headers
            logger.info("upstream overloaded (%s), retrying (retries left %s)",
                        (error or "")[:80], max_retries)
            try:
                headers = _headers(app, token)
            except Exception:
                pass
            if chat_id and payload:
                _do_attempt(headers, payload)
            else:
                if _prepare_chat():
                    _do_attempt(headers, payload)

        if not _prepare_chat():
            # Chat create failed (likely WAF/401); error set, drive_retries
            # classifies it and may recover.
            pass
        else:
            _do_attempt(headers, payload)
        drive_retries(
            max_retries,
            _cancelled,
            lambda: error,
            (
                ("waf", lambda e: classify_error(e) == RetryKind.WAF,
                 _recover_waf),
                ("invalid_token",
                 lambda e: classify_error(e) == RetryKind.INVALID_TOKEN,
                 _recover_invalid_token),
                ("overload",
                 lambda e: classify_error(e) == RetryKind.OVERLOAD
                 and not _emit_tracker.emitted,
                 _recover_overload),
            ),
        )
        if _cancelled():
            raise _StreamCancelled()
        if error:
            _raise_for_terminal_error(app, item, prep, token, error,
                                      stream_ban_until)
        if not content and not think:
            # Upstream closed without content/thinking/error (e.g. unparsed
            # toast or a silent WAF swallow). Never return silent empty 200:
            # it surfaces as "tool stopped working".
            raise _engine.empty_response_error("Qwen")
        return {"content": content, "thinking": think, "usage_total": usage_total}
    finally:
        try:
            # Guard chat_id first so no header build happens when there is
            # nothing to delete; _headers itself is bridge-safe (getattr).
            if auto_delete and chat_id:
                _delete_chat_bg(_headers(app, token), chat_id, enabled=True)
        except Exception:
            pass


def _attempt(headers: dict, payload: dict, on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             ) -> tuple[str, str, str | None, int]:
    """Drain one SSE stream (engine-owned); keeps this module's patch seam."""
    return _engine.drain_attempt(QC.stream_completion, headers, payload,
                                 on_content=on_content, on_thinking=on_thinking,
                                 cancel_event=cancel_event)


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end.

    Producer (account-holding thread) is Qwen-specific only in the failover
    entrypoint; the SSE consumer (chunk framing, tool tail, timeout/errors)
    is engine-owned via ``drain_stream_queue``.
    """
    loop = asyncio.get_running_loop()
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "live")).lower() == "live"
    cid, created, out, cancel_event, consumer_timeout = _engine.new_stream_context(
        prep, QC.DEFAULT_TIMEOUT, _CONSUMER_GRACE_SECONDS)
    model = prep["model"]

    def _emit(item: tuple) -> None:
        _engine.emit_to_loop(loop, out, item)

    def _put(kind: str, text: str) -> None:
        if text:
            _emit((kind, text))

    def _produce(*, live: bool) -> None:
        # Single publisher: buffered and live differ only in callbacks.
        # Thinking always streams live (never tool JSON); content streams
        # live only when no tools requested, else buffered so a tool_calls
        # reply is not also emitted as user-visible text.
        try:
            if live:
                result = _complete_with_failover_sync(
                    app, pool, prep, item,
                    on_content=(None if prep.get("tools") else lambda t: _put("content", t)),
                    on_thinking=lambda t: _put("thinking", t),
                    cancel_event=cancel_event,
                )
            else:
                # Ban failover lives in the helper (owns item's lock): a
                # banned account retries on the next unbanned one inside the
                # same request. Ban errors carry no content prefix, so
                # nothing partial leaks.
                result = _complete_with_failover_sync(
                    app, pool, prep, item, cancel_event=cancel_event)
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))

    def produce_buffered():
        _produce(live=False)

    def produce_live():
        _produce(live=True)

    # Bounded producer: one account-holding thread per stream, capped by the
    # shared executor so a parallel-subagent burst queues instead of piling
    # unbounded raw threads.
    try:
        _STREAM_EXECUTOR.submit(produce_live if live else produce_buffered)
    except RuntimeError:
        # Executor shut down (interpreter exit): fall back to a raw thread.
        threading.Thread(target=produce_live if live else produce_buffered, daemon=True).start()
    try:
        async for frame in _engine.drain_stream_queue(
                out, prep, live, cid, created, model, consumer_timeout,
                _parse_tool_calls_local, looks_truncated_tool_attempt):
            yield frame
    finally:
        # Generator closed (client disconnect, cancellation, completion): tell
        # the producer to stop at the next event so the account lock frees.
        cancel_event.set()
