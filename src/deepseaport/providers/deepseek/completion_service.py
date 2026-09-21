"""DeepSeek completion engine: failover, retries, PoW/session, token refresh.

DeepSeek-specific adapter over the provider-neutral mechanics in
``deepseaport.core.engine``. The engine owns the generic retry driver, the
cross-account failover / account-slot lifecycle, and the stream producer
bridge; this module owns everything DeepSeek: session create/delete, PoW,
SSE parsing, ban probes, and the Obscura token refresh. The canonical
monkeypatch surface is this module (``providers.deepseek.completion_service``);
``deepseaport.server`` re-exports the same objects as thin compat aliases.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from . import client as DS
from . import pow as PoW
from . import protocol as P
from ...accounts import AccountPool, PooledAccount
from ...config import Settings, load_settings
from ...core.engine import ProducerBridge, drive_retries, run_with_failover
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
    _CHALLENGE_EXECUTOR,
    _CLEANUP_EXECUTOR,
    _CONSUMER_GRACE_SECONDS,
    _STREAM_EXECUTOR,
    _StreamCancelled,
    _await_cancelable,
)

logger = logging.getLogger("deepseaport.server")


# Invalid/expired token signal (biz 40003 "Authorization Failed (invalid
# token)"). On this the request auto-refreshes via the Obscura browser login
# and retries once; only a second failure surfaces the error.
# Concurrent 40003s on the same account share one browser refresh.
_TOKEN_REFRESH_LOCK = threading.Lock()
_TOKEN_REFRESH_INFLIGHT: dict[str, threading.Event] = {}
# Login-page ban verdict from the leader refresh, shared with followers that
# waited on the same browser run (they never saw the page themselves).
_TOKEN_REFRESH_BAN_HINT: dict[str, tuple[bool, str]] = {}
# Browser-login backoff now lives on AccountHealth (accounts.py), owned by the
# PooledAccount, so the REST unblock/set-token paths clear it through the same
# pool instead of importing a DeepSeek-private dict. The singleflight below
# stays process-global: it coordinates concurrent 40003s only.


def _headers(app: FastAPI, token: str) -> dict:
    # getattr guard: embedded/test apps may have no lifespan bridge. Missing
    # bridge must not mask the real error with AttributeError (notably from
    # the session-cleanup finally, which must never replace the exception).
    bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
    if bridge is None:
        return P.base_headers(bearer=token)
    return P.base_headers(user_agent=bridge.state.user_agent,
                          waf_cookies=bridge.cookie_header(), bearer=token)


def _ensure_token(app: FastAPI, item: PooledAccount) -> str:
    """Return the account token, or a fast 401 that callers can fail over on.

    A missing token is a provisioning state, not something to solve by
    spawning a browser inside the first API request.  The explicit
    ``deepseaport login``/TUI flow populates it; the failover helper can move
    to another account that already has a token.  Expired tokens still take
    the one-time Obscura refresh path in _run_completion_core.
    """
    from ...accounts import TOKEN_HELP, extract_token

    if item.cfg.token:
        token = extract_token(item.cfg.token)
        if token != item.cfg.token:
            item.cfg.token = token
        return token
    raise HTTPException(
        status_code=401,
        detail=f"account {item.cfg.identifier} has no token. " + TOKEN_HELP)


def _is_invalid_token_error(error: str | None) -> bool:
    """True when an attempt error is the expired/invalid token (biz 40003)."""
    low = (error or "").lower()
    return "40003" in low or (
        "authorization failed" in low and "token" in low)


def _is_transient_overload(error: str | None) -> bool:
    """True for retryable upstream overload (never a 500).

    e.g. "generation_timeout: Server busy, please try again later."
    Global throttle, not account-specific: must surface as 503 (retryable),
    never RuntimeError (unhandled 500 "Exception in ASGI application").
    """
    low = (error or "").lower()
    return any(k in low for k in (
        "generation_timeout", "server busy", "server_busy",
        "overloaded", "overload", "service unavailable",
        "temporarily unavailable", "try again", "timed out", "timeout",
        "bad gateway", "gateway timeout", "upstream timed out",
    ))


class RetryKind(enum.Enum):
    """Single classifier for attempt errors (retry + terminal share it).

    Retryable: POW, INVALID_SESSION, INVALID_TOKEN, WAF, OVERLOAD.
    Terminal: BAN, RATE_LIMIT, UPSTREAM_INVALID, UPSTREAM (generic 502);
    INVALID_TOKEN is retryable but its terminal handler (401, after a ban
    probe) is also table-driven. One ``classify_error`` maps text -> kind;
    retry and terminal tables consume the same value so the string matching
    lives in exactly one place.
    """

    POW = "pow"
    INVALID_SESSION = "invalid_session"
    INVALID_TOKEN = "invalid_token"
    WAF = "waf"
    OVERLOAD = "overload"
    BAN = "ban"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_INVALID = "upstream_invalid"
    UPSTREAM = "upstream"


def _is_waf_error(error: str | None) -> bool:
    """True for 403/WAF error events (cookie refresh may help)."""
    e = error or ""
    return "403" in e or "WAF" in e


def _is_invalid_session_error(error: str | None) -> bool:
    """True for stale session ids (recreate + fresh PoW + retry)."""
    return bool(error) and error.startswith("INVALID_SESSION_ID")


def _is_pow_error(error: str | None) -> bool:
    """True for PoW rejections (fresh solve + retry)."""
    return bool(error) and "pow" in error.lower()


def _is_rate_limit_error(error: str | None) -> bool:
    """True for global throttles (429, never account-marked)."""
    low = (error or "").lower()
    return any(k in low for k in ("rate_limit", "too frequent", "too_frequent", "429"))


def _is_upstream_invalid_error(error: str | None) -> bool:
    """True for upstream payload rejections (422/400 family)."""
    low = (error or "").lower()
    return "http_422" in low or "unprocessable" in low or "http_400" in low


def classify_error(error: str | None) -> RetryKind | None:
    """Map an attempt error string to its retry/terminal kind.

    Order matches the historical retry loops (pow -> session -> token ->
    waf -> overload) then the terminal ladder (ban -> rate-limit ->
    overload -> upstream-invalid -> upstream). Returns None only when there
    is no error text to classify.

    The retry table additionally guards OVERLOAD with the "partial live emit"
    flag (a re-stream would duplicate text); that is a retry-policy decision,
    not a classification, so it stays in the table rather than here.
    """
    if not error:
        return None
    low = error.lower()
    if _is_pow_error(error):
        return RetryKind.POW
    if _is_invalid_session_error(error):
        return RetryKind.INVALID_SESSION
    if _is_invalid_token_error(error):
        return RetryKind.INVALID_TOKEN
    if _is_waf_error(error):
        return RetryKind.WAF
    if _is_transient_overload(error):
        # Partial live emit: do not retry (would duplicate), but the
        # terminal ladder still surfaces it as 503. Signal that by
        # returning OVERLOAD anyway; the retry recover checks the guard.
        return RetryKind.OVERLOAD
    if P.is_ban_error_text(error) or any(
            k in low for k in ("restricted", "unauthorized", "401")):
        return RetryKind.BAN
    if _is_rate_limit_error(error):
        return RetryKind.RATE_LIMIT
    if _is_upstream_invalid_error(error):
        return RetryKind.UPSTREAM_INVALID
    return RetryKind.UPSTREAM


def _probe_ban(app: FastAPI, token: str, timeout: int = 5,
               provider: str = "deepseek") -> tuple[bool, float | None]:
    """Best-effort ban probe for one token. Never raises.

    Routes through ``Provider.probe_ban`` so the pool/engine share one hook;
    DeepSeek (default) keeps the users/current probe over app-bridge headers.
    Used to distinguish a muted/suspended account (must become 403 +
    cooldown + failover) from a merely stale token (401 + browser refresh).
    Banned accounts can surface as 40003 on session/create, so every 40003
    path probes here before concluding "invalid token".
    """
    try:
        from ...core import registry as _reg

        pid = (provider or "deepseek").strip().lower() or "deepseek"
        prov = _reg.get(pid) or _reg.get("deepseek")
        if prov is None:
            return DS.check_ban(_headers(app, (token or "").strip()),
                                timeout=timeout)
        if pid == "deepseek":
            headers = _headers(app, (token or "").strip())
        else:
            headers = {"Authorization": f"Bearer {(token or '').strip()}"}
        return prov.probe_ban(headers, timeout=timeout)
    except Exception:
        return False, None


def _ban_403_for_item(app: FastAPI, item: PooledAccount, ban_until: float | None,
                      error_text: str = "") -> HTTPException:
    """Mark + persist an account ban and build the 403.

    Persisting ``cfg.banned``/``cfg.banned_until`` is what makes the account
    show up as ``(BANNED: 16 September)`` in the CLI/TUI even after the server
    is restarted or the in-memory pool is rebuilt.  ``ban_until=None`` still
    marks the account (unknown expiry -> ``(BANNED)``) instead of losing the
    ban verdict.
    """
    try:
        AccountPool.mark_banned(item, until=ban_until)
        try:
            _app_settings(app).save()
        except Exception:
            pass
    except Exception:
        pass
    try:
        base = P.ban_message(ban_until) if ban_until else ""
        if not base:
            detail = (error_text or "account banned")[:300]
            if "banned" not in detail.lower() and "muted" not in detail.lower():
                detail = f"Account {item.cfg.identifier} banned. {detail}"
        else:
            detail = f"Account {item.cfg.identifier} banned. {base}"
    except Exception:
        detail = (error_text or f"Account {item.cfg.identifier} banned")[:300]
    return HTTPException(status_code=403, detail=detail)


def _refresh_token_via_obscura(app: FastAPI, item: PooledAccount,
                               cancel_event: threading.Event | None = None,
                               ) -> tuple[str | None, bool, str]:
    """Re-login through the Obscura browser and persist a fresh userToken.

    Returns (token_or_None, login_banned, ban_evidence). login_banned=True is
    ground truth from the login page suspension banner — the caller must treat
    the account as banned even when the API only ever said 40003.
    Singleflight per account: concurrent 40003s share one browser run.
    Never raises (except _StreamCancelled when the client vanished while waiting).
    """
    cfg = item.cfg
    if not (cfg.email or cfg.mobile) or not cfg.password:
        return None, False, ""
    settings = _app_settings(app)
    key = cfg.identifier or cfg.email or cfg.mobile

    with _TOKEN_REFRESH_LOCK:
        inflight = _TOKEN_REFRESH_INFLIGHT.get(key)
        if inflight is not None:
            leader = False
        else:
            inflight = threading.Event()
            _TOKEN_REFRESH_INFLIGHT[key] = inflight
            leader = True
    if not leader:
        # Chunked wait so a disconnected follower stops holding the account
        # slot instead of parking the full 180s. Leader browser run itself
        # is not killed; follower just aborts its wait promptly.
        deadline = time.monotonic() + 180
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise _StreamCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                if inflight.wait(timeout=min(0.2, remaining)):
                    break
            except Exception:
                pass
        if cancel_event is not None and cancel_event.is_set():
            raise _StreamCancelled()
        fresh = (item.cfg.token or "").strip()
        try:
            with _TOKEN_REFRESH_LOCK:
                banned, detail = _TOKEN_REFRESH_BAN_HINT.get(key, (False, ""))
        except Exception:
            banned, detail = False, ""
        return (fresh or None), banned, detail
    banned = False
    detail = ""
    # Backoff: a recent no-token login (captcha/credential) means the login
    # page is challenging this profile right now. Spawning the browser on
    # every 40003 flags it further — skip fast with the previous evidence.
    skip_reason = item.login_backoff_skip()
    if skip_reason:
        msg = (f"browser login skipped for {key} ({skip_reason}; "
               f"manual set-token bypasses this)")
        logger.info(msg)
        return None, False, msg
    try:
        from .auth import obscura_login_token as _login_token
        from .auth import unpack_login_result as _unpack_login
        try:
            raw = _login_token(cfg.email or cfg.mobile,
                               cfg.password, settings)
            new_token, banned, detail = _unpack_login(raw)
            try:
                from ...accounts import extract_token as _extract_token
                new_token = _extract_token(new_token) or None
            except Exception:
                pass
        except Exception as exc:
            logger.warning("token refresh via Obscura failed for %s: %s", key, exc)
            new_token, banned, detail = None, False, ""
        try:
            with _TOKEN_REFRESH_LOCK:
                _TOKEN_REFRESH_BAN_HINT[key] = (bool(banned), str(detail or ""))
        except Exception:
            pass
        if banned:
            logger.warning("Obscura login page shows account %s BANNED: %s",
                           key, (detail or "")[:200])
            item.clear_login_backoff()
        if new_token:
            from ...accounts import set_account_token
            item.cfg.token = new_token
            try:
                set_account_token(settings, key, new_token)
                settings.save()
            except Exception as exc:
                logger.warning("token refresh save failed for %s: %s", key, exc)
            if not banned:
                logger.info("token refreshed via Obscura for %s", key)
                item.clear_login_backoff()
            return new_token, bool(banned), str(detail or "")
        if not banned:
            # No token, no ban page: back off the browser so repeated 40003s
            # don't hammer the challenge endpoint. Manual set-token clears it.
            low = (detail or "").lower()
            if "captcha" in low:
                item.set_login_backoff("captcha-challenge", 180)
            elif "credential" in low or "incorrect" in low or "wrong password" in low:
                item.set_login_backoff("credential-error", 600)
            elif "form interaction failed" in low:
                item.set_login_backoff("login-form-changed", 600)
            else:
                item.set_login_backoff("no-token", 120)
        return None, bool(banned), str(detail or "")
    finally:
        with _TOKEN_REFRESH_LOCK:
            _TOKEN_REFRESH_INFLIGHT.pop(key, None)
            try:
                inflight.set()
            except Exception:
                pass


def _solve(app: FastAPI, token: str) -> dict[str, str]:
    """Fresh PoW header merged over base headers."""
    headers = _headers(app, token)
    return {**headers, "X-DS-PoW-Response": DS.solve_pow(headers)}


def _session_and_challenge(headers: dict, parallel: bool = True,
                            cancel_event: threading.Event | None = None,
                            ) -> tuple[str, dict]:
    """Create session + fetch PoW challenge concurrently (same headers).

    Sequential code paid 2 RTTs; these calls are independent. Session errors
    take priority to match old sequential ordering. When parallel=False the
    two calls run sequentially (session first, same error priority).

    cancel_event aborts the wait between SSE-independent setup calls so a
    disconnected client frees the account slot instead of running the full
    session→PoW→stream while the next request queues to 90s/429.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise _StreamCancelled()
    if not parallel:
        session_id = DS.create_session(headers)
        if cancel_event is not None and cancel_event.is_set():
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise _StreamCancelled()
        try:
            challenge = DS.fetch_pow(headers)
        except Exception:
            # Session created but challenge failed: don't leak it.
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise
        if cancel_event is not None and cancel_event.is_set():
            try:
                DS.delete_session(headers, session_id)
            except Exception:
                pass
            raise _StreamCancelled()
        return session_id, challenge
    fut_session = _CHALLENGE_EXECUTOR.submit(DS.create_session, headers)
    fut_challenge = _CHALLENGE_EXECUTOR.submit(DS.fetch_pow, headers)
    try:
        session_id = _await_cancelable(fut_session, cancel_event)
    except _StreamCancelled:
        try:
            fut_challenge.cancel()
        except Exception:
            pass
        raise
    except Exception:
        try:
            _await_cancelable(fut_challenge, None)
        except Exception:
            pass
        raise
    try:
        challenge = _await_cancelable(fut_challenge, cancel_event)
    except _StreamCancelled:
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass
        raise
    except Exception:
        # Session was created but challenge fetch failed: don't leak it.
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass
        raise
    return session_id, challenge


def _header_from_challenge(challenge: dict) -> str:
    """Solve an already-fetched PoW challenge (no refetch)."""
    return DS.solve_challenge(challenge)


def _delete_session_bg(headers: dict, session_id: str | None, enabled: bool = True) -> None:
    """Best-effort session cleanup without blocking the response."""
    if not enabled or not session_id:
        return

    def _run() -> None:
        try:
            DS.delete_session(headers, session_id)
        except Exception:
            pass

    try:
        _CLEANUP_EXECUTOR.submit(_run)
    except Exception:
        pass


def _app_settings(app: FastAPI) -> Settings:
    settings = getattr(app.state, "settings", None)
    if settings is None:
        settings = load_settings()
        app.state.settings = settings
    return settings


def _refresh_waf_without_account_lock(app: FastAPI, item: PooledAccount,
                                        cancel_event: threading.Event | None = None) -> None:
    """Global WAF refresh without wasting the per-account slot.

    The caller holds item.lock (one in-flight per account). A WAF refresh is
    global (shared profile/cookies) and singleflight-deduped in the bridge,
    so release the account slot while waiting for it, then reacquire before
    the retry attempt. Safe when the lock isn't held (unit tests calling
    _run_completion_core with a fresh account): release fails -> just warm.
    Reacquire polls so a disconnected client aborts instead of blocking
    indefinitely under contention (raises _StreamCancelled).
    """
    released = False
    try:
        item.lock.release()
        released = True
    except RuntimeError:
        released = False
    except Exception:
        released = False
    try:
        try:
            bridge = getattr(app.state, "bridge", None)
            if bridge is not None:
                bridge.warmup()
        except Exception as exc:
            logger.warning("WAF warmup failed: %s", exc)
    finally:
        if released:
            # Cancelable reacquire: unbounded acquire() would pin an orphaned
            # producer indefinitely when another request holds the slot.
            # AccountPool.release() swallows RuntimeError, so leaving the
            # slot released on cancel is safe (outer release becomes no-op).
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise _StreamCancelled()
                try:
                    if item.lock.acquire(timeout=0.5):
                        break
                except Exception:
                    break


def _is_ban_http_exception(exc: BaseException) -> bool:
    """True when exc is the ban 403 raised by _run_completion_core."""
    return (isinstance(exc, HTTPException) and exc.status_code == 403
            and P.is_ban_error_text(str(getattr(exc, "detail", "") or "")))


def _switch_current_after_ban_failover(pool: AccountPool, settings: Settings,
                                       used: PooledAccount,
                                       failed: PooledAccount,
                                       reason: str = "unusable") -> None:
    """Move CURRENT to the working account after failover.

    Only fires when CURRENT pointed at the failed account: the request
    already proved it unusable (ban/auth cooldown), so future requests should
    prefer the healthy one first. Best-effort, never raises.
    """
    try:
        from ...accounts import _matches as _m
        cur = pool.current
        if not cur or not _m(failed.cfg, cur):
            return  # auto-failover mode or CURRENT wasn't the failed one
        if _m(used.cfg, cur):
            return  # same account (shouldn't happen after mark_bad)
        if pool.set_current(used.cfg.identifier):
            try:
                settings.active_account = pool.current
                settings.save()
            except Exception:
                pass
            logger.warning("account %s %s, CURRENT auto-switched -> %s",
                           failed.cfg.identifier, reason, used.cfg.identifier)
    except Exception:
        pass


def _complete_with_failover_sync(app: FastAPI, pool: AccountPool, prep: dict,
                                 first_item: PooledAccount,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Run one completion, auto-failing over on ban or broken credentials.

    DeepSeek supplies the retryability predicate (ban 403 / auth 401), the
    cooling work, and the account after a success; ``core.engine`` owns the
    generic slot lifecycle: release exactly once per attempt, acquire the next
    account before retrying, re-raise when no healthy account remains.
    allow_failover=False (settings.use_multiple_accounts=False) surfaces the
    first error directly, never touching another account.
    None (default) reads the flag from app settings.
    """
    if allow_failover is None:
        try:
            allow_failover = bool(getattr(_app_settings(app), "use_multiple_accounts", True))
        except Exception:
            allow_failover = True

    first_exc: HTTPException | None = None
    failed_item: PooledAccount | None = None
    used_item: PooledAccount | None = None

    def _run_once(account: PooledAccount) -> dict:
        nonlocal used_item
        result = _run_completion_core(
            app, account, prep, on_content=on_content, on_thinking=on_thinking,
            cancel_event=cancel_event)
        used_item = account
        return result

    def _is_retryable(exc: BaseException) -> bool:
        # Bans and 401 auth failures are account-specific; try a healthy
        # account before surfacing the error. Other HTTP failures (400, 429,
        # WAF, ...) and non-HTTP errors belong to the caller/current request.
        return (isinstance(exc, HTTPException)
                and (_is_ban_http_exception(exc) or exc.status_code == 401))

    def _on_before_retry(exc, account: PooledAccount, tries: int, max_tries: int) -> None:
        nonlocal first_exc, failed_item
        if first_exc is None:
            first_exc, failed_item = exc, account
        if isinstance(exc, HTTPException) and exc.status_code == 401:
            AccountPool.mark_bad(account, seconds=INVALID_TOKEN_COOLDOWN_SECONDS)
            logger.info("account %s has no usable token, failing over "
                        "(%d/%d)", account.cfg.identifier, tries + 1, max_tries)
        else:
            logger.info("account %s banned, failing over to next account "
                        "(%d/%d)", account.cfg.identifier, tries + 1, max_tries)

    result = run_with_failover(
        pool,
        first_item,
        _run_once,
        _is_retryable,
        on_before_retry=_on_before_retry,
        acquire_timeout=FAILOVER_ACQUIRE_TIMEOUT,
        is_cancelled=lambda: cancel_event is not None and cancel_event.is_set(),
        allow_failover=allow_failover,
        release=AccountPool.release,
        acquire=lambda timeout: pool.acquire(timeout=timeout),
    )
    if failed_item is not None and used_item is not None:
        reason = "banned" if _is_ban_http_exception(first_exc) else "has no usable token"
        _switch_current_after_ban_failover(
            pool, _app_settings(app), used_item, failed_item, reason=reason)
    return result


def _terminal_invalid_token(app: FastAPI, item: PooledAccount, prep: dict,
                            token: str | None, error: str, login_note: str,
                            stream_ban_until: float | None) -> None:
    """Terminal 40003: already refreshed once above; a muted ban can present
    as 40003, so probe users/current once before concluding "invalid" —
    otherwise bans surface as 401 and the account is never marked cooling."""
    _banned, _ban_until = _probe_ban(app, token)
    if _banned:
        raise _ban_403_for_item(app, item, _ban_until, error)
    _note = f" Browser login note: {login_note[:300]}." if login_note else ""
    raise HTTPException(
        status_code=401,
        detail=f"Authorization Failed: token invalid for "
               f"{item.cfg.identifier} even after browser refresh "
               f"({error[:200]}).{_note} Re-login the account. If the web "
               f"UI shows suspension/violation, treat it as BANNED.")


def _terminal_ban(app: FastAPI, item: PooledAccount, prep: dict,
                  token: str | None, error: str, login_note: str,
                  stream_ban_until: float | None) -> None:
    """Banned accounts stay unusable until mute_until: cool down for the
    remaining suspension. The completion ban envelope already carried
    mute_until (stream_ban_until) — use it directly and skip the extra
    users/current HTTP; fall back to check_ban only when the stream had no ts."""
    ban_until: float | None = stream_ban_until
    if ban_until is None:
        try:
            _, ban_until = _probe_ban(
                app, token, timeout=5,
                provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
        except Exception:
            ban_until = None
    raise _ban_403_for_item(app, item, ban_until, error)


def _terminal_rate_limit(app: FastAPI, item: PooledAccount, prep: dict,
                         token: str | None, error: str, login_note: str,
                         stream_ban_until: float | None) -> None:
    # Global throttle: do NOT poison a specific account with a cooldown.
    raise HTTPException(status_code=429,
                        detail=f"DeepSeek rate limited: {error[:200]}. "
                               "Wait 2-20 min, space requests, avoid parallel loops.")


def _terminal_overload(app: FastAPI, item: PooledAccount, prep: dict,
                       token: str | None, error: str, login_note: str,
                       stream_ban_until: float | None) -> None:
    raise HTTPException(status_code=503,
                        detail=f"DeepSeek server busy: {error[:200]}. "
                               "Retry shortly (transient overload).")


def _terminal_upstream_invalid(app: FastAPI, item: PooledAccount, prep: dict,
                               token: str | None, error: str, login_note: str,
                               stream_ban_until: float | None) -> None:
    logger.warning("upstream rejected completion payload (%s); "
                   "prompt_len=%d model=%s",
                   error[:500], len(prep.get("prompt", "")),
                   prep.get("model"))
    raise HTTPException(
        status_code=502,
        detail=f"DeepSeek rejected the request as invalid ({error[:800]}). "
               "The prompt/payload failed upstream validation, not a ban or "
               "rate limit. Retry with a non-empty prompt; if it persists, "
               "the upstream API schema likely changed.")


def _terminal_upstream(app: FastAPI, item: PooledAccount, prep: dict,
                       token: str | None, error: str, login_note: str,
                       stream_ban_until: float | None) -> None:
    logger.warning("upstream completion error: %s", error[:500])
    raise HTTPException(
        status_code=502,
        detail=f"DeepSeek upstream error: {error[:800]}")


#: Terminal dispatch table: one RetryKind (from the shared classifier) per
#: HTTP outcome. POW/INVALID_SESSION/WAF are retryable kinds that only reach
#: here after exhausting the retry budget -> generic bad-gateway, exactly as
#: the historical fallthrough did.
_TERMINAL_HANDLERS: dict[RetryKind, Callable[..., None]] = {
    RetryKind.INVALID_TOKEN: _terminal_invalid_token,
    RetryKind.BAN: _terminal_ban,
    RetryKind.RATE_LIMIT: _terminal_rate_limit,
    RetryKind.OVERLOAD: _terminal_overload,
    RetryKind.UPSTREAM_INVALID: _terminal_upstream_invalid,
    RetryKind.POW: _terminal_upstream,
    RetryKind.INVALID_SESSION: _terminal_upstream,
    RetryKind.WAF: _terminal_upstream,
    RetryKind.UPSTREAM: _terminal_upstream,
}


def _raise_for_terminal_error(app: FastAPI, item: PooledAccount, prep: dict,
                              token: str | None, error: str,
                              login_note: str,
                              stream_ban_until: float | None) -> None:
    """Map a terminal completion error event to its HTTP status. Always raises.

    Single dispatch: ``classify_error`` (the same classifier the retry table
    uses) produces one RetryKind, then the terminal handler for that kind
    raises. No string matching lives here; ban-vs-rate-limit and
    invalid-token-before-ban ordering are encoded once, in ``classify_error``.
    """
    kind = classify_error(error)
    _TERMINAL_HANDLERS.get(kind, _terminal_upstream)(
        app, item, prep, token, error, login_note, stream_ban_until)


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
    already emitted (re-stream would duplicate text); overload retries use a
    fresh PoW header. PoW/session/WAF retries normally carry no content
    prefix, so the residual interleaving risk is minimal.
    Buffered callers pass no callbacks and get identical behaviour to before.

    cancel_event, when set, aborts the attempt at the next SSE event so a
    disconnected client does not keep the account lock for the whole reply.

    Retry dispatch is table-driven through ``core.engine.drive_retries``:
    ``classify_error`` maps the current error text to one ``RetryKind`` and
    the ordered handler table below recovers once per kind per pass. The
    historical five sequential string loops and the string-matched terminal
    ladder are gone; the classifier is the single source of truth.
    """
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    parallel = bool(getattr(settings, "parallel_challenge_fetch", True))
    auto_delete = bool(getattr(settings, "auto_delete_session", True))
    token = _ensure_token(app, item)
    session_id: str | None = None
    token_refreshed = False
    # Diagnostic from the browser login (no-token reason, last URL/body) so
    # the terminal 401 tells exactly what the login saw. Never marks banned.
    login_note = ""

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _refresh_token_once() -> bool:
        """Refresh the account token via Obscura, once per request.

        Returns True when a new token replaced the stale one. Shared by the
        session-create path (40003 raised) and the stream path (40003 event).
        Raises 403 immediately when the login page itself showed a suspension
        banner (deterministic ban, even if the API only ever said 40003).
        """
        nonlocal token, token_refreshed, login_note
        if token_refreshed:
            return False
        logger.info("token invalid for %s, refreshing via Obscura browser",
                    item.cfg.identifier)
        old_token = (token or "").strip()
        new_token, login_banned, login_detail = _refresh_token_via_obscura(
            app, item, cancel_event=cancel_event)
        if _cancelled():
            raise _StreamCancelled()
        if login_banned:
            # Ground truth from the browser: the login page showed
            # suspend/banned/violation. The API may only say 40003 for this
            # account, so the users/current probe cannot confirm it — trust
            # the page, mark + 403 for failover. Probe for mute_until first;
            # fall back to parsing the human date printed on the page.
            _, ban_until = _probe_ban(
                app, new_token or old_token,
                provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
            if not ban_until:
                try:
                    from .auth import parse_ban_until
                    ban_until = parse_ban_until(login_detail)
                except Exception:
                    ban_until = None
            raise _ban_403_for_item(
                app, item, ban_until,
                f"Obscura login page shows account {item.cfg.identifier} "
                f"suspended: {(login_detail or '')[:200]}")
        if login_detail and not new_token:
            login_note = str(login_detail)[:500]
            logger.warning("Obscura login for %s produced no usable token: %s",
                           item.cfg.identifier, login_note[:300])
        if not new_token:
            return False
        if old_token and new_token.strip() == old_token:
            # Login demonstrably succeeded (left sign_in, else the login
            # helper would have ignored this value as stale), so the same
            # value is a stable token re-issue, not a failure. Retry with it;
            # a renewed 40003 then means account status, handled by the probe.
            logger.info("token refresh for %s returned same value after "
                        "verified login (stable token); retrying", item.cfg.identifier)
        token = new_token
        token_refreshed = True
        return True

    def _open_session() -> tuple[dict, str, dict]:
        """Create session + PoW; auto-refresh the token once on 40003.

        The 40003 surfaces as a RuntimeError from create_session. On the first
        hit the token is refreshed via Obscura and session creation retried;
        a second failure propagates to the caller untouched — except bans:
        a ban payload (biz 5 muted) or a users/current probe showing muted
        becomes 403 + cooldown (so failover + accounts list mark it) instead
        of a RuntimeError/401 that never marks the account. Banned accounts
        can present as 40003 on session/create, hence the probe on both
        attempts before concluding "invalid token".
        """
        nonlocal session_id
        last_exc: Exception | None = None
        for attempt in range(2):
            if _cancelled():
                raise _StreamCancelled()
            hdrs = _headers(app, token)
            try:
                sid, ch = _session_and_challenge(
                    hdrs, parallel=parallel, cancel_event=cancel_event)
                return hdrs, sid, ch
            except _StreamCancelled:
                raise
            except Exception as exc:
                last_exc = exc
                exc_s = str(exc)
                # Ban at session/create (biz 5 "user is muted"): mark + 403
                # so the failover helper retries on the next account. The
                # probe supplies mute_until for a precise cooldown; the ban
                # payload itself is trusted even when the probe has no ts.
                if P.is_ban_error_text(exc_s):
                    _, ban_until = _probe_ban(
                        app, token,
                        provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
                    raise _ban_403_for_item(app, item, ban_until, exc_s)
                if attempt == 0 and _is_invalid_token_error(exc_s):
                    # Old token 40003 may be a muted ban masquerading as
                    # invalid: probe before spending a ~2min browser run.
                    # Probe uses users/current: 40003 there means stale token
                    # (not banned); a mute payload means banned. An invalid
                    # token can never prove a ban — only the browser page can.
                    banned, ban_until = _probe_ban(
                        app, token,
                        provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
                    if banned:
                        raise _ban_403_for_item(app, item, ban_until, exc_s)
                    logger.debug("ban probe for %s: not banned "
                                 "(old token invalid), trying browser refresh",
                                 item.cfg.identifier)
                    if _refresh_token_once():
                        continue
                if _is_invalid_token_error(exc_s):
                    # Fresh token also 40003 (or refresh impossible): probe
                    # once more — a ban that presents as 40003 must surface
                    # as 403 + cooldown, not a misleading 401. When the probe
                    # still says 40003 (invalid), there is no ban evidence, so
                    # 401 stays UNMARKED by design: marking it banned would
                    # false-positive wrong-password/captcha outages.
                    banned, ban_until = _probe_ban(
                        app, token,
                        provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
                    if banned:
                        raise _ban_403_for_item(app, item, ban_until, exc_s)
                    note = f" Browser login note: {login_note[:300]}." if login_note else ""
                    raise HTTPException(
                        status_code=401,
                        detail=f"Authorization Failed: token invalid for "
                               f"{item.cfg.identifier} and could not be "
                               f"refreshed ({exc_s[:200]}).{note} Re-login the account. "
                               f"If the web UI shows suspension/violation, "
                               f"treat it as BANNED.")
                raise
        raise last_exc if last_exc else RuntimeError("session setup failed")

    # Fast path: skip cookie-header builds when wasm already cached.
    try:
        wp = PoW.wasm_path()
        if not (wp.exists() and wp.stat().st_size > 1000):
            _bridge = getattr(app.state, "bridge", None)
            PoW.ensure_wasm(cookies=_bridge.cookie_header() if _bridge else "",
                            user_agent=_bridge.state.user_agent if _bridge else "")
    except Exception as exc:
        logger.warning("wasm ensure failed: %s", exc)
    # Session creation + PoW solve sit inside the try so a solver failure
    # (answer None -> RuntimeError) still runs the finally cleanup below.
    try:
        if _cancelled():
            raise _StreamCancelled()
        headers, session_id, challenge = _open_session()
        if _cancelled():
            raise _StreamCancelled()
        headers = {**headers, "X-DS-PoW-Response": _header_from_challenge(challenge)}
        if _cancelled():
            raise _StreamCancelled()
        payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                       prep["search"], prep["model_type"])
        # Ban expiry rides along with the completion error event (mute_until
        # from the ban envelope) so the ban path below needs no extra HTTP.
        # check_ban stays only as fallback when the stream carried no ts.
        stream_ban_until: float | None = None
        # Live mode already _put partial deltas to the client; a retry after
        # partial emit would re-stream from 0 (duplicated/interleaved text
        # and corrupt tool-call JSON). Track emits so overload retries skip
        # when output already left the server.
        emitted_live = False

        def _wrapped_content(t: str) -> None:
            nonlocal emitted_live
            if t:
                emitted_live = True
            if on_content is not None:
                try:
                    on_content(t)
                except Exception:
                    pass

        def _wrapped_thinking(t: str) -> None:
            nonlocal emitted_live
            if t:
                emitted_live = True
            if on_thinking is not None:
                try:
                    on_thinking(t)
                except Exception:
                    pass

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

        def _fresh_pow() -> None:
            """Re-solve PoW into ``headers`` (the old header may have expired)."""
            nonlocal headers
            headers = _solve(app, token)

        def _recover_pow() -> None:
            """PoW rejection: fresh solve + retry."""
            logger.info("pow rejected, solving fresh (retries left %s)", max_retries)
            _fresh_pow()
            _do_attempt(headers, payload)

        def _recreate_and_attempt(refresh_token: bool) -> None:
            """Shared session-recreate path for INVALID_SESSION / INVALID_TOKEN.

            Fresh payload is bound to the new session id and a fresh PoW header
            is solved (the old one may have expired). ``refresh_token`` runs the
            one-shot Obscura refresh first and deletes the old session under the
            headers it was created with (the new token would 401 the delete).
            """
            nonlocal session_id, headers, payload
            old_sid = session_id
            old_headers = _headers(app, token)
            if refresh_token:
                if not _refresh_token_once():
                    return
                logger.info("retrying completion with refreshed token")
            try:
                session_id = DS.create_session(_headers(app, token))
            except Exception as exc:
                exc_s = str(exc)
                # Recreated session can itself reveal a ban (biz 5): it must
                # mark + 403 for failover, not leak as RuntimeError.
                if P.is_ban_error_text(exc_s):
                    _, _ban_until = _probe_ban(
                        app, token,
                        provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
                    raise _ban_403_for_item(app, item, _ban_until, exc_s)
                # A 40003 on the recreated session with a freshly refreshed
                # token may be a ban presenting as invalid — probe once.
                if refresh_token and _is_invalid_token_error(exc_s):
                    _banned, _ban_until = _probe_ban(
                        app, token,
                        provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
                    if _banned:
                        raise _ban_403_for_item(app, item, _ban_until, exc_s)
                raise
            # Old id was reported invalid upstream, but delete best-effort so
            # a stale-but-valid id never lingers when the error text lied.
            if old_sid and old_sid != session_id:
                _delete_session_bg(old_headers, old_sid, enabled=auto_delete)
            _fresh_pow()
            payload = P.completion_payload(session_id, prep["prompt"], prep["thinking"],
                                           prep["search"], prep["model_type"])
            _do_attempt(headers, payload)

        def _recover_invalid_session() -> None:
            logger.info("session invalid, recreating (retries left %s)", max_retries)
            _recreate_and_attempt(refresh_token=False)

        def _recover_invalid_token() -> None:
            """Mid-stream 40003: refresh once, recreate the session, retry."""
            _recreate_and_attempt(refresh_token=True)

        def _recover_waf() -> None:
            """Refresh WAF cookies (slot released during the global wait)."""
            logger.info("possible WAF block, refreshing cookies (retries left %s)", max_retries)
            # Global refresh (singleflight in bridge) must not hold the
            # per-account slot while all in-flight requests wait on it.
            _refresh_waf_without_account_lock(app, item, cancel_event=cancel_event)
            if _cancelled():
                return
            _fresh_pow()
            _do_attempt(headers, payload)

        def _recover_overload() -> None:
            """Retry transient overload with a fresh PoW header."""
            logger.info("upstream overloaded (%s), retrying (retries left %s)",
                        (error or "")[:80], max_retries)
            # Fresh PoW: the prior header may have expired during a long
            # reasoning stream; reusing it risks POW_HEADER_ERROR on retry.
            try:
                _fresh_pow()
            except Exception as exc:
                logger.warning("overload retry PoW refresh failed: %s", exc)
            _do_attempt(headers, payload)

        content = think = ""
        error: str | None = None
        usage_total = 0
        _do_attempt(headers, payload)
        # Ordered like the historical loops (pow -> session -> token -> waf ->
        # overload); a later kind sees whatever error the earlier one left.
        # Overload is not retried after a partial live emit (would duplicate).
        drive_retries(
            max_retries,
            _cancelled,
            lambda: error,
            (
                ("pow", lambda e: classify_error(e) == RetryKind.POW,
                 _recover_pow),
                ("invalid_session",
                 lambda e: classify_error(e) == RetryKind.INVALID_SESSION,
                 _recover_invalid_session),
                ("invalid_token",
                 lambda e: classify_error(e) == RetryKind.INVALID_TOKEN,
                 _recover_invalid_token),
                ("waf", lambda e: classify_error(e) == RetryKind.WAF,
                 _recover_waf),
                ("overload",
                 lambda e: classify_error(e) == RetryKind.OVERLOAD and not emitted_live,
                 _recover_overload),
            ),
        )
        if _cancelled():
            raise _StreamCancelled()
        if error:
            _raise_for_terminal_error(app, item, prep, token, error,
                                      login_note, stream_ban_until)
        if not content and not think:
            # Upstream closed without content/thinking/error (e.g. unparsed toast).
            # Never return silent empty 200: it surfaces as "tool stopped working".
            raise HTTPException(status_code=502,
                                detail="empty upstream response from DeepSeek "
                                       "(likely throttling; wait then retry)")
        return {"content": content, "thinking": think, "usage_total": usage_total}
    finally:
        try:
            # Guard session_id first so no header build happens when there is
            # nothing to delete; _headers itself is bridge-safe (getattr).
            if auto_delete and session_id:
                _delete_session_bg(_headers(app, token), session_id, enabled=True)
        except Exception:
            pass


class _AttemptOutcome(tuple):
    """4-tuple (content, thinking, error, usage) + ban_until attr.

    Subclass so existing unpacking `a, b, c, d = _attempt(...)` keeps
    working (tests + callers), while _run_completion_core reads
    `.ban_until` to skip the extra users/current HTTP on bans.
    Plain-tuple mocks simply have no attr -> getattr(..., None) fallback.
    """

    def __new__(cls, content: str, think: str, error: str | None,
                usage: int, ban_until: float | None = None):
        obj = super().__new__(cls, (content, think, error, usage))
        obj.ban_until = ban_until
        return obj


def _attempt(headers: dict, payload: dict, on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             ) -> tuple[str, str, str | None, int]:
    content: list[str] = []
    think: list[str] = []
    error: str | None = None
    usage_total = 0
    ban_until: float | None = None
    # Explicit gen.close(): break out of a generator does NOT run its
    # finally (resp.close() in client.stream_completion) until close()/GC.
    # Close promptly so the curl connection returns to the pool instead of
    # lingering through retries (burst contention) or GC timing.
    gen = DS.stream_completion(headers, payload)
    try:
        for event in gen:
            if cancel_event is not None and cancel_event.is_set():
                # Client gone: stop reading and close below so the HTTP
                # response closes and the account lock releases promptly.
                break
            if event.kind == "content":
                content.append(event.text)
                if on_content is not None:
                    try:
                        on_content(event.text)
                    except Exception:
                        pass
            elif event.kind == "thinking":
                think.append(event.text)
                if on_thinking is not None:
                    try:
                        on_thinking(event.text)
                    except Exception:
                        pass
            elif event.kind == "usage":
                try:
                    usage_total = max(usage_total, int(event.text))
                except ValueError:
                    pass
            elif event.kind == "error":
                error = f"{event.code}: {event.message}"
                try:
                    ban_until = getattr(event, "ban_until", None)
                except Exception:
                    ban_until = None
                break
            elif event.kind == "finished":
                break
    finally:
        try:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    return _AttemptOutcome("".join(content), "".join(think), error,
                           usage_total, ban_until)


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = prep["model"]
    out: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "live")).lower() == "live"
    # Consumer wait must outlast the blocking curl stream timeout, otherwise the
    # async reader times out while the producer is still working on a valid long
    # reply. Derive it so the two can never drift apart.
    consumer_timeout = float(DS.DEFAULT_TIMEOUT) + _CONSUMER_GRACE_SECONDS
    # The engine owns the producer bridge's cancel flag: the client-disconnect
    # event is set on generator close, the producer checks it between SSE
    # events, and the account lock is released promptly.
    bridge = ProducerBridge(_STREAM_EXECUTOR)
    cancel_event = bridge.cancel_event

    def _emit(item: tuple) -> None:
        # call_soon_threadsafe raises RuntimeError once the loop is closed
        # (shutdown). Swallow: producer work is best-effort at that point.
        try:
            loop.call_soon_threadsafe(out.put_nowait, item)
        except RuntimeError:
            pass

    def _put(kind: str, text: str) -> None:
        if text:
            _emit((kind, text))

    def produce_buffered():
        # Ban failover lives in the helper (owns item's lock): a banned
        # account retries on the next unbanned one inside the same request.
        # Ban errors carry no content prefix, so nothing partial leaks.
        try:
            result = _complete_with_failover_sync(
                app, pool, prep, item, cancel_event=cancel_event)
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))

    def produce_live():
        # Thinking always streams live (never contains tool JSON).
        # Content streams live only when no tools requested; with tools the
        # content is buffered so a tool_calls reply is not also emitted as
        # user-visible text.
        try:
            result = _complete_with_failover_sync(
                app, pool, prep, item,
                on_content=(None if prep.get("tools") else lambda t: _put("content", t)),
                on_thinking=lambda t: _put("thinking", t),
                cancel_event=cancel_event,
            )
            _emit(("done", result))
        except _StreamCancelled:
            pass
        except HTTPException as exc:
            _emit(("http_error", exc))
        except Exception as exc:  # noqa: BLE001
            _emit(("error", str(exc)))

    def chunk(delta: dict, finish: str | None = None) -> bytes:
        return ("data: " + json.dumps(
            {"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
            ensure_ascii=False) + "\n\n").encode("utf-8")

    async def _parse_stream_calls(content: str):
        """Tool-parse buffered content (tools requested); log when none found."""
        limit = prep.get("tool_args_max_chars")
        forgiving = bool(prep.get("forgiving_toolcalls", False))
        if limit:
            calls, remaining = await asyncio.to_thread(
                _parse_tool_calls_local, content, prep["tools"],
                max_args_chars=limit, forgiving=forgiving)
        else:
            calls, remaining = await asyncio.to_thread(
                _parse_tool_calls_local, content, prep["tools"], forgiving=forgiving)
        truncated = False
        if not calls:
            truncated = bool(looks_truncated_tool_attempt(
                content, prep["tools"], forgiving=forgiving))
            if truncated:
                logger.warning("stream tool call looks truncated model=%s preview=%.200s",
                               prep.get("model"), (content or "")[:200])
            else:
                logger.debug("stream no tool call parsed model=%s preview=%.200s",
                             prep.get("model"), (content or "")[:200])
        return calls, remaining, truncated

    def _tool_finish_chunks(calls, truncated: bool = False):
        """tool_call deltas + finish chunk; length when a call was cut off."""
        if calls:
            for call in calls:
                yield chunk({"tool_calls": [{
                    "id": call["id"], "type": "function",
                    "function": {"name": call["function"]["name"],
                                 "arguments": call["function"]["arguments"]}}]})
            yield chunk({}, "tool_calls")
        else:
            yield chunk({}, "length" if truncated else "stop")

    # Bounded producer: one account-holding thread per stream, capped by the
    # shared executor so a parallel-subagent burst queues instead of piling
    # unbounded raw threads (each with account + PoW CPU).
    try:
        _STREAM_EXECUTOR.submit(produce_live if live else produce_buffered)
    except RuntimeError:
        # Executor shut down (interpreter exit): fall back to a raw thread.
        threading.Thread(target=produce_live if live else produce_buffered, daemon=True).start()
    try:
        yield chunk({"role": "assistant"})
        if not live:
            # Buffered: producer sends one reply per request; emit thinking+content
            # once the reply lands, then tool deltas. First-byte ~= model latency.
            while True:
                try:
                    kind, payload = await asyncio.wait_for(out.get(), timeout=consumer_timeout)
                except asyncio.TimeoutError:
                    yield ("data: " + json.dumps({"error": {"message": "upstream timed out"}},
                                                ensure_ascii=False) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    return
                if kind == "done":
                    thinking, content = payload["thinking"], payload["content"]
                    truncated = False
                    if prep["tools"]:
                        calls, remaining, truncated = await _parse_stream_calls(content)
                    else:
                        calls, remaining = None, content
                    if thinking:
                        yield chunk({"reasoning_content": thinking})
                    if remaining:
                        yield chunk({"content": remaining})
                    for piece in _tool_finish_chunks(calls, truncated):
                        yield piece
                    yield b"data: [DONE]\n\n"
                    return
                if kind == "http_error":
                    err = payload
                    yield ("data: " + json.dumps({"error": {"message": err.detail, "code": err.status_code}},
                                                ensure_ascii=False) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    return
                yield ("data: " + json.dumps({"error": {"message": str(payload)}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
        # Live: forward thinking/content deltas as they arrive from DeepSeek.
        while True:
            try:
                kind, payload = await asyncio.wait_for(out.get(), timeout=consumer_timeout)
            except asyncio.TimeoutError:
                yield ("data: " + json.dumps({"error": {"message": "upstream timed out"}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
            if kind == "thinking":
                yield chunk({"reasoning_content": payload})
                continue
            if kind == "content":
                yield chunk({"content": payload})
                continue
            if kind == "done":
                content = payload["content"]
                if prep.get("tools"):
                    calls, remaining, truncated = await _parse_stream_calls(content)
                    # Thinking already streamed live; emit only the remainder.
                    if remaining:
                        yield chunk({"content": remaining})
                    for piece in _tool_finish_chunks(calls, truncated):
                        yield piece
                else:
                    # Content already streamed live; just close.
                    yield chunk({}, "stop")
                yield b"data: [DONE]\n\n"
                return
            if kind == "http_error":
                err = payload
                yield ("data: " + json.dumps({"error": {"message": err.detail, "code": err.status_code}},
                                            ensure_ascii=False) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                return
            yield ("data: " + json.dumps({"error": {"message": str(payload)}},
                                        ensure_ascii=False) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            return
    finally:
        # Generator closed (client disconnect, cancellation, completion): tell
        # the producer to stop at the next event so the account lock frees.
        cancel_event.set()
