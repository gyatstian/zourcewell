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

import enum
import logging
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from . import client as DS
from . import pow as PoW
from . import protocol as P
from ...accounts import AccountPool, PooledAccount
from ...config import Settings
from ...core import engine as _engine
from ...core.engine import (
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
    from ...accounts import TOKEN_HELP

    return _engine.ensure_token(
        item, f"account {item.cfg.identifier} has no token. " + TOKEN_HELP)


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

    Order: pow -> session -> token -> waf -> rate-limit -> overload -> ban ->
    401 -> upstream-invalid -> upstream. Returns None only when there is no
    error text to classify.

    The retry table additionally guards EVERY kind with the "partial live emit"
    flag (a re-POST plus re-stream would duplicate text); that is a
    retry-policy decision, not a classification, so it stays in the table
    rather than here.
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
    # Rate limit BEFORE transient overload: a throttled body commonly also
    # says "try again"/"busy" ("429 ... please try again"), and classified
    # OVERLOAD it would be retried with a fresh PoW into the same throttle
    # (503) instead of the terminal 429 the RATE_LIMIT row gives. Qwen's
    # classifier orders these the same way.
    if _is_rate_limit_error(error):
        return RetryKind.RATE_LIMIT
    if _is_transient_overload(error):
        # Partial live emit: do not retry (would duplicate), but the
        # terminal ladder still surfaces it as 503. Signal that by
        # returning OVERLOAD anyway; the retry recover checks the guard.
        return RetryKind.OVERLOAD
    if P.is_ban_error_text(error) or "restricted" in low:
        return RetryKind.BAN
    # A bare HTTP 401 is an AUTH failure, not a suspension.
    # ``transport.http_error_code(401)`` renders as "HTTP_401: <body>": it
    # carries neither "40003" nor "authorization failed" (so
    # ``_is_invalid_token_error`` misses it) and no ban word, but it used to
    # match the "401" entry of the ban family below, so one transient 401
    # labelled a healthy account ``banned: true`` in config.json forever and
    # answered 403 with ban wording. INVALID_TOKEN is both retryable and
    # terminal-mapped: ``_terminal_invalid_token`` probes users/current
    # first, so a genuinely muted account still becomes 403 + cooldown while
    # an unconfirmed 401 surfaces as an UNMARKED 401. Qwen already mapped it
    # this way; the two providers now agree.
    if "401" in low or "unauthorized" in low:
        return RetryKind.INVALID_TOKEN
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
    """Mark + persist an account ban and build the 403 (delegates to engine)."""
    return _engine.ban_403_for_item(
        app, item, ban_until, error_text,
        ban_message_fn=(lambda u: P.ban_message(u) if u else ""))


def _refresh_token_via_obscura(app: FastAPI, item: PooledAccount,
                               cancel_event: threading.Event | None = None,
                               ) -> tuple[str | None, bool, str]:
    """Re-login through the Obscura browser and persist a fresh userToken.

    Returns (token_or_None, login_banned, ban_evidence). login_banned=True is
    ground truth from the login page suspension banner — the caller must treat
    the account as banned even when the API only ever said 40003.
    Singleflight per account: concurrent 40003s share one browser run.
    Never raises (except _StreamCancelled when the client vanished during
    the browser run or the wait).
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
        # Token BEFORE the wait: only a value that differs afterwards is the
        # leader's fresh token. Returning an unchanged one as "refreshed" made
        # the caller retry once with the same invalid token and report a
        # misleading outcome.
        before = (item.cfg.token or "").strip()
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
        refreshed = fresh if (fresh and fresh != before) else None
        return refreshed, banned, detail
    # Backoff: a recent no-token login (captcha/credential) means the login
    # page is challenging this profile right now. Spawning the browser on
    # every 40003 flags it further — skip fast with the previous evidence.
    # NOTE: the backoff check must precede the singleflight leave below:
    # a skipped run still wakes followers (they read the ban hint + token).
    skip_reason = item.login_backoff_skip()
    if skip_reason:
        msg = (f"browser login skipped for {key} ({skip_reason}; "
               f"manual set-token bypasses this)")
        logger.info(msg)
        _leave_token_refresh(key, inflight, banned=False, detail=msg)
        return None, False, msg
    try:
        # The browser run (profile-lock wait up to 180s + LOGIN_TIMEOUT 90s)
        # is global work: release the per-account slot while it runs so other
        # requests can fail over to healthy accounts instead of queueing
        # behind a minute-long login (same pattern as the WAF refresh below).
        # Another request may meanwhile run an attempt on this account with
        # the stale token; it 40003s and joins this run as a follower —
        # strictly better than parking behind the leader's slot.
        if cancel_event is not None and cancel_event.is_set():
            raise _StreamCancelled()
        try:
            with _slot_releaser(item, cancel_event):
                new_token, banned, detail = _run_obscura_login(
                    app, item, settings, key)
        except _StreamCancelled:
            # slot_releaser's reacquire check raises when the client
            # disconnected during the browser run. The login itself is real
            # work (up to 180s profile lock + 90s login): fall through and
            # persist it rather than throw the token away. The new_token
            # path re-raises below; the banned/no-token paths re-raise in
            # the caller's post-refresh check.
            pass
        try:
            with _TOKEN_REFRESH_LOCK:
                _TOKEN_REFRESH_BAN_HINT[key] = (bool(banned), str(detail or ""))
        except Exception:
            pass
        if banned:
            # Scrubbed for the same reason as the no-token log below: `detail`
            # is the raw Obscura login result, and on a banned page it is the
            # page body itself. A slice is a length guard, not a secret guard.
            logger.warning("Obscura login page shows account %s BANNED: %s",
                           key, _engine.scrub_upstream_text(detail, 200))
            item.clear_login_backoff()
            _leave_token_refresh(key, inflight, banned=True,
                                 detail=str(detail or ""))
            return None, True, str(detail or "")
        if new_token:
            from ...accounts import set_account_token
            item.cfg.token = new_token
            try:
                # Scoped to the account's own provider: settings may hold the
                # same identity under another provider, and the unscoped first
                # match wrote this token onto that record.
                set_account_token(settings, key, new_token,
                                  provider=getattr(item.cfg, "provider", "")
                                  or "deepseek")
                settings.save()
            except Exception as exc:
                logger.warning("token refresh save failed for %s: %s", key, exc)
            logger.info("token refreshed via Obscura for %s", key)
            item.clear_login_backoff()
            _leave_token_refresh(key, inflight)
            # Cancellation check AFTER persistence + wake: a disconnect must
            # not discard a token the browser just produced. Followers were
            # already woken with the new value in place.
            if cancel_event is not None and cancel_event.is_set():
                raise _StreamCancelled()
            return new_token, False, ""
        # No token, no ban page: back off the browser so repeated 40003s
        # don't hammer the challenge endpoint. Manual set-token clears it.
        if detail:
            # Scrubbed: the note is the Obscura login result, which for a
            # no-token run is the raw page/body (localStorage JSON with the
            # userToken in it). It reaches the 401 detail AND this log, so
            # the long opaque runs go before either.
            login_note = _engine.scrub_upstream_text(detail, 500)
            logger.warning("Obscura login for %s produced no usable token: %s",
                           item.cfg.identifier, login_note[:300])
        else:
            login_note = ""
        low = (detail or "").lower()
        if "captcha" in low:
            item.set_login_backoff("captcha-challenge", 180)
        elif ("credential" in low or "incorrect" in low
                or "wrong password" in low):
            item.set_login_backoff("credential-error", 600)
        elif "form interaction failed" in low:
            item.set_login_backoff("login-form-changed", 600)
        else:
            item.set_login_backoff("no-token", 120)
        _leave_token_refresh(key, inflight, banned=False, detail=login_note)
        return None, False, login_note
    except _StreamCancelled:
        # Slot already released by _slot_releaser; just wake followers.
        _leave_token_refresh(key, inflight)
        raise
    except Exception:
        _leave_token_refresh(key, inflight)
        raise


def _solve(app: FastAPI, token: str) -> dict[str, str]:
    """Fresh PoW header merged over base headers."""
    headers = _headers(app, token)
    return {**headers, "X-DS-PoW-Response": DS.solve_pow(headers)}


def _delete_observed_session_on_completion(headers: dict,
                                          created: list,
                                          fut) -> None:
    """Delete a session whose creation we stopped waiting for (best effort).

    ``concurrent.futures.Future.cancel()`` returns False once the callable
    has started, so ``_await_cancelable``'s cancel of a create_session that
    is already in flight does nothing: the request raises
    ``_StreamCancelled`` while the executor worker goes on to create the
    session anyway. The id that call is about to produce then lands nowhere,
    the ``finally`` in ``_run_completion_core`` has no id to delete, and the
    session lingers upstream (plus a challenge worker stays pinned for the
    full create_session timeout).

    Fix without changing the ``create_session(headers)`` call signature
    (tests and the real client both call it with exactly one argument): the
    submitted callable records the id in a one-element holder that travels
    out of band, and this attaches a done-callback that deletes it. The
    callback runs on the worker that just created the session, or inline if
    the future already completed -- either way the abort path does not wait
    for a network round trip, and a create_session that raised (or never
    started) leaves the holder empty and nothing is signalled.
    """
    def _cleanup(_fut) -> None:
        try:
            session_id = created[0] if created else None
        except Exception:
            session_id = None
        if not session_id:
            return
        try:
            _delete_session_bg(headers, session_id, enabled=True)
        except Exception:
            pass
    try:
        fut.add_done_callback(_cleanup)
    except Exception:
        _cleanup(fut)


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
    # Out-of-band id holder: see _delete_observed_session_on_completion. The
    # wrapper keeps DS.create_session(headers)'s one-argument call shape, so
    # every existing patch seam (`lambda headers: "sess-1"`) still applies.
    created: list = [None]

    def _create_session_observed(hdrs: dict) -> str:
        session_id = DS.create_session(hdrs)
        if session_id:
            created[0] = session_id
        return session_id

    fut_session = _CHALLENGE_EXECUTOR.submit(_create_session_observed, headers)
    fut_challenge = _CHALLENGE_EXECUTOR.submit(DS.fetch_pow, headers)
    try:
        session_id = _await_cancelable(fut_session, cancel_event)
    except _StreamCancelled:
        try:
            fut_challenge.cancel()
        except Exception:
            pass
        # The cancel above is a no-op for an already-running create_session:
        # collect the id it still produces instead of leaking the session.
        _delete_observed_session_on_completion(headers, created, fut_session)
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
    return _engine.app_settings(app)


def _leave_token_refresh(key: str, inflight: threading.Event,
                       banned: bool = False, detail: str = "") -> None:
    """Wake singleflight followers: record the ban hint, pop, set."""
    try:
        with _TOKEN_REFRESH_LOCK:
            _TOKEN_REFRESH_BAN_HINT[key] = (bool(banned), str(detail or ""))
            _TOKEN_REFRESH_INFLIGHT.pop(key, None)
            try:
                inflight.set()
            except Exception:
                pass
    except Exception:
        pass


def _run_obscura_login(app: FastAPI, item: PooledAccount,
                       settings, key: str) -> tuple[str | None, bool, str]:
    """One Obscura browser login; returns (token_or_None, banned, detail)."""
    cfg = item.cfg
    from .auth import obscura_login_token as _login_token
    from .auth import unpack_login_result as _unpack_login
    try:
        raw = _login_token(cfg.email or cfg.mobile, cfg.password, settings)
        new_token, banned, detail = _unpack_login(raw)
        try:
            from ...accounts import extract_token as _extract_token
            new_token = _extract_token(new_token) or None
        except Exception:
            pass
    except Exception as exc:
        logger.warning("token refresh via Obscura failed for %s: %s",
                       key, _engine.scrub_upstream_text(exc, 300))
        new_token, banned, detail = None, False, ""
    return new_token, bool(banned), str(detail or "")


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
    with _slot_releaser(item, cancel_event):
        try:
            bridge = getattr(app.state, "bridge", None)
            if bridge is not None:
                bridge.warmup()
        except Exception as exc:
            logger.warning("WAF warmup failed: %s", exc)


def _is_ban_http_exception(exc: BaseException) -> bool:
    """True when exc is the ban 403 raised by _run_completion_core."""
    return _engine.is_ban_http_exception(exc, P.is_ban_error_text)


def _complete_with_failover_sync(app: FastAPI, pool: AccountPool, prep: dict,
                                 first_item: PooledAccount,
                                 cancel_event: threading.Event | None = None,
                                 on_content=None, on_thinking=None,
                                 allow_failover: bool | None = None) -> dict:
    """Run one completion, auto-failing over on ban or broken credentials.

    DeepSeek supplies the retryability predicate (ban 403 / auth 401) via
    module-global ``_is_ban_http_exception`` and its ``_run_completion_core``;
    ``core.engine`` owns the generic slot lifecycle. Late-binds both globals
    so monkeypatching this module keeps working.
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


def _invalid_token_401_detail(identifier: str, error: str,
                             login_note: str, refreshed: bool) -> str:
    """One 401 message for both invalid-token sites (session + terminal).

    Two code paths raise "token invalid" 401s (``_open_session`` after a
    failed refresh, ``_terminal_invalid_token`` after a mid-stream 40003);
    one helper keeps the wording — and the BANNED-vs-stale distinction —
    identical. (Smallest risk-reducing step toward typed errors; the full
    ``RetryKind``-carrying error-type-refactor stays deferred.)

    ``error`` and ``login_note`` are scrubbed before they are quoted: the
    first is the upstream biz body, the second the Obscura login result, and
    both end up in a client-visible 401 detail. Long opaque runs (the 64-char
    userToken is the realistic one) are redacted; the codes and the ban
    wording an operator needs are not touched. EVERY return path must quote
    ``err``/``note`` and never the raw arguments -- a second, unscrubbed
    return below an unconditional one is dead code today and a live leak the
    day someone reorders the ``if refreshed:`` above it.
    """
    err = _engine.scrub_upstream_text(error, 200)
    note = (f" Browser login note: {_engine.scrub_upstream_text(login_note, 300)}."
            if login_note else "")
    if refreshed:
        return (f"Authorization Failed: token invalid for {identifier} "
                f"even after browser refresh ({err}).{note} "
                f"Re-login the account. If the web UI shows "
                f"suspension/violation, treat it as BANNED.")
    return (f"Authorization Failed: token invalid for {identifier} "
            f"and could not be refreshed ({err}).{note} "
            f"Re-login the account. If the web UI shows "
            f"suspension/violation, treat it as BANNED.")



def _terminal_invalid_token(app: FastAPI, item: PooledAccount, prep: dict,
                            token: str | None, error: str, login_note: str,
                            stream_ban_until: float | None) -> None:
    """Terminal 40003: already refreshed once above; a muted ban can present
    as 40003, so probe users/current once before concluding "invalid" —
    otherwise bans surface as 401 and the account is never marked cooling."""
    _banned, _ban_until = _probe_ban(app, token)
    if _banned:
        raise _ban_403_for_item(app, item, _ban_until, error)
    raise HTTPException(
        status_code=401,
        detail=_invalid_token_401_detail(
            item.cfg.identifier, error, login_note, refreshed=True))


def _terminal_ban(app: FastAPI, item: PooledAccount, prep: dict,
                  token: str | None, error: str, login_note: str,
                  stream_ban_until: float | None) -> None:
    """Banned accounts stay unusable until mute_until: cool down for the
    remaining suspension. The completion ban envelope already carried
    mute_until (stream_ban_until) — use it directly and skip the extra
    users/current HTTP; fall back to check_ban only when the stream had no ts.

    The probe's BOOLEAN is honoured, not just its timestamp: a negative
    verdict means "not muted", so a stray ``(False, ts)`` must not be adopted
    as a ban cooldown (the ban text is evidence enough to answer 403 and fail
    over, but not to pin an expiry the probe just denied).
    """
    ban_until: float | None = stream_ban_until
    if ban_until is None:
        try:
            banned, probed_until = _probe_ban(
                app, token, timeout=5,
                provider=(getattr(item.cfg, "provider", "deepseek") or "deepseek"))
        except Exception:
            banned, probed_until = False, None
        ban_until = probed_until if banned else None
    raise _ban_403_for_item(app, item, ban_until, error)


def _terminal_upstream(app: FastAPI, item: PooledAccount, prep: dict,
                       token: str | None, error: str, login_note: str,
                       stream_ban_until: float | None) -> None:
    _engine.raise_upstream("DeepSeek", error)


#: Terminal dispatch table: one RetryKind (from the shared classifier) per
#: HTTP outcome. RATE_LIMIT/OVERLOAD/UPSTREAM_INVALID/UPSTREAM are the
#: label-only handlers every provider shares (core.engine), so a fix to one
#: raise_* message lands everywhere at once; only INVALID_TOKEN and BAN are
#: DeepSeek-specific. POW/INVALID_SESSION/WAF are retryable kinds that only
#: reach here after exhausting the retry budget -> generic bad-gateway,
#: exactly as the historical fallthrough did.
_TERMINAL_HANDLERS: dict[RetryKind, Callable[..., None]] = {
    **_engine.generic_terminal_handlers(RetryKind, "DeepSeek",
                                        upstream=_terminal_upstream),
    RetryKind.INVALID_TOKEN: _terminal_invalid_token,
    RetryKind.BAN: _terminal_ban,
    RetryKind.POW: _terminal_upstream,
    RetryKind.INVALID_SESSION: _terminal_upstream,
    RetryKind.WAF: _terminal_upstream,
}


def _raise_for_terminal_error(app: FastAPI, item: PooledAccount, prep: dict,
                              token: str | None, error: str,
                              login_note: str,
                              stream_ban_until: float | None) -> None:
    """Map a terminal completion error event to its HTTP status. Always raises."""
    _engine.dispatch_terminal(classify_error, _TERMINAL_HANDLERS,
                              app, item, prep, token,
                              error, login_note, stream_ban_until)


def _ensure_wasm_cached(app: FastAPI) -> None:
    """Best-effort PoW wasm prefetch; skips when already cached. Never raises."""
    # Fast path: skip cookie-header builds when wasm already cached.
    try:
        wp = PoW.wasm_path()
        if not (wp.exists() and wp.stat().st_size > 1000):
            _bridge = getattr(app.state, "bridge", None)
            PoW.ensure_wasm(cookies=_bridge.cookie_header() if _bridge else "",
                            user_agent=_bridge.state.user_agent if _bridge else "")
    except Exception as exc:
        logger.warning("wasm ensure failed: %s", exc)


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
    live stream mode. Once any delta has reached the client NO retry row fires
    again, whichever kind it is: the completion POST is not idempotent and a
    re-stream into a half-read stream duplicates text or interleaves two halves
    of one tool_calls JSON object, then ends in a clean [DONE]. Overload
    additionally uses a fresh PoW header. Buffered callers pass no callbacks and
    get identical behaviour to before.

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
            login_note = _engine.scrub_upstream_text(login_detail, 500)
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
                    raise HTTPException(
                        status_code=401,
                        detail=_invalid_token_401_detail(
                            item.cfg.identifier, exc_s, login_note,
                            refreshed=False))
                raise
        raise last_exc if last_exc else RuntimeError("session setup failed")

    _ensure_wasm_cached(app)
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
        # After-partial-emit guard lives in core.engine (LiveEmitTracker) and
        # now covers every row: a re-POST plus re-stream after output left
        # would duplicate text and corrupt tool-call JSON.
        _emit_tracker, _wrapped_content, _wrapped_thinking = _engine.live_emit_pair(
            on_content, on_thinking)

        def _do_attempt(h: dict, p: dict) -> None:
            nonlocal content, think, error, usage_total, stream_ban_until
            outcome = _attempt(
                h, p,
                on_content=(_wrapped_content if on_content is not None else None),
                on_thinking=(_wrapped_thinking if on_thinking is not None else None),
                cancel_event=cancel_event)
            content, think, error, usage_total = outcome
            cur_ban = _engine.ban_until_of(outcome)
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
            # Scrubbed before the 80-char slice: a slice is not a redactor, and
            # Qwen's twin of this line already says so at its own copy.
            logger.info("upstream overloaded (%s), retrying (retries left %s)",
                        _engine.scrub_upstream_text(error, 80), max_retries)
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
        def _retryable(kind) -> Callable[[str | None], bool]:
            """One predicate factory so NO row retries after a live emit.

            The guard used to sit on the overload row alone, on the assumption
            that the other kinds fail before any content. ``drain_attempt``
            makes content-then-error a legal ordering, and the POST is not
            idempotent (upstream appended a turn), so a pow/session/token/waf
            recovery here re-POSTs the same payload on the same session and
            re-streams from delta 0 into a stream the client already read:
            duplicated text, or two interleaved halves of one tool_calls JSON
            object, then a clean ``[DONE]`` and no error to stop the harness
            executing the corrupt call.
            """
            def _ok(e: str | None) -> bool:
                return (classify_error(e) == kind
                        and not _emit_tracker.emitted)

            return _ok

        drive_retries(
            max_retries,
            _cancelled,
            lambda: error,
            (
                ("pow", _retryable(RetryKind.POW), _recover_pow),
                ("invalid_session", _retryable(RetryKind.INVALID_SESSION),
                 _recover_invalid_session),
                ("invalid_token", _retryable(RetryKind.INVALID_TOKEN),
                 _recover_invalid_token),
                ("waf", _retryable(RetryKind.WAF), _recover_waf),
                ("overload", _retryable(RetryKind.OVERLOAD), _recover_overload),
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
            raise _engine.empty_response_error("DeepSeek")
        return {"content": content, "thinking": think, "usage_total": usage_total}
    finally:
        try:
            # Guard session_id first so no header build happens when there is
            # nothing to delete; _headers itself is bridge-safe (getattr).
            if auto_delete and session_id:
                _delete_session_bg(_headers(app, token), session_id, enabled=True)
        except Exception:
            pass


def _attempt(headers: dict, payload: dict, on_content=None, on_thinking=None,
             cancel_event: threading.Event | None = None,
             ) -> tuple[str, str, str | None, int]:
    """Drain one SSE stream (engine-owned); keeps this module's patch seam."""
    return _engine.drain_attempt(DS.stream_completion, headers, payload,
                                 on_content=on_content, on_thinking=on_thinking,
                                 cancel_event=cancel_event)


def _run_completion(app: FastAPI, item: PooledAccount, prep: dict,
                    cancel_event: threading.Event | None = None) -> dict:
    """Blocking full completion with retries per failure class."""
    return _run_completion_core(app, item, prep, cancel_event=cancel_event)


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end.

    Producer (account-holding thread) is DeepSeek-specific only in the
    failover entrypoint; queue/cancel/emit shape + SSE consumer are
    engine-owned (``new_producer_context`` / ``consume_stream``).
    """
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "live")).lower() == "live"
    (cid, created, out, cancel_event, consumer_timeout, model,
     _emit, _put) = _engine.new_producer_context(
        prep, DS.DEFAULT_TIMEOUT, _CONSUMER_GRACE_SECONDS, buffered=not live)

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
                # Ban failover lives in the helper (owns item's lock): a banned
                # account retries on the next unbanned one inside the same request.
                # Ban errors carry no content prefix, so nothing partial leaks.
                result = _complete_with_failover_sync(
                    app, pool, prep, item, cancel_event=cancel_event)
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
        _produce(live=False)

    def produce_live():
        _produce(live=True)

    # Bounded producer: one account-holding thread per stream, capped by the
    # shared executor so a parallel-subagent burst queues instead of piling
    # unbounded raw threads (each with account + PoW CPU).
    try:
        _STREAM_EXECUTOR.submit(produce_live if live else produce_buffered)
    except RuntimeError:
        # Executor shut down (interpreter exit): fall back to a raw thread.
        threading.Thread(target=produce_live if live else produce_buffered, daemon=True).start()
    async for frame in _engine.consume_stream(
            out, prep, live, cid, created, model, consumer_timeout,
            _parse_tool_calls_local, looks_truncated_tool_attempt,
            cancel_event):
        yield frame
