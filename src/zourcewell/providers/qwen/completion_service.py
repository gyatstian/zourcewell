"""Qwen completion engine: failover, retries, chat lifecycle.

Qwen-specific adapter over the provider-neutral mechanics in
``zourcewell.core.engine``. Unlike DeepSeek there is no PoW, no wasm, no
browser token refresh and no chat-list management: a completion is
create-chat -> POST /api/v2/chat/completions -> best-effort delete-chat.
WAF recovery is a bridge warmup (Aliyun anti-bot), not a challenge solve.
The canonical monkeypatch surface is this module
(``providers.qwen.completion_service``).
"""

from __future__ import annotations

import datetime
import enum
import logging
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from . import auth as QA
from . import client as QC
from . import protocol as P
from ...accounts import AccountPool, PooledAccount
from ...config import Settings
from ...core import engine as _engine
from ...core.engine import (
    drive_retries,
    slot_releaser as _slot_releaser,
)
from ...core.obscura_bridge import ObscuraBridge
from ...core.tools import (
    looks_truncated_tool_attempt,
    parse_tool_calls as _parse_tool_calls_local,
)
from ...server_concurrency import (
    _CLEANUP_EXECUTOR,
    _CONSUMER_GRACE_SECONDS,
    _StreamCancelled,
)

logger = logging.getLogger("zourcewell.server")


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

    Families per the Qwen recon (see README error table): ban/suspension ->
    BAN; WAF/challenge -> WAF; 401/unauthorized -> INVALID_TOKEN; rate limits
    -> RATE_LIMIT; 422/400 -> UPSTREAM_INVALID; overload/busy/timeout/502 ->
    OVERLOAD; anything else -> UPSTREAM. Every kind also carries the
    partial-live-emit guard (see providers/AGENTS.md). Marker tuples come from
    ``core.engine`` (RATE_LIMIT_MARKERS / UPSTREAM_INVALID_MARKERS /
    TRANSIENT_OVERLOAD_MARKERS) plus this upstream's "quota" extra, so a bare
    "502"/"500" classifies like "bad gateway" here too.

    BAN is checked FIRST: Qwen's WAF answers 200 + text/html, never 403, so a
    403 is a real upstream status whose body may be a suspension -- and no
    Qwen WAF string carries a ban-family word (WAF_CHALLENGE / aliyun_waf /
    AliyunCaptcha / nc-container), so ban-first cannot steal a WAF verdict.
    Matching the WAF branch on the "403" substring first swallowed one such
    body: no cooldown, no failover, a pointless 502 after re-primes.
    """
    if not error:
        return None
    low = error.lower()
    if P.is_ban_error_text(error) or "restricted" in low:
        return RetryKind.BAN
    if "waf" in low or "403" in low or "challenge" in low:
        return RetryKind.WAF
    if any(k in low for k in ("401", "unauthorized", "invalid", "authorization")):
        return RetryKind.INVALID_TOKEN
    if any(k in low for k in _engine.RATE_LIMIT_MARKERS + (
            "quota", "quota_limit")):
        return RetryKind.RATE_LIMIT
    if any(k in low for k in _engine.UPSTREAM_INVALID_MARKERS):
        return RetryKind.UPSTREAM_INVALID
    if any(k in low for k in _engine.TRANSIENT_OVERLOAD_MARKERS):
        # Partial live emit: do not retry (would duplicate), but the
        # terminal ladder still surfaces it as 503. Signal that by returning
        # OVERLOAD anyway; the retry recover checks the guard.
        return RetryKind.OVERLOAD
    return RetryKind.UPSTREAM


def _headers(app: FastAPI, token: str) -> dict:
    # getattr guard: embedded/test apps may have no lifespan bridge. Missing
    # bridge must not mask the real error with AttributeError (notably from
    # the delete-chat cleanup finally, which must never replace the exception).
    # The app bridge contributes its User-Agent only: its cookie jar is the
    # DeepSeek-host WAF jar (warmup() fetches CHAT_HOME), so it must never ride
    # a qwen request. Qwen cookies are its own qwen-host acw_tc + x-ap/sca/cna/
    # tfstk/isg/xlly_s harvest (prime/fetch), merged below so Cookie carries
    # token= + the jar (Bearer kept as well).
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
            return _with_cached("", bridge.state.user_agent)
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


def _ensure_token(item: PooledAccount) -> str:
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


# Backoff after a failed direct re-signin so agentic 401 loops don't hammer
# /api/v1/auths/signin: WAF/challenge 60s, wrong-password 300s.
_RELOGIN_WAF_BACKOFF_SECONDS = 60.0
_RELOGIN_CRED_BACKOFF_SECONDS = 300.0
# Short tokens are unit-test fakes; probing them wastes a request (mirrors
# accounts.MIN_TOKEN_LEN_FOR_BAN_CHECK).
_MIN_TOKEN_LEN_FOR_BAN_PROBE = 20


def _probe_ban(app: FastAPI, token: str | None,
               timeout: int = 12) -> tuple[bool, float | None]:
    """Best-effort ban probe for one Qwen token. Never raises."""
    try:
        tok = (token or "").strip()
        if len(tok) < _MIN_TOKEN_LEN_FOR_BAN_PROBE:
            return False, None
        try:
            headers = _headers(app, tok)
        except Exception:
            headers = P.base_headers(bearer=tok)
        return QC.check_ban(headers, timeout=timeout)
    except Exception:
        return False, None


def _try_relogin(app: FastAPI, item: PooledAccount) -> str | None:
    """Direct-API re-signin with the stored email/password.

    Cheap POST, no browser/captcha. On success persists the fresh JWT +
    ``token_expires_at`` on the account (settings saved best-effort, only
    when the app carries real settings) and returns the new token. Returns
    None when there are no stored credentials, a login backoff is active,
    or signin failed (backoff set for WAF/challenge + wrong-password).
    Raises the ban 403 when signin proves the account suspended, so the
    failover helper moves to the next account.
    """
    try:
        reason = item.login_backoff_skip()
    except Exception:
        reason = None
    if reason:
        logger.debug("qwen relogin backed off for %s: %s",
                     item.cfg.identifier, reason)
        return None
    email = (getattr(item.cfg, "email", "") or "").strip()
    password = getattr(item.cfg, "password", "") or ""
    if not email or not password:
        return None
    logger.info("qwen token expired/invalid for %s, re-signing in via direct API",
                item.cfg.identifier)
    try:
        new_token, expires_at, login_banned, detail = QA.login_with_expiry(
            email, password)
    except HTTPException:
        raise
    except _StreamCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            item.set_login_backoff("re-signin error", _RELOGIN_WAF_BACKOFF_SECONDS)
        except Exception:
            pass
        logger.warning("qwen re-signin error for %s: %s",
                       item.cfg.identifier, _engine.scrub_upstream_text(exc, 300))
        return None
    if login_banned:
        # Ground truth from signin: suspended/violation wording. Probe for a
        # precise mute_until, then mark + 403 for failover.
        try:
            _, ban_until = _probe_ban(app, new_token or item.cfg.token)
        except Exception:
            ban_until = None
        raise _ban_403_for_item(app, item, ban_until, detail or "")
    if not new_token:
        low = (detail or "").lower()
        try:
            if "waf" in low or "challenge" in low or "captcha" in low:
                item.set_login_backoff("waf-challenge", _RELOGIN_WAF_BACKOFF_SECONDS)
            elif "invalid_cred" in low or "incorrect" in low or "not found" in low:
                item.set_login_backoff("bad-credentials", _RELOGIN_CRED_BACKOFF_SECONDS)
        except Exception:
            pass
        # Scrubbed: ``detail`` is the signin RuntimeError text, i.e. Qwen's raw
        # /api/v1/auths/signin body, and this log line is written to disk (and
        # shipped to any log shipper). Same reasoning as DeepSeek's
        # ``login_note`` in ``_refresh_token_via_obscura``
        # (providers/deepseek/completion_service.py:388-394).
        logger.warning("qwen re-signin failed for %s: %s",
                       item.cfg.identifier,
                       _engine.scrub_upstream_text(detail, 200))
        return None
    try:
        item.cfg.token = new_token
        try:
            item.cfg.token_expires_at = float(expires_at or 0.0)
        except (TypeError, ValueError):
            item.cfg.token_expires_at = 0.0
        item.clear_login_backoff()
    except Exception:
        pass
    # Persist best-effort, but only when the app carries real settings:
    # _app_settings() falls back to load_settings() for embedded/test apps,
    # and saving that would write an unrelated config file.
    try:
        settings = getattr(getattr(app, "state", None), "settings", None)
        if settings is not None:
            settings.save()
    except Exception:
        pass
    logger.info("qwen token refreshed via direct re-signin for %s",
                item.cfg.identifier)
    return new_token


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
                            token: str | None, error: str, login_note: str | None,
                            stream_ban_until: float | None) -> None:
    # login_note is accepted for terminal-dispatch uniformity (DeepSeek's
    # browser note); Qwen has no browser refresh so it is ignored.
    # By terminal time a stored password has had one direct re-signin attempt
    # (pre-check + 401 recover); a ban masquerading as 401 is probed there too.
    # What reaches here is: no stored password, or re-signin failed.
    try:
        has_password = bool((getattr(item.cfg, "password", "") or "").strip())
    except Exception:
        has_password = False
    if has_password:
        hint = ("Automatic re-signin with the stored password failed — verify "
                "the password, or paste a fresh token. ")
    else:
        hint = ("No password stored — set one for automatic re-signin, or paste "
                "a fresh token. ")
    # Scrubbed, never the raw `error`: that is ``f"{code}: {upstream body}"``
    # (up to 2000 chars of Qwen's own response), the one body most likely to
    # echo the ``Authorization: Bearer`` header we sent -- and the Qwen
    # credential IS a JWT (chat.qwen.ai localStorage ``token``), whose dot-
    # framed segments are individually too short for a run-length redactor.
    # Same reason and same helper as DeepSeek's
    # ``_invalid_token_401_detail`` (providers/deepseek/completion_service.py:
    # 676-686); the codes and ban wording an operator diagnoses with survive.
    err = _engine.scrub_upstream_text(error, 200)
    raise HTTPException(
        status_code=401,
        detail=(f"token invalid for {item.cfg.identifier} ({err}). "
                + hint +
                "Paste a fresh localStorage.token from https://chat.qwen.ai "
                "(or the qwen_token/token cookies) and set-token the account. "
                "If the web UI shows suspension/violation, treat it as BANNED."))


def _terminal_ban(app: FastAPI, item: PooledAccount, prep: dict,
                  token: str | None, error: str, login_note: str | None,
                  stream_ban_until: float | None) -> None:
    # login_note ignored (dispatch uniformity); probe when the stream
    # carried no ban ts so the cooldown is precise, mirroring DeepSeek.
    ban_until: float | None = stream_ban_until
    if ban_until is None:
        try:
            banned, probed_until = _probe_ban(app, token)
        except Exception:
            banned, probed_until = False, None
        # The probe's BOOLEAN is honoured, not just its timestamp: a stray
        # (False, ts) must not be adopted as a ban cooldown. The fix landed in
        # DeepSeek only; this is its twin.
        ban_until = probed_until if banned else None
    raise _ban_403_for_item(app, item, ban_until, error)


#: Terminal dispatch table: one RetryKind (from the shared classifier) per
#: HTTP outcome. RATE_LIMIT/OVERLOAD/UPSTREAM_INVALID/UPSTREAM are the
#: label-only handlers every provider shares (core.engine); only
#: INVALID_TOKEN and BAN are Qwen-specific. WAF is a retryable kind that only
#: reaches here after exhausting the retry budget -> the generic upstream 502.
_TERMINAL_HANDLERS: dict[RetryKind, Callable[..., None]] = {
    **_engine.generic_terminal_handlers(RetryKind, "Qwen"),
    RetryKind.INVALID_TOKEN: _terminal_invalid_token,
    RetryKind.BAN: _terminal_ban,
    RetryKind.WAF: _engine.generic_terminal_handlers(
        RetryKind, "Qwen")[RetryKind.UPSTREAM],
}


def _raise_for_terminal_error(app: FastAPI, item: PooledAccount, prep: dict,
                              token: str | None, error: str, login_note: str | None,
                              stream_ban_until: float | None) -> None:
    """Map a terminal completion error event to its HTTP status. Always raises."""
    _engine.dispatch_terminal(classify_error, _TERMINAL_HANDLERS,
                              app, item, prep, token,
                              error, login_note, stream_ban_until)


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
    of one tool_calls JSON object, then ends in a clean [DONE]. WAF retries
    refresh the bridge and recreate the chat; INVALID_TOKEN retries probe for a
    ban, then re-sign in once via the cheap direct API when a password is
    stored, and recreate the chat with the fresh token. Buffered callers pass no
    callbacks and get identical behaviour. Retry dispatch is table-driven
    through ``core.engine.drive_retries`` with ``classify_error`` as the
    single source of truth. cancel_event, when set, aborts at the next SSE
    event.
    """
    settings = _app_settings(app)
    max_retries = max(0, int(getattr(settings, "max_retries", 1)))
    auto_delete = bool(getattr(settings, "auto_delete_session", True))
    token = _ensure_token(item)
    token_refreshed = False
    try:
        # Proactive: a known-expired token + stored password re-signs in
        # before burning chat creates on a dead session (cheap direct POST,
        # no browser/captcha). No password -> fall through; the 401 path +
        # cross-account failover handles it.
        if QA.token_expired(item.cfg):
            if ((getattr(item.cfg, "email", "") or "").strip()
                    and (getattr(item.cfg, "password", "") or "")):
                new = _try_relogin(app, item)
                if new:
                    token = new
                    token_refreshed = True
            else:
                logger.debug("qwen token expired for %s with no stored password; "
                             "proceeding to 401 + failover", item.cfg.identifier)
    except HTTPException:
        raise
    except _StreamCancelled:
        raise
    except Exception:
        pass
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
                        QC._log_waf_headers("prepare_chat_missing_token", hdrs)
                except Exception:
                    pass
                sid = QC.create_chat(hdrs, prep["model"])
            except _StreamCancelled:
                raise
            except Exception as exc:
                error = _error_text("", str(exc))
                try:
                    if classify_error(error) == RetryKind.WAF:
                        QC._log_waf_headers(
                            "prepare_chat_waf",
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

        def _recreate_chat() -> bool:
            """Rebind ``chat_id`` to a fresh chat, deleting the abandoned one.

            ``_prepare_chat`` overwrites ``chat_id`` in place, so without this
            the previous id is unreachable: the ``finally`` below deletes only
            the LAST one, and a WAF-challenged upstream strands one chat
            session per recovery pass (up to ``max_retries`` per request)
            that ``auto_delete_session`` can never reach again.

            Ordering mirrors DeepSeek's ``_recreate_and_attempt``
            (providers/deepseek/completion_service.py:1001-1043): the OLD id
            and the headers it was created with are captured BEFORE the create
            (a refreshed token would 401 the delete), the create assigns
            ``chat_id`` only on success (so a failed create leaves the old id
            for the ``finally`` to clean up, not a double delete), and the
            delete runs only once the replacement is established.
            """
            old_id, old_headers = chat_id, headers
            if not _prepare_chat():
                return False
            if old_id and old_id != chat_id:
                _delete_chat_bg(old_headers, old_id, enabled=auto_delete)
            return True

        def _recover_waf() -> None:
            """Refresh WAF cookies (slot released during the global wait)."""
            logger.info("possible WAF block, refreshing cookies (retries left %s)",
                        max_retries)
            try:
                QC._log_waf_headers("recover_waf_before", headers)
            except Exception:
                pass
            _refresh_waf_without_account_lock(app, item, cancel_event=cancel_event)
            if _cancelled():
                return
            if _recreate_chat():
                _do_attempt(headers, payload)

        def _recover_invalid_token() -> None:
            """401: ban-probe, then one direct re-signin, then recreate chat.

            A banned account can masquerade as 401, so probe before spending
            the single re-signin: a ban becomes 403 + cooldown (failover)
            instead of a misleading 401 loop. ``token_refreshed`` guards the
            re-signin to once per request (pre-check counts).
            """
            nonlocal token, token_refreshed
            try:
                banned, ban_until = _probe_ban(app, token)
            except Exception:
                banned, ban_until = False, None
            if banned:
                raise _ban_403_for_item(app, item, ban_until, error or "")
            # Single log for "no fresh token obtained this pass" (relogin
            # failed, or already refreshed): recreating the chat is the retry.
            new = None
            if not token_refreshed:
                try:
                    new = _try_relogin(app, item)
                except HTTPException:
                    raise
                except _StreamCancelled:
                    raise
                except Exception:
                    new = None
                if new:
                    token = new
                    token_refreshed = True
            if not new:
                logger.info("token invalid, recreating chat (retries left %s)",
                            max_retries)
            if _recreate_chat():
                _do_attempt(headers, payload)

        def _recover_overload() -> None:
            """Retry transient overload with fresh headers (no stale reuse)."""
            nonlocal headers
            # Scrubbed before the 80-char slice: a slice is not a redactor.
            # Truncating first can cut a dot-framed JWT mid-token, which is
            # exactly the shape ``core.engine._SEGMENT_FRAMED_RE`` exists to
            # catch, and this account's credential IS a JWT.
            logger.info("upstream overloaded (%s), retrying (retries left %s)",
                        _engine.scrub_upstream_text(error, 80), max_retries)
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
        # Every row goes through _retryable, which ANDs the partial-emit guard.
        def _retryable(kind) -> Callable[[str | None], bool]:
            return _engine.live_retry_predicate(
                classify_error, kind, _emit_tracker)

        drive_retries(
            max_retries,
            _cancelled,
            lambda: error,
            (
                ("waf", _retryable(RetryKind.WAF), _recover_waf),
                ("invalid_token", _retryable(RetryKind.INVALID_TOKEN),
                 _recover_invalid_token),
                ("overload", _retryable(RetryKind.OVERLOAD), _recover_overload),
            ),
        )
        if _cancelled():
            raise _StreamCancelled()
        if error:
            _raise_for_terminal_error(app, item, prep, token, error, None,
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


async def _stream_completion(app: FastAPI, pool: AccountPool, item: PooledAccount, prep: dict):
    """Async generator: buffered (default) or live deltas, tools parsed at end.

    Producer (account-holding thread) is Qwen-specific only in the failover
    entrypoint; queue/cancel/emit shape + SSE consumer are engine-owned
    (``stream_producer``), including the ``buffered=not live`` watchdog
    sizing ARCHITECTURE.md's cross-provider contract requires: in buffered
    mode the producer emits nothing until the whole reply lands, so the
    consumer watchdog must cover the WHOLE producer budget -- chat create +
    the reply + up to ``max_retries`` WAF/401 recovery passes, each of which
    can include a qwen-host solved-jar fetch and a direct re-signin. Live
    keeps the single-read budget because every delta resets the clock. Same
    call DeepSeek makes in its own ``_stream_completion``.
    """
    settings = _app_settings(app)
    live = str(getattr(settings, "stream_mode", "live")).lower() == "live"

    def _run_producer(live: bool, put, cancel_event) -> dict:
        # Single publisher: buffered and live differ only in callbacks.
        # Thinking always streams live (never tool JSON); content streams
        # live only when no tools requested, else buffered so a tool_calls
        # reply is not also emitted as user-visible text.
        if live:
            return _complete_with_failover_sync(
                app, pool, prep, item,
                on_content=(None if prep.get("tools") else lambda t: put("content", t)),
                on_thinking=lambda t: put("thinking", t),
                cancel_event=cancel_event,
            )
        # Ban failover lives in the helper (owns item's lock): a banned
        # account retries on the next unbanned one inside the same request.
        # Ban errors carry no content prefix, so nothing partial leaks.
        return _complete_with_failover_sync(
            app, pool, prep, item, cancel_event=cancel_event)

    async for frame in _engine.stream_producer(
            _run_producer, prep=prep, timeout=QC.DEFAULT_TIMEOUT, live=live,
            grace=_CONSUMER_GRACE_SECONDS,
            parse_fn=_parse_tool_calls_local,
            truncated_fn=looks_truncated_tool_attempt):
        yield frame
