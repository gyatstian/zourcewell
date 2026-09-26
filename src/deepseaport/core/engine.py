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
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager

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


def error_text(code: str | None, message: str | None) -> str:
    """Single "CODE: message" formatter (the string classify_error parses)."""
    code = (code or "").strip()
    message = (message or "").strip()
    return f"{code}: {message}" if code else message


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


def drain_attempt(stream_fn: Callable[[dict, dict], object],
                  headers: dict, payload: dict,
                  on_content=None, on_thinking=None,
                  cancel_event: threading.Event | None = None,
                  error_formatter: Callable[[str | None, str | None], str] | None = None,
                  ) -> tuple[str, str, str | None, int]:
    """Drain one SSE stream into (content, thinking, error, usage).

    ``stream_fn(headers, payload)`` yields events with ``.kind`` in
    {content, thinking, usage, error, finished}, ``.text`` / ``.code`` /
    ``.message`` / ``.ban_until``. Shared by every web2api provider; the only
    per-provider input is the stream function itself. Explicit ``gen.close()``
    so the HTTP connection returns to the pool instead of lingering through
    retries. Cancel is checked per event so a disconnected client frees the
    account slot promptly.
    """
    fmt = error_formatter or error_text
    content: list[str] = []
    think: list[str] = []
    error: str | None = None
    usage_total = 0
    ban_until: float | None = None
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
        try:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    return AttemptOutcome("".join(content), "".join(think), error,
                          usage_total, ban_until)


@contextmanager
def slot_releaser(item, cancel_event: threading.Event | None = None):
    """Release the per-account slot during global (browser/WAF) waits.

    Releases ``item.lock`` on entry (no-op when not held, e.g. unit tests
    calling the engine with a fresh account), reacquires on exit. The
    reacquire polls so a disconnect aborts promptly instead of blocking
    indefinitely under contention (raises ``_StreamCancelled``). Leaving
    the slot released on cancel is safe: the outer ``run_with_failover``
    release is ``RuntimeError``-safe.
    """
    try:
        from ..server_concurrency import _StreamCancelled as _Cancelled
    except Exception:
        class _Cancelled(Exception):  # type: ignore[no-redef]
            pass
    released = False
    try:
        item.lock.release()
        released = True
    except RuntimeError:
        released = False
    except Exception:
        released = False
    try:
        yield released
    finally:
        if released:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise _Cancelled()
                try:
                    if item.lock.acquire(timeout=0.5):
                        break
                except Exception:
                    break


def app_settings(app):
    """Return app settings, tolerating app=None and embedded test apps."""
    if app is None:
        from ..config import load_settings

        return load_settings()
    settings = getattr(getattr(app, "state", None), "settings", None)
    if settings is None:
        from ..config import load_settings

        settings = load_settings()
        try:
            app.state.settings = settings
        except Exception:
            pass
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
    from fastapi import HTTPException

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
    """
    from fastapi import HTTPException

    try:
        from ..accounts import AccountPool

        AccountPool.mark_banned(item, until=ban_until)
        try:
            app_settings(app).save()
        except Exception:
            pass
    except Exception:
        pass
    try:
        base = ban_message_fn(ban_until) if ban_message_fn is not None else ""
        ident = getattr(getattr(item, "cfg", None), "identifier", "?")
        if not base:
            detail = (error_text_ or "account banned")[:300]
            if "banned" not in detail.lower() and "muted" not in detail.lower():
                detail = f"Account {ident} banned. {detail}"
        else:
            detail = f"Account {ident} banned. {base}"
    except Exception:
        try:
            ident = getattr(getattr(item, "cfg", None), "identifier", "?")
        except Exception:
            ident = "?"
        detail = (error_text_ or f"Account {ident} banned")[:300]
    return HTTPException(status_code=403, detail=detail)


def is_ban_http_exception(exc: BaseException,
                          is_ban_text: Callable[[str], bool]) -> bool:
    """True when exc is the ban 403 raised by ``_run_completion_core``."""
    try:
        from fastapi import HTTPException

        return (isinstance(exc, HTTPException) and exc.status_code == 403
                and bool(is_ban_text(str(getattr(exc, "detail", "") or ""))))
    except Exception:
        return False


def switch_current_after_ban_failover(pool, settings, used, failed,
                                      reason: str = "unusable") -> None:
    """Move CURRENT to the working account after failover.

    Only fires when CURRENT pointed at the failed account: the request
    already proved it unusable (ban/auth cooldown), so future requests should
    prefer the healthy one first. Best-effort, never raises.
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
                settings.save()
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
    bound to app/prep/callbacks) and the ban-403 predicate; the engine owns
    the generic slot lifecycle via ``run_with_failover``: release exactly
    once per attempt, acquire the next account before retrying, re-raise when
    no healthy account remains. ``allow_failover=False`` surfaces the first
    error directly; None reads ``use_multiple_accounts`` from settings.
    Bans and 401 auth failures are account-specific; other failures belong to
    the caller/current request.
    """
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
            from fastapi import HTTPException

            return (isinstance(exc, HTTPException)
                    and (bool(is_ban_exc(exc)) or exc.status_code == 401))
        except Exception:
            return False

    def _on_before_retry(exc, account, tries: int, max_tries: int) -> None:
        nonlocal first_exc, failed_item
        if first_exc is None:
            first_exc, failed_item = exc, account
        try:
            from fastapi import HTTPException

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

    try:
        from ..accounts import AccountPool as _Pool
        _release = _Pool.release
    except Exception:
        _release = None

    result = run_with_failover(
        pool,
        first_item,
        _run_once,
        _is_retryable,
        on_before_retry=_on_before_retry,
        acquire_timeout=_FAILOVER_TIMEOUT,
        is_cancelled=lambda: cancel_event is not None and cancel_event.is_set(),
        allow_failover=allow_failover,
        release=_release,
        acquire=lambda timeout: pool.acquire(timeout=timeout),
    )
    if failed_item is not None and used_item is not None:
        try:
            reason = "banned" if bool(is_ban_exc(first_exc)) else "has no usable token"
        except Exception:
            reason = "unusable"
        try:
            switch_current_after_ban_failover(
                pool, _get_settings(app), used_item, failed_item, reason=reason)
        except Exception:
            pass
    return result  # type: ignore[return-value]


class LiveEmitTracker:
    """Flag partial live emits so overload retries skip after output left.

    Live mode already ``_put`` partial deltas to the client; a retry after a
    partial emit would re-stream from 0 (duplicated/interleaved text and
    corrupt tool-call JSON). Wrap each live callback once; the overload
    retry predicate checks ``tracker.emitted``. Shared by every provider.
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
    from fastapi import HTTPException

    raise HTTPException(status_code=429,
                        detail=f"{provider_label} rate limited: {(error or '')[:200]}. "
                               "Wait 2-20 min, space requests, avoid parallel loops.")


def raise_overload(provider_label: str, error: str | None) -> None:
    """Transient upstream overload: 503 (retryable), never a 500. Always raises."""
    from fastapi import HTTPException

    raise HTTPException(status_code=503,
                        detail=f"{provider_label} server busy: {(error or '')[:200]}. "
                               "Retry shortly (transient overload).")


def raise_upstream_invalid(provider_label: str, prep: dict, error: str | None) -> None:
    """Upstream payload rejection (422/400 family): 502. Always raises."""
    from fastapi import HTTPException

    logger.warning("upstream rejected completion payload (%s); "
                   "prompt_len=%d model=%s",
                   (error or "")[:500], len(str((prep or {}).get("prompt", ""))),
                   (prep or {}).get("model"))
    raise HTTPException(
        status_code=502,
        detail=f"{provider_label} rejected the request as invalid ({(error or '')[:800]}). "
               "The prompt/payload failed upstream validation, not a ban or "
               "rate limit. Retry with a non-empty prompt; if it persists, "
               "the upstream API schema likely changed.")


def raise_upstream(provider_label: str, error: str | None) -> None:
    """Generic upstream error: 502. Always raises."""
    from fastapi import HTTPException

    logger.warning("upstream completion error: %s", (error or "")[:500])
    raise HTTPException(
        status_code=502,
        detail=f"{provider_label} upstream error: {(error or '')[:800]}")


def empty_response_error(provider_label: str):
    """Upstream closed without content/thinking/error: 502, never silent 200."""
    from fastapi import HTTPException

    return HTTPException(status_code=502,
                         detail=f"empty upstream response from {provider_label} "
                                "(no content/thinking/error events; often "
                                "throttling — wait then retry; if it persists "
                                "for hours, the upstream stream format likely "
                                "changed — update deepseaport)")


def dispatch_terminal(classify_fn: Callable[[str | None], object],
                      handlers: dict, fallback: Callable[..., None],
                      app, item, prep: dict, token: str | None, error: str,
                      *extra) -> None:
    """Map a terminal completion error to its HTTP status. Always raises.

    Single dispatch: ``classify_fn`` (the provider's ``classify_error``)
    produces one kind, then the handler for that kind raises. No string
    matching lives at call sites.
    """
    kind = classify_fn(error)
    (handlers.get(kind, fallback))(app, item, prep, token, error, *extra)


# --- Stream bridge (producer/consumer shared by every provider) -------------


def stream_chunk(cid: str, created: int, model: str,
                 delta: dict, finish: str | None = None) -> bytes:
    """One OpenAI-style ``chat.completion.chunk`` SSE frame."""
    return ("data: " + json.dumps(
        {"id": cid, "object": "chat.completion.chunk", "created": created,
         "model": model, "choices": [{"index": 0, "delta": delta,
                                      "finish_reason": finish}]},
        ensure_ascii=False) + "\n\n").encode("utf-8")


def tool_finish_chunks(chunk_fn: Callable[..., bytes], calls,
                       truncated: bool = False):
    """Tool-call deltas + finish chunk; ``length`` when a call was cut off."""
    if calls:
        for call in calls:
            yield chunk_fn({"tool_calls": [{
                "id": call["id"], "type": "function",
                "function": {"name": call["function"]["name"],
                             "arguments": call["function"]["arguments"]}}]})
        yield chunk_fn({}, "tool_calls")
    else:
        yield chunk_fn({}, "length" if truncated else "stop")


def usage_chunk(cid: str, created: int, model: str, prep: dict,
                content: str, thinking: str,
                usage_total: int | None = None) -> bytes:
    """Final OpenAI-style usage chunk (stream_options.include_usage).

    Emitted with empty choices before [DONE] so cost-tracking harnesses see
    the same prompt/completion totals as the non-stream response. Uses the
    shared ``core.openai.build_usage`` estimate (upstream never reports
    prompt tokens); ``usage_total`` wins for completion tokens when seen.
    """
    try:
        from .openai import build_usage as _build_usage
        usage = _build_usage(prep, content or "", thinking or "", usage_total)
    except Exception:
        try:
            pt = max(1, len(str((prep or {}).get("prompt", "") or "")) // 4)
            ct = max(1, (len(content or "") + len(thinking or "")) // 4)
        except Exception:
            pt, ct = 1, 1
        usage = {"prompt_tokens": pt, "completion_tokens": ct,
                 "total_tokens": pt + ct}
    return ("data: " + json.dumps(
        {"id": cid, "object": "chat.completion.chunk", "created": created,
         "model": model, "choices": [], "usage": usage},
        ensure_ascii=False) + "\n\n").encode("utf-8")


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
    """
    _real_parse = None
    _real_ex = None
    try:
        from .tools import parse_tool_calls as _rp
        from .tools.parser import parse_tool_calls_ex as _re
        _real_parse, _real_ex = _rp, _re
    except Exception:
        pass
    if parse_fn is None:
        parse_fn = _real_parse
        if parse_fn is None:
            from .tools import parse_tool_calls as parse_fn  # type: ignore[no-redef]
    if truncated_fn is None:
        from .tools import looks_truncated_tool_attempt as truncated_fn  # type: ignore[no-redef]
    limit = prep.get("tool_args_max_chars")
    forgiving = bool(prep.get("forgiving_toolcalls", False))
    calls: list[dict] | None = None
    remaining: str = content
    dropped: list[int] | None = None
    diagnostics: list[dict] | None = None
    if parse_fn is _real_parse and _real_ex is not None:
        kwargs: dict = {"forgiving": forgiving}
        if limit:
            kwargs["max_args_chars"] = limit
        try:
            res = await asyncio.to_thread(
                _real_ex, content, prep["tools"], **kwargs)
            calls, remaining = res.calls, res.rest
            dropped, diagnostics = res.dropped, res.diagnostics
        except Exception:
            if limit:
                calls, remaining = await asyncio.to_thread(
                    parse_fn, content, prep["tools"],
                    max_args_chars=limit, forgiving=forgiving)
            else:
                calls, remaining = await asyncio.to_thread(
                    parse_fn, content, prep["tools"], forgiving=forgiving)
    else:
        if limit:
            calls, remaining = await asyncio.to_thread(
                parse_fn, content, prep["tools"],
                max_args_chars=limit, forgiving=forgiving)
        else:
            calls, remaining = await asyncio.to_thread(
                parse_fn, content, prep["tools"], forgiving=forgiving)
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
            # Rich miss log (same helper shape as core.openai): diagnostics
            # say unknown-name vs oversize vs format drift.
            try:
                if diagnostics is None and _real_ex is not None:
                    from .tools import _valid_names as _vn
                    kwargs2: dict = {"forgiving": forgiving}
                    if limit:
                        kwargs2["max_args_chars"] = limit
                    res2 = await asyncio.to_thread(
                        _real_ex, content or "", prep.get("tools") or [],
                        **kwargs2)
                    diagnostics = res2.diagnostics
                    try:
                        valid = _vn(prep.get("tools"))
                    except Exception:
                        valid = None
                    parts = []
                    for d in diagnostics or []:
                        try:
                            parts.append(
                                f"{d.get('format')}:matched="
                                f"{int(bool(d.get('matched')))} "
                                f"kept={d.get('kept', 0)} "
                                f"dropped={list(d.get('dropped') or [])} "
                                f"raw={d.get('raw', 0)}")
                        except Exception:
                            continue
                    trail = "; ".join(parts) if parts else "no formats matched"
                    if valid:
                        trail += f" valid={sorted(valid)}"
                else:
                    trail = None
                    if diagnostics is not None:
                        try:
                            from .tools import _valid_names as _vn2
                            valid2 = _vn2(prep.get("tools"))
                        except Exception:
                            valid2 = None
                        parts2 = []
                        for d in diagnostics or []:
                            try:
                                parts2.append(
                                    f"{d.get('format')}:matched="
                                    f"{int(bool(d.get('matched')))} "
                                    f"kept={d.get('kept', 0)} "
                                    f"dropped={list(d.get('dropped') or [])} "
                                    f"raw={d.get('raw', 0)}")
                            except Exception:
                                continue
                        trail = ("; ".join(parts2) if parts2
                                 else "no formats matched")
                        if valid2:
                            trail += f" valid={sorted(valid2)}"
                if trail:
                    logger.warning(
                        "stream tool-call miss model=%s forgiving=%s %s "
                        "preview=%.300s",
                        prep.get("model"), forgiving, trail,
                        (content or "")[:300])
                else:
                    logger.debug("stream no tool call parsed model=%s preview=%.200s",
                                 prep.get("model"), (content or "")[:200])
            except Exception:
                logger.debug("stream no tool call parsed model=%s preview=%.200s",
                             prep.get("model"), (content or "")[:200])
    elif dropped:
        logger.warning("stream tool-call partial drop model=%s dropped=%s",
                       prep.get("model"), list(dropped))
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
    """
    chunk = lambda delta, finish=None: stream_chunk(cid, created, model, delta, finish)  # noqa: E731
    yield chunk({"role": "assistant"})
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
                    try:
                        _ut = payload.get("usage_total")
                    except Exception:
                        _ut = None
                    yield usage_chunk(cid, created, model, prep,
                                      content, thinking, _ut)
            elif prep.get("tools"):
                calls, remaining, truncated = await parse_stream_calls(
                    prep, payload["content"], parse_fn, truncated_fn)
                if remaining:
                    yield chunk({"content": remaining})
                for piece in tool_finish_chunks(chunk, calls, truncated):
                    yield piece
                if wants_usage(prep):
                    try:
                        _ut = payload.get("usage_total")
                        _th = payload.get("thinking", "")
                        _co = payload.get("content", "")
                    except Exception:
                        _ut, _th, _co = None, "", ""
                    yield usage_chunk(cid, created, model, prep, _co, _th, _ut)
            else:
                yield chunk({}, "stop")
                if wants_usage(prep):
                    try:
                        _ut = payload.get("usage_total")
                        _th = payload.get("thinking", "")
                        _co = payload.get("content", "")
                    except Exception:
                        _ut, _th, _co = None, "", ""
                    yield usage_chunk(cid, created, model, prep, _co, _th, _ut)
            yield b"data: [DONE]\n\n"
            return
        if kind == "http_error":
            err = payload
            yield ("data: " + json.dumps({"error": {"message": err.detail,
                                                    "code": err.status_code}},
                                         ensure_ascii=False) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            return
        yield ("data: " + json.dumps({"error": {"message": str(payload)}},
                                     ensure_ascii=False) + "\n\n").encode()
        yield b"data: [DONE]\n\n"
        return


def new_stream_context(prep: dict, stream_timeout: float,
                       consumer_grace: float = 10.0):
    """Allocate one stream's (cid, created, queue, cancel_event, timeouts).

    Thin constructor so providers share the exact queue/cancel/timeout shape:
    the consumer wait outlasts the blocking stream timeout so the async
    reader never times out while the producer is still on a valid long reply.
    """
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    out: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    consumer_timeout = float(stream_timeout) + float(consumer_grace)
    return cid, created, out, cancel_event, consumer_timeout


def emit_to_loop(loop: asyncio.AbstractEventLoop, out: asyncio.Queue, item: tuple) -> None:
    """Thread-safe queue put; swallows RuntimeError after loop close."""
    try:
        loop.call_soon_threadsafe(out.put_nowait, item)
    except RuntimeError:
        pass
