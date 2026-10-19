"""FastAPI route wiring for deepseaport.server.

Pure move of the handlers that were previously defined inside
``deepseaport.server.create_app``, plus ``_check_auth``/``_pool``/``_prepare``.
``create_app`` stays in deepseaport.server as thin wiring that calls
``register_routes(app)``.

Handlers look up the completion/stream entry points through a deferred
``_server()`` reference so tests that monkeypatch them on
``deepseaport.server`` keep working.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import threading
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .accounts import AccountPool, provider_label, usable_provider as _usable_provider
from .config import AccountConfig, Settings
from .core import pools
from .core.engine import scrub_upstream_text
from .core.obscura_bridge import ObscuraBridge
from .server_concurrency import (
    _StreamCancelled,
    _acquire_slot_or_499,
    _watch_disconnect,
)
from .core.tools import render_prompt, tool_reminder_prompt, tool_system_prompt

# Body-size ceiling for every route. Enforced from Content-Length AND from the
# bytes that actually arrive, so a chunked request cannot skip it (see
# install_body_limit below). 8 MB is far above any real chat request (a large
# agentic tool result is ~1 MB of JSON) and still small enough that a single
# client cannot pin a worker on a JSON parse.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

# Tool schemas are serialized verbatim into the system prompt, so the tool
# count bounds prompt size and upstream context cost. 128 covers a big MCP
# server (Claude Code ships ~15, pi/opencode harnesses up to ~60) with room
# to spare; the old uncapped list let a client inflate the prompt arbitrarily
# at the cost of a full re-render per request.
MAX_TOOLS_PER_REQUEST = 128

# POST /v1/accounts caps. Each accepted call appends to settings.accounts and
# rewrites the entire config file (Settings.save: mkstemp + fsync +
# os.replace), so an unbounded endpoint is a disk-write amplifier sitting on
# the credential store. Limits are generous for real credentials: identifiers
# are emails/phone numbers, secrets are provider tokens.
MAX_ACCOUNTS = 64
MAX_IDENTIFIER_CHARS = 512
MAX_SECRET_CHARS = 4096

# Same shape rule as cli._valid_provider_id: provider ids are lowercase
# alnum/dash, so anything else is a typo or junk that no pool can ever match.
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9-]+$")

# A path/body segment this long and this opaque cannot be an account
# identifier: AccountConfig.identifier is an email, a mobile number, or
# token[:10] + "..." (config.AccountConfig.identifier), all far shorter. 32+
# chars of [A-Za-z0-9_.=-] is a full userToken (64 chars), a Qwen JWT (three
# dot-separated base64url segments, '=' padding included), or junk. Rejected at
# the HTTP edge so a bearer token never reaches a URL (access logs, proxy logs,
# Referer chains) or a log line, while accounts._matches keeps full-token
# matching for the CLI/local callers that need it.
_TOKEN_SHAPED_RE = re.compile(r"^[A-Za-z0-9_.=-]{32,}$")

# Cap on what a rejected request may put in the log, on top of redaction.
_MAX_LOGGED_PATH_CHARS = 200


def _server():
    """Late-bound deepseaport.server module for monkeypatch-compatible lookups."""
    from . import server as _s
    return _s


logger = logging.getLogger("deepseaport.server")


class _BodyOverLimit(HTTPException):
    """413 raised from the receive wrapper once the running total crosses the cap.

    Deliberately an ``HTTPException``: FastAPI wraps the whole request-body
    parse in ``except HTTPException: raise`` and relabels anything else as a
    400 "There was an error parsing the body", so an abort signal has to BE an
    HTTPException to reach the client as the 413 it is.
    """

    def __init__(self, seen: int) -> None:
        super().__init__(status_code=413, detail=_too_large_detail(seen))
        self.seen = seen


def _too_large_detail(length: int) -> str:
    """One detail string for both 413 paths (declared length or bytes seen)."""
    return (f"request body too large: {length} bytes exceeds the "
            f"{MAX_REQUEST_BYTES} byte limit")


def _too_large(length: int) -> JSONResponse:
    return JSONResponse(status_code=413,
                        content=error_envelope(413, _too_large_detail(length)))


def _redact_path(path: str) -> str:
    """Path with every token-shaped segment replaced and the rest bounded.

    The raw path is never logged: accounts._matches accepts a full bearer token
    as an identifier, so ``DELETE /v1/accounts/<token>`` would otherwise write
    a live credential into the log before routing and before _check_auth.
    """
    parts = [("<redacted>" if _TOKEN_SHAPED_RE.match(seg) else seg[:64])
             for seg in str(path or "").split("/")]
    out = "/".join(parts)
    if len(out) > _MAX_LOGGED_PATH_CHARS:
        return out[:_MAX_LOGGED_PATH_CHARS] + "..."
    return out


def _log_target(scope) -> str:
    """What to log instead of the raw path: route template, else redacted path.

    ``scope["route"]`` is set by the router (FastAPI's APIRoute.matches), so it
    is available when a streamed body is aborted mid-parse - by then the route
    is matched - and absent on the Content-Length fast path, which runs before
    routing.
    """
    method = str(scope.get("method") or "?")
    template = getattr(scope.get("route"), "path", None)
    if template:
        return f"{method} {template}"
    return f"{method} {_redact_path(scope.get('path') or '')}"


def _log_oversized(scope, length: int) -> None:
    logger.warning("rejecting %s: body %d bytes > %d",
                   _log_target(scope), length, MAX_REQUEST_BYTES)


def _declared_length(scope) -> int | None:
    """Content-Length as an int, or None when absent or unparseable."""
    for key, value in scope.get("headers") or ():
        if key == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


class BodyLimitMiddleware:
    """413 above MAX_REQUEST_BYTES, with or without a declared Content-Length.

    Two layers, because a chunked request declares nothing:

    * Content-Length fast path - reject before the body is read at all, so an
      oversized declared body is never materialized or JSON-parsed. The body
      stream is handed to the app untouched, so every normal request takes zero
      extra hops.
    * Receive wrapper - with ``Transfer-Encoding: chunked`` there is no
      Content-Length to check, and FastAPI materializes + parses the whole body
      to bind ``body: dict`` before _check_auth runs, so one unauthenticated
      client could force an unbounded parse. The wrapper only COUNTS the bytes
      the app pulls (it never buffers them and never touches
      ``http.disconnect``), and aborts the read with ``_BodyOverLimit`` on the
      first chunk that crosses the cap, so the parse stops there. The rest of
      the body is deliberately not drained: the 413 is on the wire before the
      client finishes sending, and the server reaps the connection through its
      normal keep-alive teardown.

    Pure ASGI (not ``@app.middleware("http")``) because only the raw receive
    callable can be wrapped this way, and because starlette's
    ``BaseHTTPMiddleware`` reaches the server's receive through private,
    version-specific plumbing.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None and declared > MAX_REQUEST_BYTES:
            _log_oversized(scope, declared)
            await _too_large(declared)(scope, receive, send)
            return
        if declared is not None:
            # Declared and within the cap: byte-exact, nothing to wrap.
            await self.app(scope, receive, send)
            return
        seen = 0
        started = False

        async def _send(message) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        async def _counting_receive() -> dict:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body") or b"")
                if seen > MAX_REQUEST_BYTES:
                    # Logged at the point of detection, which is the only place
                    # that sees both the running total and the already-matched
                    # route template: starlette's exception middleware answers
                    # the HTTPException itself, so the abort never reaches the
                    # handler below.
                    _log_oversized(scope, seen)
                    raise _BodyOverLimit(seen)
            return message

        try:
            await self.app(scope, _counting_receive, _send)
        except _BodyOverLimit as exc:
            if started:
                # A response is already on the wire (never happens for the
                # parse-then-reject paths, but a second one would corrupt the
                # stream): surface it instead of appending to the response.
                raise
            # Fallback for an app that does not turn HTTPException into a
            # response itself. FastAPI's own exception middleware gets there
            # first and produces the identical body.
            await _too_large(exc.seen)(scope, receive, send)


def install_body_limit(app: FastAPI) -> None:
    """Install the MAX_REQUEST_BYTES ceiling on ``app`` (idempotent per app).

    Module-level so the dry-run app gets it too: create_dry_app builds its own
    FastAPI and never calls register_routes, which is how it ended up echoing
    and printing an unbounded body to the terminal.
    """
    if getattr(app.state, "deepseaport_body_limit", False):
        return
    app.state.deepseaport_body_limit = True
    app.add_middleware(BodyLimitMiddleware)


# OpenAI error-type vocabulary (api.openai.com slugs). Only the statuses this
# surface can actually raise are listed; everything else -- every 5xx -- is
# api_error, which is also the bucket OpenAI uses for "the server broke".
_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error", 409: "invalid_request_error",
    413: "invalid_request_error", 422: "invalid_request_error",
    401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 429: "rate_limit_error",
}


def _error_message(detail) -> str:
    """``detail`` as the string ``error.message`` must stay (never null).

    Same rule as the SSE error frame (core.engine.sse_error_frame): strict
    OpenAI SDKs type ``error.message`` as a string, and FastAPI's own 422
    detail is a LIST of objects, so the raw value would land as a nested JSON
    array. Non-str details are JSON-encoded rather than ``str()``-ed, which
    would print a Python repr.
    """
    if isinstance(detail, str):
        return detail
    if detail is None:
        return ""
    try:
        return json.dumps(detail, ensure_ascii=False, default=str)
    except Exception:
        return str(detail)


def error_envelope(status_code: int, detail) -> dict:
    """OpenAI-shaped error body: the pinned ``detail`` PLUS an ``error`` object.

    Additive by contract: ``detail`` keeps the exact value every existing
    client and test reads, while ``error`` ({message, type, param, code}) is
    what OpenAI SDKs and LiteLLM-style retry classifiers consume. ``param``
    and ``code`` stay null: this bridge has no per-field or error-code
    namespace of its own, and a fabricated value is worse than an absent one.

    ``_too_large`` (the middleware-side 413) builds the same body through this
    function: the body-limit middleware sits OUTSIDE the exception handlers, so
    it cannot rely on them.
    """
    return {"detail": detail,
            "error": {"message": _error_message(detail),
                      "type": _ERROR_TYPE_BY_STATUS.get(int(status_code), "api_error"),
                      "param": None, "code": None}}


async def _http_exception_response(request: Request,
                                   exc: StarletteHTTPException) -> Response:
    """HTTPException -> envelope, with the status and headers preserved.

    Registered on Starlette's ``HTTPException`` so it also covers FastAPI's
    subclass (the one every route raises) and the router's own 404/405.
    ``exc.headers`` must survive: Retry-After and friends are already attached
    by the raiser, and dropping them silently un-does that decision.
    """
    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        # 204/304/1xx: FastAPI's own handler sends no body here, and a JSON
        # body would be a protocol error.
        return Response(status_code=exc.status_code, headers=headers)
    return JSONResponse(status_code=exc.status_code,
                        content=error_envelope(exc.status_code, exc.detail),
                        headers=headers)


async def _validation_error_response(request: Request,
                                     exc: RequestValidationError) -> JSONResponse:
    """FastAPI's own 422 detail (byte-identical), plus the envelope."""
    return JSONResponse(
        status_code=422,
        content=error_envelope(422, jsonable_encoder(exc.errors())))


async def _unhandled_error_response(request: Request, exc: Exception) -> JSONResponse:
    """500 envelope for an unhandled exception; its text is never echoed.

    Deliberately does not log: Starlette's ServerErrorMiddleware re-raises after
    the response is on the wire, so uvicorn still prints the traceback exactly
    as before. Adding an ``exc_info`` line here would be a second, unscrubbed
    copy of a message that can carry an upstream body (core scrub contract).
    """
    return JSONResponse(status_code=500,
                        content=error_envelope(500, "internal server error"))


def install_error_envelope(app: FastAPI) -> None:
    """Install the JSON error envelope on ``app`` (idempotent per app).

    Every error response keeps its existing ``detail`` byte-for-byte and gains
    the OpenAI-shaped ``error`` object: FastAPI's HTTPException (every 4xx this
    surface raises, plus the router's own 404/405), RequestValidationError
    (422), and the unhandled-exception 500. Module-level for the same reason as
    install_body_limit: create_dry_app builds its own FastAPI and never calls
    register_routes.
    """
    if getattr(app.state, "deepseaport_error_envelope", False):
        return
    app.state.deepseaport_error_envelope = True
    app.add_exception_handler(StarletteHTTPException, _http_exception_response)
    app.add_exception_handler(RequestValidationError, _validation_error_response)
    app.add_exception_handler(Exception, _unhandled_error_response)


def register_routes(app: FastAPI) -> None:
    install_body_limit(app)
    install_error_envelope(app)

    @app.get("/health")
    async def health():
        # Liveness stays 200/ok:true so existing pollers (_wait_for_server,
        # simple uptime checks) keep working. Readiness details ride along so
        # an orchestrator can avoid routing to a live-but-dead instance
        # (empty pool / no tokens) without us 503ing liveness (which would
        # crash-loop a process that just needs config, not a restart).
        # No pool creation / settings.save here: health polls every few
        # seconds and must stay side-effect free.
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        try:
            settings = getattr(app.state, "settings", None)
            accs = list(getattr(settings, "accounts", []) or [])
        except Exception:
            accs = []
        try:
            total = len(accs)
            with_token = sum(1 for a in accs if (getattr(a, "token", "") or "").strip())
            banned = sum(1 for a in accs if getattr(a, "banned", False))
        except Exception:
            total = with_token = banned = 0
        # Per-provider readiness (additive): a multi-provider orchestrator needs
        # to know WHICH pool is empty, not just the total. Counted from
        # settings.accounts alone so /health stays pool-free and side-effect
        # free; the provider default mirrors core.pools._provider_of.
        by_provider: dict = {}
        try:
            for a in accs:
                pid = (getattr(a, "provider", "") or "deepseek").strip().lower()
                row = by_provider.setdefault(
                    pid, {"accounts": 0, "accounts_with_token": 0, "banned": 0})
                row["accounts"] += 1
                if (getattr(a, "token", "") or "").strip():
                    row["accounts_with_token"] += 1
                if getattr(a, "banned", False):
                    row["banned"] += 1
        except Exception:
            by_provider = {}
        try:
            pool = getattr(app.state, "pool", None)
            if pool is None:
                # Read the shared registry without building anything (health
                # must stay side-effect free); fall back to account count.
                cache = getattr(app.state, "pools", None) or {}
                try:
                    selected = pools.selected_provider(settings)
                except Exception:
                    selected = "deepseek"
                pool = cache.get(selected)
            pool_size = len(pool) if pool is not None else total
            current = getattr(pool, "current", None)
            if current is None:
                try:
                    current = (getattr(settings, "active_account", "") or "").strip() or None
                except Exception:
                    current = None
        except Exception:
            pool_size = total
            current = None
        waf = bool(bridge.has_waf_token()) if bridge else False
        ready = bool(total > 0 and with_token > 0 and with_token > banned)
        # `current` is a boolean, not the identifier it used to be:
        # pool.current / settings.active_account carry AccountConfig.identifier
        # (an email, a phone number, or token[:10] + "...") and this route is
        # deliberately unauthenticated (server_runner._wait_for_server and
        # uptime checks only read the status code). Pollers need presence, not
        # identity; /v1/accounts serves the identity behind the API key.
        return {"ok": True, "ready": ready, "waf": waf, "version": __version__,
                "accounts": total, "accounts_with_token": with_token,
                "banned": banned, "pool": pool_size, "current": bool(current),
                "providers": by_provider}

    @app.get("/v1/waf/status")
    async def waf_status(authorization: str = Header(default="")):
        # Auth-gated: the payload names the absolute obscura binary path, the
        # absolute profile path and the WAF cookie names, which is exactly the
        # reconnaissance a LAN scanner wants (filesystem layout + which
        # session cookies unlock the upstream). Nothing polls this route
        # (_wait_for_server uses /health), so gating it costs no readiness.
        _check_auth(app.state.settings, authorization)
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        if bridge is None:
            return {"warmed": False, "error": "bridge not initialised yet"}
        return await asyncio.to_thread(bridge.status)

    @app.get("/v1/models")
    async def list_models(authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        # Same key gate as every other /v1/* route (only /health stays open):
        # the listing without it was the one route a LAN client could read
        # unauthenticated. Empty keys keep the documented local-dev default.
        _check_auth(settings, authorization)
        # openapi_url is None, so before this the running server exposed no
        # version at all: a user genuinely could not ask what they were talking
        # to. Same single source as server.py/dry_run.py's FastAPI(version=...).
        # Off the event loop: the opencode catalogue read is a blocking socket
        # (2s live, up to 30s while a managed serve boots), and a cold-cache
        # request here froze /health and every disconnect watcher for the read.
        return {"object": "list", "version": __version__,
                "data": await asyncio.to_thread(build_models_list, settings)}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request,
                               authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        # Still raises 400/404 fast, before touching accounts; off the event
        # loop because the render (and, for opencode, the catalogue read) is
        # blocking work -- inline it froze /health and every disconnect watcher
        # for the whole read.
        prep = await asyncio.to_thread(_prepare, body, settings)
        from .core import registry as _registry
        provider = _registry.get(prep.get("provider")) or _registry.default()
        # Local providers (opencode CLI) need no account slot: skip the pool
        # entirely and pass item=None (their engine ignores pool/item).
        needs_slot = bool(getattr(provider, "requires_accounts", True))
        pool = _pool_for(app, prep.get("provider")) if needs_slot else None
        # Async poll (no executor thread held): blocking acquire would
        # occupy default-executor threads and starve other to_thread work.
        # use_multiple_accounts=False pins to CURRENT (queue on its lock);
        # True (default) fails over to another healthy account when
        # CURRENT is busy (per-account lock) or cooling down (banned).
        allow_multi = bool(getattr(settings, "use_multiple_accounts", True))
        if prep["stream"]:
            # Same queued-acquire cancel as non-stream: a client that gives up
            # while queued must surface 499 without starting orphaned work
            # once a slot frees (previously parked full 90s here).
            cancel_event = threading.Event()
            watcher = asyncio.create_task(_watch_disconnect(request, cancel_event))
            try:
                if needs_slot:
                    item = await _acquire_slot_or_499(pool, request, cancel_event, 90, allow_multi)
                    if cancel_event.is_set():
                        AccountPool.release(item)
                        raise HTTPException(status_code=499, detail="client disconnected")
                else:
                    item = None
                    if cancel_event.is_set():
                        raise HTTPException(status_code=499, detail="client disconnected")
            finally:
                watcher.cancel()
            return StreamingResponse(
                provider.stream_response(app, pool, item, prep),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        # Failover helper owns item's lock (releases each attempt exactly
        # once): a banned CURRENT transparently retries on the next unbanned
        # account instead of surfacing 403 for a usable pool.
        # Non-stream disconnect: poll the client socket into cancel_event so
        # a timed-out/cancelled client does not hold the slot for the full
        # session→PoW→stream (next requests would queue to 90s/429). The
        # watcher starts BEFORE acquire: a client that gives up while queued
        # must not then run a full orphaned session once a slot frees.
        cancel_event = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(request, cancel_event))
        try:
            # Chunked acquire: a disconnect while queued surfaces as 499
            # within ~2s (instead of running an orphaned session once freed).
            # Local (accountless) providers skip acquisition (item=None).
            if needs_slot:
                item = await _acquire_slot_or_499(pool, request, cancel_event, 90, allow_multi)
                if cancel_event.is_set():
                    AccountPool.release(item)
                    raise HTTPException(status_code=499, detail="client disconnected")
            else:
                item = None
                if cancel_event.is_set():
                    raise HTTPException(status_code=499, detail="client disconnected")
            try:
                result = await asyncio.to_thread(
                    provider.complete_request, app, pool, prep, item, cancel_event)
            except _StreamCancelled:
                raise HTTPException(status_code=499, detail="client disconnected")
            except HTTPException:
                # A decision already made upstream (404 model, 429 no free
                # account, 503 no accounts). Keep its status; only the
                # catch-all below rewrites anything.
                raise
            except Exception as exc:
                # A provider client's own RuntimeError ("create_session: no id
                # (HTTP 500): <raw body>", "pow challenge: bad shape: <raw
                # body>") used to escape the route entirely: the client got a
                # 500 and uvicorn wrote the traceback -- raw upstream body
                # included -- to the server log. The engine already maps every
                # upstream failure it recognises onto a scrubbed 502
                # (``core.engine.raise_upstream``); this is the same shape for
                # the failures that never reached it. Scrubbed before the
                # truncation, per the core contract, and the log line is scrubbed
                # too, so no unredacted upstream text reaches either.
                logger.warning("unhandled %s from %s: %s", type(exc).__name__,
                               getattr(provider, "id", "?"),
                               scrub_upstream_text(exc, 500))
                raise HTTPException(status_code=502, detail=(
                    f"{provider_label(getattr(provider, 'id', ''))} upstream "
                    f"error: {scrub_upstream_text(exc, 800)}")) from exc
        finally:
            watcher.cancel()
        return _server()._openai_response(prep, result)

    @app.get("/v1/accounts")
    async def list_accounts(provider: str = "", authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pid = (provider or "").strip()
        pool = _pool_for(app, pid)
        data = await asyncio.to_thread(pool.status)
        return {"object": "list", "current": pool.current or None, "data": data}

    @app.post("/v1/accounts/select")
    async def select_account(body: dict | None = None,
                             authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        ident = ((body or {}).get("identifier") or "").strip() if isinstance(body, dict) else ""
        # Ahead of _check_auth (see _reject_token_shaped_identifier): a token
        # never reaches a code path that could echo, log or persist it.
        _reject_token_shaped_identifier(ident)
        _check_auth(settings, authorization)
        pool = _server()._pool(app)
        if not ident:
            # Only persist an actual change. An empty body used to assign ""
            # and run a full Settings.save (mkstemp + fsync + os.replace) on
            # every call, so a UI polling select turned into a config write
            # storm. Clear the pool whenever either side still holds a
            # selection, so drift is repaired without a redundant write.
            if settings.active_account or pool.current:
                settings.active_account = ""
                pool.set_current(None)
                await asyncio.to_thread(settings.save)
            return {"ok": True, "current": None}
        ok = await asyncio.to_thread(pool.set_current, ident)
        if not ok:
            raise HTTPException(status_code=404, detail=f"account not found: {ident}")
        # Same no-op rule for the named path: re-selecting CURRENT is a read.
        if settings.active_account != pool.current:
            settings.active_account = pool.current
            await asyncio.to_thread(settings.save)
        return {"ok": True, "current": pool.current}

    @app.post("/v1/accounts", status_code=201)
    async def add_account(body: dict, provider: str = "", authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, extract_token
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        fields = {name: str((body or {}).get(name) or "")
                  for name in ("email", "mobile", "password", "token")}
        _reject_oversized_fields(fields)
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        if pid:
            # cfg.provider used to be assigned unvalidated, so a junk id
            # persisted into config.json as an account no pool, login or ban
            # check can ever match.
            _reject_unknown_provider(pid)
            # Registered is not enough: a non-active id (e.g. qwen on a
            # deepseek-only deployment) would land in the SELECTED provider's
            # pool, which then serves it as its own credential.
            _reject_inactive_provider(settings, pid)
        cfg = AccountConfig(email=fields["email"], mobile=fields["mobile"],
                            password=fields["password"],
                            token=extract_token(fields["token"]))
        if not (cfg.email or cfg.mobile or cfg.token):
            raise HTTPException(status_code=400,
                                detail="email, mobile, or token required. " + TOKEN_HELP)
        if pid:
            cfg.provider = pid
        else:
            # Validated too: the configured selected provider is not
            # registry-checked on load, and the junk id used to be persisted
            # onto every account added without an explicit provider.
            cfg.provider = _usable_provider(pools.selected_provider(settings))
        # Per PROVIDER, not per config: pools.account_for_provider partitions
        # the list, so a 3-provider deployment with 22 accounts each was hitting
        # this cap at 64 with every single pool nowhere near full. Counted after
        # cfg.provider is resolved, because that is the pool this account lands
        # in (and it is the value _usable_provider already coerced).
        if len(pools.accounts_for_provider(settings, cfg.provider)) >= MAX_ACCOUNTS:
            raise HTTPException(
                status_code=409,
                detail=(f"account limit reached for provider "
                        f"'{cfg.provider}' ({MAX_ACCOUNTS}): delete an account, "
                        "or give this deployment its own config file"))
        pool = _pool_for(app, pid)
        try:
            from .accounts import add_account as _add
            item = await asyncio.to_thread(_add, settings, pool, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        await asyncio.to_thread(settings.save)
        resp: dict = {"ok": True, "account": item.status()}
        if not cfg.token:
            resp["warning"] = ("no token: password-only fails "
                               "(RISK_DEVICE_DETECTED). " + TOKEN_HELP)
        return resp

    @app.put("/v1/accounts/{identifier}/token")
    async def set_account_token(identifier: str, body: dict, provider: str = "",
                                authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, extract_token, set_account_token
        settings: Settings = app.state.settings
        _reject_token_shaped_identifier(identifier)
        _check_auth(settings, authorization)
        # Same ceiling as POST /v1/accounts, on the same field: this token is
        # persisted verbatim by set_account_token and rewritten by every later
        # save, so an uncapped body lets a caller store megabytes of junk in
        # the credential store. Pure predicate, so calling it on a second route
        # is safe; it only raises.
        _reject_oversized_fields({"token": str((body or {}).get("token") or "")})
        token = extract_token(str((body or {}).get("token") or ""))
        if not token:
            raise HTTPException(status_code=400,
                                detail="token required (raw value or full JSON). " + TOKEN_HELP)
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        _reject_inactive_provider(settings, pid)
        pool = _pool_for(app, pid)
        item = await asyncio.to_thread(pool.get, identifier)
        if item is None:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        item.cfg.token = token
        # Scope the settings write to the pool's provider: the same identifier
        # can exist once per provider, and the unscoped first match wrote the
        # token onto the OTHER provider's record as well.
        effective = pid or pool.provider_id
        await asyncio.to_thread(set_account_token, settings, identifier, token,
                                provider=effective)
        await asyncio.to_thread(settings.save)
        # clear_cooldown now resets the whole AccountHealth (cooldown + ban +
        # browser-login backoff) through the pool the account actually lives in.
        await asyncio.to_thread(pool.clear_cooldown, identifier)
        return {"ok": True, "account": item.status()}

    @app.delete("/v1/accounts/{identifier}")
    async def delete_account(identifier: str, provider: str = "", authorization: str = Header(default="")):
        from .accounts import remove_account
        settings: Settings = app.state.settings
        _reject_token_shaped_identifier(identifier)
        _check_auth(settings, authorization)
        pid = (provider or "").strip()
        _reject_inactive_provider(settings, pid)
        pool = _pool_for(app, pid)
        # Resolved BEFORE pool.remove: provider_id falls back to a member's own
        # provider, and removing the last member of that provider would change
        # the answer.
        effective = pid or pool.provider_id
        removed = await asyncio.to_thread(pool.remove, identifier)
        if not removed:
            # A non-secret identifier: safe (and useful) to echo back.
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        # Scoped: the same identifier can exist once per provider, and a
        # settings-wide first-match delete removed the other provider's record.
        await asyncio.to_thread(remove_account, settings, identifier,
                                provider=effective)
        await asyncio.to_thread(settings.save)
        return {"ok": True, "removed": identifier, "current": pool.current or None}

    @app.post("/v1/accounts/unblock")
    async def unblock_accounts(body: dict | None = None, provider: str = "",
                              authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        identifier = (body or {}).get("identifier") if isinstance(body, dict) else None
        # Same edge rule as the path routes: unblock's identifier arrives in
        # the body, but it lands in the same matcher, so a full token is not an
        # accepted identifier over HTTP either.
        _reject_token_shaped_identifier(identifier)
        _check_auth(settings, authorization)
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        _reject_inactive_provider(settings, pid)
        pool = _pool_for(app, pid)
        cleared = await asyncio.to_thread(pool.clear_cooldown, identifier)
        # Persist the ban-flag clear so labels survive the next restart too.
        await asyncio.to_thread(settings.save)
        return {"ok": True, "cleared": cleared}


def _reject_oversized_fields(fields: dict) -> None:
    """400 on any POST /v1/accounts field above its documented char cap.

    Fields are persisted verbatim and rewritten by every later Settings.save,
    so an uncapped field lets a caller store megabytes of junk in an email or
    password that no code path will ever read.
    """
    limits = {"email": MAX_IDENTIFIER_CHARS, "mobile": MAX_IDENTIFIER_CHARS,
              "password": MAX_SECRET_CHARS, "token": MAX_SECRET_CHARS}
    for name, value in (fields or {}).items():
        cap = limits.get(name, MAX_SECRET_CHARS)
        if len(value) > cap:
            raise HTTPException(
                status_code=400,
                detail=(f"{name} too long: {len(value)} chars exceeds the "
                        f"{cap} char limit"))


def _reject_unknown_provider(pid: str) -> None:
    """400 unless ``pid`` is a registered provider id (shape + membership).

    The registry is the authority on "known provider"; the shape rule mirrors
    ``cli._valid_provider_id`` so a junk id is rejected for its own reason
    rather than as an unknown provider. Imported lazily, the way
    ``server.active_models`` does it: ``core.registry`` must not be pulled in
    at server_routes import time, where it would risk a cycle with
    ``deepseaport.server``.
    """
    from .core import registry as _registry
    candidate = str(pid or "").strip().lower()
    if _PROVIDER_ID_RE.match(candidate) and _registry.get(candidate) is not None:
        return
    try:
        known = ", ".join(sorted(_registry.registered())) or "none"
    except Exception:
        known = "unavailable"
    raise HTTPException(status_code=400,
                        detail=f"unknown provider '{pid}' (known: {known})")


def _reject_inactive_provider(settings, pid) -> None:
    """400 when ``pid`` is a provider this deployment does not serve.

    A registered id is not necessarily an ACTIVE one, and ``_pool_for`` keeps
    its fallback-to-selected behavior for an id outside the active set (pinned
    by tests/test_multi_provider.py). So an account route that accepted one
    applied it to the selected provider's pool: a Qwen JWT stored into -- and
    made CURRENT in -- the DeepSeek pool, which then replayed it to
    chat.deepseek.com. Refusing the explicit id at the edge is the fix, not a
    second-guessing _pool_for. Empty/blank pid is a no-op: the caller asked
    for the selected provider.
    """
    candidate = str(pid or "").strip().lower()
    if not candidate:
        return
    try:
        active = pools.active_provider_ids(settings)
    except Exception:
        active = []
    if candidate in active:
        return
    raise HTTPException(status_code=400, detail=(
        f"provider '{pid}' is not active for this deployment "
        f"(active: {', '.join(active) or 'none'})"))


def _reject_token_shaped_identifier(identifier) -> None:
    """400 when an HTTP-supplied identifier is a full bearer token, not an id.

    Applied at the HTTP edge only: ``accounts._matches`` deliberately accepts a
    full token (the CLI deletes and selects accounts locally), but a URL is the
    wrong place for a credential - it lands in access logs, proxy logs and
    Referer headers, and the old 404 detail echoed it straight back to the
    caller. The message never quotes the value.
    """
    if not _TOKEN_SHAPED_RE.match(str(identifier or "").strip()):
        return
    raise HTTPException(status_code=400, detail=(
        "account identifier looks like a bearer token, not an account "
        "identifier. Use the account's email, mobile number, or identifier (the "
        "'abcdef0123...' form GET /v1/accounts returns). Tokens are not "
        "accepted in a URL because URLs end up in access logs, proxy logs "
        "and Referer headers."))


def _key_matches(token: str, keys) -> bool:
    """Constant-time membership test for an API key; never raises.

    Two things ``hmac.compare_digest`` does not do for free: it refuses
    non-ASCII ``str`` (TypeError, so a UTF-8 Authorization header would 500
    instead of 401), and callers that return on the first match leak *which*
    configured key authenticated through timing. So encode once, compare every
    key, and report the result at the end.
    """
    token_bytes = token.encode("utf-8", "surrogatepass")
    matched = False
    for key in keys or ():
        try:
            key_bytes = str(key).encode("utf-8", "surrogatepass")
        except Exception:
            continue
        if hmac.compare_digest(token_bytes, key_bytes):
            matched = True
    return matched


def _check_auth(settings: Settings, authorization: str) -> None:
    if not settings.keys:
        # Documented local-dev default (tests pin it): no configured key means
        # no gate. config._normalize_keys is what keeps a blank key from
        # silently degrading into this state.
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    # An empty presented token is always a miss, never a match: a client that
    # sends no header must not authenticate against a key that happens to be
    # "" (load-time normalization makes that unreachable, but this is the gate
    # that must not depend on that).
    if not token or not _key_matches(token, settings.keys):
        raise HTTPException(status_code=401, detail="invalid API key")


def _sync_pool_current(pool: AccountPool, settings) -> None:
    """Single sync path: pool CURRENT <-> settings.active_account + save."""
    from .accounts import sync_current_from_settings
    try:
        if sync_current_from_settings(pool, settings):
            try:
                settings.save()
            except Exception:
                pass
    except Exception:
        pass


def _pool_cache(app: FastAPI) -> dict:
    """Return the single provider->pool registry, creating it on demand.

    One storage for every mode: single-provider is just ``{selected: pool}``.
    ``app.state.pool`` remains only as an explicit injected override
    (tests/embedded CLI); when set, it wins and the registry is not consulted.
    """
    cache = getattr(app.state, "pools", None)
    if cache is None:
        cache = {}
        app.state.pools = cache
    return cache


def _pool_for(app: FastAPI, provider_id) -> AccountPool:
    """Return the account pool for ``provider_id``.

    Single-provider mode (or an unknown provider id) delegates to the
    historical ``_server()._pool(app)`` so tests patching
    ``deepseaport.server._pool`` keep their monkeypatch seam. Multi mode
    returns the provider's pool from the shared ``app.state.pools`` registry
    (the same registry single mode uses), building it lazily.
    """
    settings = app.state.settings
    pid = str(provider_id or "").strip().lower()
    if not pools.is_multi(settings) or not pid:
        return _server()._pool(app)
    if pid not in pools.active_provider_ids(settings):
        # Unknown provider: fall back to the primary pool.
        return _server()._pool(app)
    cache = _pool_cache(app)
    if pid not in cache:
        cache[pid] = pools.make_pool(settings, pid, current=pools.current_for(settings, pid))
    return cache[pid]


def _pool(app: FastAPI) -> AccountPool:
    """Return the pool for the selected provider.

    Backed by the same ``app.state.pools`` registry as ``_pool_for`` so single
    and multi mode never keep two divergent stores. An explicitly injected
    ``app.state.pool`` (tests/embedding) still wins for backward compat.
    """
    from .accounts import _matches
    settings = app.state.settings

    injected = getattr(app.state, "pool", None)
    if injected is not None:
        # Settings may change via API (add/remove/select): keep pool in sync.
        # Pool holds refs to same AccountConfig objects when created via API,
        # but CLI/TUI edits replace settings.accounts list: rebind not needed
        # for in-memory pool except CURRENT selection.
        _sync_pool_current(injected, settings)
        return injected

    selected = pools.selected_provider(settings)
    cache = _pool_cache(app)
    pool = cache.get(selected)
    if pool is None:
        # ONLY this provider's accounts, always. The unfiltered
        # ``list(settings.accounts)`` fallback that used to sit here for an
        # unregistered id handed provider X a pool of provider Y's credentials:
        # a bad id on a multi-provider config handed DeepSeek a list of Qwen
        # ES256 JWTs, replayed to chat.deepseek.com as userTokens. A provider
        # with no bound account gets an EMPTY pool and a 503 that names it,
        # which is the rule the sibling sites already follow
        # (``_reject_unknown_provider`` on the account-POST route,
        # ``accounts.usable_provider``, ``tests/test_live_smoke.py``).
        accounts = pools.accounts_for_provider(settings, selected)
        # current_for returns "" in multi mode: no shared CURRENT there.
        current = pools.current_for(settings, selected)
        if current and not any(_matches(a, current) for a in accounts):
            current = ""
        pool = AccountPool(accounts, current=current, provider=selected)
        cache[selected] = pool
    elif not pools.is_multi(settings):
        # Single provider follows settings.active_account; multi keeps each
        # provider on independent auto-failover (see core.pools.current_for).
        _sync_pool_current(pool, settings)
    return pool


def _catalogue_for(settings) -> dict:
    """Central model catalogue dict for /v1/models (F6).

    ``active_models(settings)`` with a MODELS-shaped fallback, so the live
    route and the dry-run route share one listing instead of two loops.
    """
    try:
        catalogue = _server().active_models(settings)
    except Exception:
        catalogue = {}
    if not catalogue:
        try:
            catalogue = {mid: tuple(spec) for mid, spec in _server().MODELS.items()}
        except Exception:
            catalogue = {"deepseek-flash": (False, False, "default")}
    return catalogue or {"deepseek-flash": (False, False, "default")}


def build_models_list(settings, now: int | None = None) -> list[dict]:
    """Central /v1/models payload shared by live + dry-run routes (F6).

    Owner per model via ``registry.resolve_model`` over the active provider
    ids; unknown models fall back to ``deepseek``. Never raises: per-model
    resolution is best-effort so one broken provider cannot 500 the listing.
    """
    from .core import registry as _registry
    if now is None:
        now = int(time.time())
    catalogue = _catalogue_for(settings)
    try:
        resolved = _registry.resolved_ids(pools.active_provider_ids(settings))
    except Exception:
        resolved = ["deepseek"]
    data: list[dict] = []
    for mid in catalogue:
        owner = None
        # Capability flags ride along as extra TOP-LEVEL keys: SDKs ignore
        # unknown keys, and they are the only way a client learns whether
        # /v1/chat/completions will honor the thinking/search toggles for this
        # model (the tuple the live catalogue computes them from is dropped
        # otherwise). False/False when the spec could not be resolved.
        flags = {"thinking": False, "search": False}
        try:
            found = _registry.resolve_model(mid, provider_ids=resolved)
        except Exception:
            found = None
        if found is not None:
            try:
                provider, spec = found
                owner = spec.owned_by or provider.id
                flags = {"thinking": bool(spec.thinking),
                         "search": bool(spec.search)}
            except Exception:
                owner = None
        data.append({"id": mid, "object": "model", "created": now,
                     "owned_by": owner or "deepseek", **flags})
    return data


def _validate_messages(body: dict) -> list:
    """Validate the OpenAI messages array (400 on any shape error)."""
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400,
                            detail="messages must be a list of message objects")
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or not msg.get("role"):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{idx}] must be an object with a 'role' field")
        # Container shape only: render_prompt iterates tool_calls, so a scalar
        # or an object there raised TypeError inside the prompt render and
        # surfaced as an opaque 500 (dry_run reached it through _prepare too).
        # Non-dict ELEMENTS stay accepted: the renderer skips those.
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{idx}].tool_calls must be a list")
    return messages


def _resolve_prep_model(model: str, settings: Settings | None):
    """Resolve ``model`` to (provider, spec) across active providers (404)."""
    from .core import registry
    resolved = registry.resolve_model(
        model,
        provider_ids=(registry.resolved_ids(pools.active_provider_ids(settings))
                      if settings is not None else None))
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"model '{model}' not available")
    return resolved


def _decide_use_tools(body: dict, settings: Settings | None, tools: list) -> bool:
    """Apply the tool_choice + enable_tools gates (warns when gated off)."""
    model = str(body.get("model") or "deepseek-flash")
    use_tools = bool(tools) and body.get("tool_choice", "auto") != "none"
    if settings is not None and not getattr(settings, "enable_tools", True):
        if use_tools:
            logger.warning(
                "ignoring %d client tool(s) for model=%s: enable_tools is "
                "off (agentic harnesses like pi need it on for tool_calls)",
                len(tools), model)
        use_tools = False
    return use_tools


def _prep_tool_limits(settings: Settings | None) -> tuple:
    """Return (tool_args_max_chars|None, forgiving_toolcalls bool)."""
    tool_args_max_chars = None
    forgiving_toolcalls = False
    if settings is not None:
        try:
            tool_args_max_chars = int(
                getattr(settings, "tool_args_max_chars", 0) or 0) or None
        except (TypeError, ValueError):
            tool_args_max_chars = None
        forgiving_toolcalls = bool(getattr(settings, "forgiving_toolcalls", False))
    return tool_args_max_chars, forgiving_toolcalls


def _render_prep_prompt(messages: list, tools: list, use_tools: bool,
                        tool_args_max_chars, settings: Settings | None) -> str:
    """Build the upstream prompt: tool system + reminder + appends (400 if empty)."""
    if use_tools:
        messages = [{"role": "system",
                     "content": tool_system_prompt(
                         tools, max_args_chars=tool_args_max_chars)},
                    *messages]
    prompt = render_prompt(messages)
    if use_tools:
        # Recency reminder: long histories bury the start instruction.
        # Appended after user content; non-tool path unchanged.
        prompt = prompt + "\n\n" + tool_reminder_prompt(tools)
    # User-configured wrapper: append_top at the absolute top (before even
    # the tool system prompt), append_bottom at the absolute bottom (after
    # the tool reminder). Gated by enable_append (False keeps texts stored
    # but sends nothing). Internal newlines preserved for multi-line blocks;
    # edge blank lines trimmed so joining never stacks empty gaps.
    # Whitespace-only counts as empty ("" disables). Checked on the final
    # prompt so appends alone can satisfy the non-empty requirement.
    appends_on = bool(getattr(settings, "enable_append", True)) if settings is not None else True
    raw_top = getattr(settings, "append_top", "") if (settings is not None and appends_on) else ""
    raw_bottom = getattr(settings, "append_bottom", "") if (settings is not None and appends_on) else ""
    top = str(raw_top or "").strip("\r\n")
    bottom = str(raw_bottom or "").strip("\r\n")
    if not top.strip():
        top = ""
    if not bottom.strip():
        bottom = ""
    if top or bottom:
        parts = ([top] if top else []) + ([prompt] if prompt.strip() else []) + ([bottom] if bottom else [])
        prompt = "\n\n".join(parts)
    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must produce a non-empty prompt "
                   "(upstream rejects empty prompts with 422)")
    return prompt


def _parse_reasoning_effort(body: dict):
    """Return (reasoning_effort|None, invalid_raw|None) for the opencode variant."""
    # Reasoning effort (e.g. pi agents send reasoning_effort: medium).
    # Honored by the opencode provider as the serve message `variant`
    # (built-ins: none/minimal/low/medium/high/xhigh); other providers
    # ignore it like the thinking flag. Unknown values stay warned below.
    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is None and isinstance(body.get("reasoning"), dict):
        reasoning_effort = body["reasoning"].get("effort")
    if isinstance(reasoning_effort, str):
        reasoning_effort = reasoning_effort.strip().lower() or None
    else:
        reasoning_effort = None
    if reasoning_effort is not None and reasoning_effort not in (
            "none", "minimal", "low", "medium", "high", "xhigh"):
        # The resolved local, never body["reasoning_effort"]: the nested
        # ``reasoning.effort`` spelling is what agent harnesses send, and
        # returning the flat key made that form lose the warning entirely
        # (server_routes.py:984-993 returned None for it). This is the normalized
        # value (stripped + lowercased), which is the spelling that was
        # actually compared against the built-in variants above, so the log
        # quotes exactly the token that was rejected.
        return None, reasoning_effort
    return reasoning_effort, None


# `stream` spellings, mirroring config.py:919-922 (`_bool`): the same
# truthy/falsy token sets the operator's env and config.json already use, so
# there is one table in the project rather than one per call site.
_TRUTHY_TOKENS = ("1", "true", "yes", "on")
_FALSY_TOKENS = ("0", "false", "no", "off")


def _stream_flag(body: dict) -> bool:
    """Parse the OpenAI ``stream`` field. Unrecognised input keeps the default.

    ``bool(body.get("stream"))`` was wrong for every non-empty string: an
    agent harness (or a hand-rolled curl) sending ``{"stream": "false"}`` --
    JSON quoting a boolean, which is how the field arrives from shell wrappers
    and YAML/ini config layers -- got ``True`` and an SSE body when it asked
    for JSON. Reading the tokens explicitly keeps those requests non-streaming
    without turning an odd-but-previously-accepted body into a 400.
    """
    raw = body.get("stream")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in _TRUTHY_TOKENS:
            return True
        if token in _FALSY_TOKENS:
            return False
    # Missing, null, or a token in neither set: fall back to the old default
    # (no streaming) instead of raising.
    return False


def _extract_passthrough(body: dict) -> dict:
    """Collect OpenAI passthrough fields + unsupported map + stream_options.

    Upstream payloads (DeepSeek/Qwen completion_payload) carry only
    prompt + thinking/search/model — no temperature/top_p/stop/n/
    penalties/response_format, and no way to force or serialize parallel tool
    calls (tools are rendered into the prompt). stream_options is honored
    locally: when include_usage is true the streamer emits a final usage chunk.
    """
    unsupported: dict = {}
    # temperature / top_p / parallel_tool_calls all send the value straight
    # through when present. The web upstreams have no native tool calling:
    # tools are serialized into the prompt and parsed back out, so there is
    # nothing for parallel_tool_calls to switch off. Warn and drop, never echo.
    # logprobs / top_logprobs / seed / logit_bias are in the same bucket: the
    # upstream payload carries prompt/thinking/search only, so a caller asking
    # for token logprobs or a reproducible sample used to get a silent drop.
    for _k in ("temperature", "top_p", "parallel_tool_calls",
               "logprobs", "top_logprobs", "seed", "logit_bias"):
        if body.get(_k) is not None:
            unsupported[_k] = body[_k]
    _max_tok = body.get("max_tokens", body.get("max_completion_tokens", None))
    if _max_tok is not None:
        unsupported["max_tokens"] = _max_tok
    if body.get("stop") not in (None, "", []):
        unsupported["stop"] = body.get("stop")
    try:
        _n = int(body.get("n", 1) or 1)
    except (TypeError, ValueError):
        _n = 1
    if _n != 1:
        unsupported["n"] = body.get("n")
    if body.get("presence_penalty") not in (None, 0, 0.0):
        unsupported["presence_penalty"] = body.get("presence_penalty")
    if body.get("frequency_penalty") not in (None, 0, 0.0):
        unsupported["frequency_penalty"] = body.get("frequency_penalty")
    _rf = body.get("response_format", None)
    if _rf not in (None, {}, {"type": "text"}):
        unsupported["response_format"] = _rf
    # tool_choice: "none" and "auto" are honored (_decide_use_tools gates
    # tools on the former), so only the values that ask the bridge to FORCE a
    # call are reported. Those are the ones a client waits on: with
    # "required" or a specific function named, a caller expects a tool_call
    # and the web upstream may answer in prose instead -- previously that
    # expectation was never told to the user. Reporting them must not change
    # whether tools are enabled; that stays entirely in _decide_use_tools.
    _tc = body.get("tool_choice")
    if isinstance(_tc, dict) or (_tc is not None and _tc not in ("auto", "none")):
        unsupported["tool_choice"] = _tc
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        stream_options = {}
    reasoning_effort, invalid_raw = _parse_reasoning_effort(body)
    if invalid_raw is not None:
        unsupported["reasoning_effort"] = invalid_raw
    return {
        "stream_options": stream_options,
        "reasoning_effort": reasoning_effort,
        "unsupported": unsupported,
    }


def _prepare(body: dict, settings: Settings | None = None) -> dict:
    """Validate + format one chat request (F6 orchestrator, signature frozen).

    Steps (each in its helper above; dry_run reuses this): validate messages,
    resolve model->provider, gate tools, render prompt + appends, collect
    OpenAI passthrough/unsupported. Return shape unchanged.
    Provider dispatch below (registry.get(prep["provider"])) is the prior
    boundary-wave injection point — kept as-is.
    """
    model = str(body.get("model") or "deepseek-flash")
    messages = _validate_messages(body)
    provider, spec = _resolve_prep_model(model, settings)
    thinking, search, model_type = spec.thinking, spec.search, spec.model_type
    tools = body.get("tools") or []
    use_tools = _decide_use_tools(body, settings, tools)
    if use_tools and len(tools) > MAX_TOOLS_PER_REQUEST:
        # Counted after the enable_tools/tool_choice gates on purpose: only the
        # rendered path serializes schemas into the prompt, so gating first
        # keeps the cap tied to the work it bounds (a gated-off tool list
        # costs nothing and stays accepted). Checked before the render, which
        # is the expensive part: every schema is dumped into the system
        # prompt, and that prompt is what upstream bills and context-limits.
        raise HTTPException(
            status_code=400,
            detail=(f"too many tools: {len(tools)} exceeds the "
                    f"{MAX_TOOLS_PER_REQUEST} tool limit"))
    tool_args_max_chars, forgiving_toolcalls = _prep_tool_limits(settings)
    prompt = _render_prep_prompt(messages, tools, use_tools,
                                tool_args_max_chars, settings)
    passthrough = _extract_passthrough(body)
    unsupported = passthrough["unsupported"]
    if unsupported:
        logger.warning(
            "ignoring unsupported OpenAI params for web upstream "
            "model=%s params=%s (upstream accepts prompt/thinking/search only)",
            model, sorted(unsupported))
    return {
        "model": model, "stream": _stream_flag(body),
        "tools": tools if use_tools else [], "prompt": prompt,
        "tool_args_max_chars": tool_args_max_chars,
        "forgiving_toolcalls": forgiving_toolcalls,
        "thinking": thinking, "search": search, "model_type": model_type,
        "reasoning_effort": passthrough["reasoning_effort"],
        "provider": provider.id,
        "stream_options": passthrough["stream_options"],
        "unsupported_params": unsupported,
    }
