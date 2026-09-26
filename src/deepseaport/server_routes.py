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
import logging
import threading
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .accounts import AccountPool
from .config import AccountConfig, Settings
from .core import pools
from .core.obscura_bridge import ObscuraBridge
from .server_concurrency import (
    _StreamCancelled,
    _acquire_slot_or_499,
    _watch_disconnect,
)
from .core.tools import render_prompt, tool_reminder_prompt, tool_system_prompt


def _server():
    """Late-bound deepseaport.server module for monkeypatch-compatible lookups."""
    from . import server as _s
    return _s


logger = logging.getLogger("deepseaport.server")


def register_routes(app: FastAPI) -> None:
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
        return {"ok": True, "ready": ready, "waf": waf,
                "accounts": total, "accounts_with_token": with_token,
                "banned": banned, "pool": pool_size, "current": current}

    @app.get("/v1/waf/status")
    async def waf_status():
        bridge: ObscuraBridge | None = getattr(app.state, "bridge", None)
        if bridge is None:
            return {"warmed": False, "error": "bridge not initialised yet"}
        return await asyncio.to_thread(bridge.status)

    @app.get("/v1/models")
    async def list_models():
        settings: Settings = app.state.settings
        return {"object": "list", "data": build_models_list(settings)}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict, request: Request,
                               authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        prep = _prepare(body, settings)  # raises 400/404 fast, before touching accounts
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
        finally:
            watcher.cancel()
        return _server()._openai_response(prep, result)

    @app.get("/v1/accounts")
    async def list_accounts(provider: str = "", authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pid = (provider or "").strip()
        pool = _pool_for_request(app, pid)
        data = await asyncio.to_thread(pool.status)
        return {"object": "list", "current": pool.current or None, "data": data}

    @app.post("/v1/accounts/select")
    async def select_account(body: dict | None = None,
                             authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        ident = ((body or {}).get("identifier") or "").strip() if isinstance(body, dict) else ""
        pool = _server()._pool(app)
        if not ident or ident.lower() in ("none", "all", "auto"):
            settings.active_account = ""
            pool.set_current(None)
            await asyncio.to_thread(settings.save)
            return {"ok": True, "current": None}
        ok = await asyncio.to_thread(pool.set_current, ident)
        if not ok:
            raise HTTPException(status_code=404, detail=f"account not found: {ident}")
        settings.active_account = pool.current
        await asyncio.to_thread(settings.save)
        return {"ok": True, "current": pool.current}

    @app.post("/v1/accounts", status_code=201)
    async def add_account(body: dict, provider: str = "", authorization: str = Header(default="")):
        from .accounts import TOKEN_HELP, extract_token
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        cfg = AccountConfig(email=str(body.get("email") or ""),
                            mobile=str(body.get("mobile") or ""),
                            password=str(body.get("password") or ""),
                            token=extract_token(str(body.get("token") or "")))
        if not (cfg.email or cfg.mobile or cfg.token):
            raise HTTPException(status_code=400,
                                detail="email, mobile, or token required. " + TOKEN_HELP)
        if pid:
            cfg.provider = pid
        else:
            cfg.provider = pools.selected_provider(settings)
        pool = _pool_for_request(app, pid)
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
        _check_auth(settings, authorization)
        token = extract_token(str((body or {}).get("token") or ""))
        if not token:
            raise HTTPException(status_code=400,
                                detail="token required (raw value or full JSON). " + TOKEN_HELP)
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        pool = _pool_for_request(app, pid)
        item = await asyncio.to_thread(pool.get, identifier)
        if item is None:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        item.cfg.token = token
        await asyncio.to_thread(set_account_token, settings, identifier, token)
        await asyncio.to_thread(settings.save)
        # clear_cooldown now resets the whole AccountHealth (cooldown + ban +
        # browser-login backoff) through the pool the account actually lives in.
        await asyncio.to_thread(pool.clear_cooldown, identifier)
        return {"ok": True, "account": item.status()}

    @app.delete("/v1/accounts/{identifier}")
    async def delete_account(identifier: str, provider: str = "", authorization: str = Header(default="")):
        from .accounts import remove_account
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        pid = (provider or "").strip()
        pool = _pool_for_request(app, pid)
        removed = await asyncio.to_thread(pool.remove, identifier)
        if not removed:
            raise HTTPException(status_code=404, detail=f"account not found: {identifier}")
        await asyncio.to_thread(remove_account, settings, identifier)
        await asyncio.to_thread(settings.save)
        return {"ok": True, "removed": identifier, "current": pool.current or None}

    @app.post("/v1/accounts/unblock")
    async def unblock_accounts(body: dict | None = None, provider: str = "",
                              authorization: str = Header(default="")):
        settings: Settings = app.state.settings
        _check_auth(settings, authorization)
        identifier = (body or {}).get("identifier") if isinstance(body, dict) else None
        pid = str((body.get("provider") if isinstance(body, dict) else "") or provider or "").strip()
        pool = _pool_for_request(app, pid)
        cleared = await asyncio.to_thread(pool.clear_cooldown, identifier)
        # Persist the ban-flag clear so labels survive the next restart too.
        await asyncio.to_thread(settings.save)
        return {"ok": True, "cleared": cleared}


def _check_auth(settings: Settings, authorization: str) -> None:
    if not settings.keys:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in settings.keys:
        raise HTTPException(status_code=401, detail="invalid API key")


def _sync_pool_current(pool: AccountPool, settings) -> None:
    """Single sync path: pool CURRENT <-> settings.active_account + save.

    Replaces the inline sync_current_from_settings + settings.save() copies
    below. Signatures/persistence unchanged.
    """
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
    from .core import registry as _registry
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
        accounts = pools.accounts_for_provider(settings, selected)
        if not accounts and getattr(settings, "accounts", []):
            if _registry.get(selected) is None:
                logger.warning("provider '%s' not registered; using all accounts", selected)
                accounts = list(settings.accounts)
        # current_for returns "" in multi mode: no shared CURRENT there.
        current = pools.current_for(settings, selected)
        if current and not any(_matches(a, current) for a in accounts):
            current = ""
        pool = AccountPool(accounts, current=current)
        cache[selected] = pool
    elif not pools.is_multi(settings):
        # Single provider follows settings.active_account; multi keeps each
        # provider on independent auto-failover (see core.pools.current_for).
        _sync_pool_current(pool, settings)
    return pool


def _pool_for_request(app: FastAPI, provider_id) -> AccountPool:
    """Central pool selection for the account routes (F5).

    Single home for the ``_pool_for(app, pid) if (pid and is_multi) else
    _server()._pool(app)`` gate that was repeated in every handler. Empty pid
    or single-provider mode goes through the historical ``_server()._pool``
    seam so ``monkeypatch.setattr("deepseaport.server._pool", ...)`` keeps
    working; multi mode with a real pid uses the per-provider registry.
    """
    settings = app.state.settings
    if pools.needs_provider_pool(settings, provider_id):
        return _pool_for(app, provider_id)
    return _server()._pool(app)


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
        try:
            found = _registry.resolve_model(mid, provider_ids=resolved)
        except Exception:
            found = None
        if found is not None:
            try:
                provider, spec = found
                owner = spec.owned_by or provider.id
            except Exception:
                owner = None
        data.append({"id": mid, "object": "model", "created": now,
                     "owned_by": owner or "deepseek"})
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
        return None, body.get("reasoning_effort")
    return reasoning_effort, None


def _extract_passthrough(body: dict) -> dict:
    """Collect OpenAI passthrough fields + unsupported map + stream_options.

    Upstream payloads (DeepSeek/Qwen completion_payload) carry only
    prompt + thinking/search/model — no temperature/top_p/stop/n/
    penalties/response_format. stream_options is honored locally: when
    include_usage is true the streamer emits a final usage chunk.
    """
    unsupported: dict = {}
    if body.get("temperature") is not None:
        unsupported["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        unsupported["top_p"] = body.get("top_p")
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
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        stream_options = {}
    reasoning_effort, invalid_raw = _parse_reasoning_effort(body)
    if invalid_raw is not None:
        unsupported["reasoning_effort"] = invalid_raw
    return {
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_tokens": _max_tok,
        "stop": body.get("stop"),
        "n": _n,
        "presence_penalty": body.get("presence_penalty"),
        "frequency_penalty": body.get("frequency_penalty"),
        "response_format": _rf,
        "stream_options": stream_options,
        "reasoning_effort": reasoning_effort,
        "unsupported": unsupported,
    }


def _prepare(body: dict, settings: Settings | None = None) -> dict:
    """Validate + format one chat request (F6 orchestrator, signature frozen).

    Steps (each in its helper above; dry_run reuses this): validate messages,
    resolve model->provider, gate tools, render prompt + appends, collect
    OpenAI passthrough/unsupported. Return shape unchanged (21 keys).
    Provider dispatch below (registry.get(prep["provider"])) is the prior
    boundary-wave injection point — kept as-is.
    """
    model = str(body.get("model") or "deepseek-flash")
    messages = _validate_messages(body)
    provider, spec = _resolve_prep_model(model, settings)
    thinking, search, model_type = spec.thinking, spec.search, spec.model_type
    tools = body.get("tools") or []
    use_tools = _decide_use_tools(body, settings, tools)
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
        "model": model, "stream": bool(body.get("stream", False)),
        "tools": tools if use_tools else [], "prompt": prompt,
        "tool_args_max_chars": tool_args_max_chars,
        "forgiving_toolcalls": forgiving_toolcalls,
        "thinking": thinking, "search": search, "model_type": model_type,
        "reasoning_effort": passthrough["reasoning_effort"],
        "provider": provider.id,
        "temperature": passthrough["temperature"],
        "top_p": passthrough["top_p"],
        "max_tokens": passthrough["max_tokens"],
        "stop": passthrough["stop"],
        "n": passthrough["n"],
        "presence_penalty": passthrough["presence_penalty"],
        "frequency_penalty": passthrough["frequency_penalty"],
        "response_format": passthrough["response_format"],
        "stream_options": passthrough["stream_options"],
        "unsupported_params": unsupported,
    }
