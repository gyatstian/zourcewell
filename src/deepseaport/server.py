"""FastAPI server: OpenAI-compatible endpoints over the DeepSeek web client.

Concurrency model: endpoints are async; all blocking DeepSeek I/O runs in
worker threads. Streaming responses bridge a producer thread (DeepSeek SSE)
to the response generator through a queue, so first tokens reach the
harness without waiting for the full reply. Tool calls are parsed from the
buffered full text at FINISHED and emitted as OpenAI tool_call deltas.

The endpoints live in ``deepseaport.server_routes``, the completion/retry
engine in ``deepseaport.providers.deepseek.completion_service``, and the
executors + lifecycle in ``deepseaport.server_concurrency``. This module
keeps ``create_app`` as thin wiring and re-exports every historical name
for backward compatibility (tests + provider compat seams).
"""

from __future__ import annotations

import asyncio  # noqa: F401  (re-exported: historical surface)
import json  # noqa: F401  (re-exported: historical surface)
import logging
import os
import threading  # noqa: F401  (re-exported: historical surface)
import time  # noqa: F401  (re-exported: historical surface)
import uuid  # noqa: F401  (re-exported: historical surface)
from concurrent.futures import ThreadPoolExecutor  # noqa: F401
from contextlib import asynccontextmanager  # noqa: F401

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: F401
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: F401

from .providers.deepseek import client as DS  # noqa: F401
from .providers.deepseek import pow as PoW  # noqa: F401
from .providers.deepseek import protocol as P  # noqa: F401
from .providers.deepseek.models import MODELS_SPEC
from .accounts import AccountPool, PooledAccount  # noqa: F401
from .config import AccountConfig, Settings, load_settings
from .core.obscura_bridge import ObscuraBridge, register_default_bridge  # noqa: F401
from .core.tools import (
    looks_truncated_tool_attempt,  # noqa: F401
    parse_tool_calls,  # noqa: F401
    render_prompt,  # noqa: F401
    tool_reminder_prompt,  # noqa: F401
    tool_system_prompt,  # noqa: F401
)

# --- Re-exports from the split modules (historical public surface) ---------
# Order matters only for readability; all names are importable from
# deepseaport.server exactly as before the split. The completion/stream
# engine lives in providers.deepseek.completion_service; this module holds
# no engine logic, only create_app + catalogue + compat aliases.
from .core.openai import _openai_response  # noqa: F401
from .providers.deepseek.completion_service import (
    _AttemptOutcome,  # noqa: F401
    _LOGIN_FAIL_UNTIL,  # noqa: F401
    _TOKEN_REFRESH_BAN_HINT,  # noqa: F401
    _TOKEN_REFRESH_INFLIGHT,  # noqa: F401
    _TOKEN_REFRESH_LOCK,  # noqa: F401
    _app_settings,  # noqa: F401
    _attempt,  # noqa: F401
    _ban_403_for_item,  # noqa: F401
    _complete_with_failover_sync,  # noqa: F401
    _delete_session_bg,  # noqa: F401
    _ensure_token,  # noqa: F401
    _header_from_challenge,  # noqa: F401
    _headers,  # noqa: F401
    _is_ban_http_exception,  # noqa: F401
    _is_invalid_token_error,  # noqa: F401
    _is_transient_overload,  # noqa: F401
    _login_backoff_set,  # noqa: F401
    _login_backoff_skip,  # noqa: F401
    _probe_ban,  # noqa: F401
    _refresh_token_via_obscura,  # noqa: F401
    _refresh_waf_without_account_lock,  # noqa: F401
    _run_completion,  # noqa: F401
    _run_completion_core,  # noqa: F401
    _server_override,  # noqa: F401
    _session_and_challenge,  # noqa: F401
    _solve,  # noqa: F401
    _stream_completion,  # noqa: F401
    _switch_current_after_ban_failover,  # noqa: F401
    clear_login_backoff,  # noqa: F401
)
from .server_concurrency import (
    FAILOVER_ACQUIRE_TIMEOUT,  # noqa: F401
    INVALID_TOKEN_COOLDOWN_SECONDS,  # noqa: F401
    _CHALLENGE_EXECUTOR,  # noqa: F401
    _CLEANUP_EXECUTOR,  # noqa: F401
    _CONSUMER_GRACE_SECONDS,  # noqa: F401
    _DEFAULT_POOL_SIZE,  # noqa: F401
    _STREAM_EXECUTOR,  # noqa: F401
    _StreamCancelled,  # noqa: F401
    _acquire_slot_or_499,  # noqa: F401
    _await_cancelable,  # noqa: F401
    _startup_warm,  # noqa: F401
    _watch_disconnect,  # noqa: F401
    apply_log_level,
    lifespan,
)
from .server_routes import _check_auth, _pool, _prepare, register_routes  # noqa: F401

logger = logging.getLogger("deepseaport.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Derived from the DeepSeek provider catalogue, kept as a plain dict for
# backwards compatibility (tests assert exact tuple values and iteration).
MODELS = {mid: (s.thinking, s.search, s.model_type)
          for mid, s in MODELS_SPEC.items()}


def active_models(settings=None) -> dict[str, tuple[bool, bool, str]]:
    """Return the MODELS-shaped catalogue for the given settings.

    Union of ``tuple_spec()`` for ``resolved_ids(active_provider_ids(...))``,
    falling back to ``dict(MODELS)`` when empty or settings is None.
    """
    from .core import pools, registry
    if settings is None:
        return dict(MODELS)
    out: dict[str, tuple[bool, bool, str]] = {}
    for pid in registry.resolved_ids(pools.active_provider_ids(settings)):
        provider = registry.get(pid)
        if provider is None:
            continue
        out.update(provider.tuple_spec())
    return out or dict(MODELS)


def error_json(request, exc: HTTPException):  # helper for tests
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="deepseaport", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or load_settings()
    apply_log_level(getattr(app.state.settings, "log_level", "INFO"))
    app.state.pool = None
    register_routes(app)
    return app


app = create_app()
