"""FastAPI server: OpenAI-compatible endpoints over the DeepSeek, Qwen and opencode backends.

Concurrency model: endpoints are async; all blocking upstream I/O runs in
worker threads. Streaming responses bridge a producer thread (upstream SSE)
to the response generator through a queue, so first tokens reach the
harness without waiting for the full reply. Tool calls are parsed from the
buffered full text at FINISHED and emitted as OpenAI tool_call deltas.

The endpoints live in ``deepseaport.server_routes``, the completion/retry
engine in ``deepseaport.providers.deepseek.completion_service``, and the
executors + lifecycle in ``deepseaport.server_concurrency``. This module is
thin wiring only: ``create_app``, the model catalogue, log configuration,
and the three monkeypatch seams the route layer deliberately looks up here.
Everything else is imported from the module that owns it.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from . import __version__
from .providers.deepseek.models import MODELS_SPEC
from .config import Settings, load_settings
from .server_concurrency import apply_log_level, lifespan
from .server_routes import _pool, register_routes

# Three names stay reachable as attributes of THIS module, not for
# backward compatibility but because the route layer resolves them through
# it on purpose:
#   _pool          -- server_routes._pool_for / _pool delegate the
#                     single-provider path here, so patching one function
#                     covers every pool entry point;
#   _openai_response / parse_tool_calls -- the response shim and the tool
#                     parser the routes substitute in tests.
# They are re-pointed at the owning module if that indirection is ever
# dropped; nothing else re-exports anything.
from .core.openai import _openai_response
from .core.tools import parse_tool_calls

# Log format for the process-owning entry point (create_app). It used to run at
# import time, which made `import deepseaport.server` mutate process-global
# logging for every embedder, test run and CLI import that never serves a
# request. ``logging.basicConfig`` keeps its "only if the root logger has no
# handlers" contract, so a host that configured logging itself (or chat_ui,
# which forces the root level to ERROR) is left alone; ``force=True`` is the
# opt-in for callers that want deepseaport's format to win.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(*, force: bool = False) -> None:
    """Install the deepseaport console log format (no-op when already set up)."""
    logging.basicConfig(format=LOG_FORMAT, force=force)


# Derived from the DeepSeek provider catalogue, kept as a plain dict for
# backwards compatibility (tests assert exact tuple values and iteration).
MODELS = {mid: (s.thinking, s.search, s.model_type)
          for mid, s in MODELS_SPEC.items()}


def active_models(settings=None) -> dict[str, tuple[bool, bool, str]]:
    """Return the MODELS-shaped catalogue for the given settings.

    Union of ``tuple_spec()`` for ``resolved_ids(active_provider_ids(...))``,
    falling back to ``dict(MODELS)`` when empty or settings is None.
    Delegates the union to ``registry.catalogue_tuples_for`` (single home);
    this wrapper stays as the frozen compat entry point.
    """
    from .core import registry
    if settings is None:
        return dict(MODELS)
    try:
        out = registry.catalogue_tuples_for(settings)
    except Exception:
        out = {}
    return out or dict(MODELS)


def create_app(settings: Settings | None = None) -> FastAPI:
    # Process-owning entry point: this is where the process decides it wants
    # console logging, so importing this module stays side-effect free.
    configure_logging()
    # No /docs, /redoc or /openapi.json: those are an unauthenticated,
    # machine-readable inventory of every route (including the account CRUD
    # surface and the {identifier} path shape) on a local-first bridge that
    # holds a credential store. The live routes are the only surface we expose.
    app = FastAPI(title="deepseaport", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings or load_settings()
    apply_log_level(getattr(app.state.settings, "log_level", "INFO"))
    # One registry for every mode; `pool` stays as an explicit override seam.
    app.state.pools = {}
    app.state.pool = None
    register_routes(app)
    return app
