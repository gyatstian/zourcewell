"""Provider registry: lazy discovery of built-in provider modules.

Modules listed in ``_BUILTIN_MODULES`` are imported lazily and asked for a
``create_provider()`` factory. Import/registration failures are logged and
never raised, so a broken optional provider cannot take the server down.
The load is published atomically (registry plus loaded flag together, under
``_LOAD_LOCK``), so no reader ever observes a partial registry.
This module must never import ``deepseaport.server`` (cycle).
"""

from __future__ import annotations

import importlib
import logging
import threading

from .provider import ModelSpec, Provider

logger = logging.getLogger("deepseaport.providers")

# The single registration point: adding a built-in provider is one line here,
# never a caller-side import. A module that fails to import is logged and
# skipped, never raised, so one broken provider cannot take the rest down.
# deepseek must stay in this tuple: default() returns it by name whenever it is
# registered, and it is the fallback of resolved_ids() and most callers' default.
_BUILTIN_MODULES = ("deepseaport.providers.deepseek", "deepseaport.providers.qwen",
                     "deepseaport.providers.opencode")

_REGISTRY: dict[str, Provider] = {}
_LOADED = False
# Serialises _ensure_loaded. The flag is checked outside the lock, so two
# threads can both decide to load; the lock keeps them from interleaving
# registrations into _REGISTRY, which a reader may be walking.
_LOAD_LOCK = threading.Lock()


def _key(provider: Provider) -> str:
    """Registry key for a provider: its lowercased, stripped id."""
    key = str(getattr(provider, "id", "") or "").strip().lower()
    if not key:
        raise ValueError("provider.id must be non-empty")
    return key


def _ensure_loaded() -> None:
    """Import the built-in provider modules once, on first use.

    ``_LOADED`` is set AFTER the imports, never before: warmup_on_startup
    defaults on and its daemon thread can still be importing when uvicorn
    accepts connections, so latching early let a reader see a half-populated
    registry -- resolve_model() returned None and a valid model answered 404,
    or default() raised as an unhandled 500. Readers either block here or see
    the finished registry.

    A module that fails to import is logged and skipped; the load still
    latches. Every other module is one line away in a shipped tree, so
    retrying a broken import on every request only bought a warning per
    request.
    """
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:  # another thread finished the load while we waited
            return
        for mod_name in _BUILTIN_MODULES:
            try:
                module = importlib.import_module(mod_name)
                factory = getattr(module, "create_provider", None)
                if factory is None:
                    logger.warning("provider module %s has no create_provider()", mod_name)
                    continue
                provider = factory()
                if provider is not None:
                    # Updated in place, never rebound: callers and tests hold
                    # _REGISTRY itself (they pop ids out of it), so
                    # reassigning the name would drop whatever they put there.
                    _REGISTRY[_key(provider)] = provider
            except Exception as exc:  # noqa: BLE001
                logger.warning("provider module %s failed to load: %s", mod_name, exc)
        _LOADED = True


def get(provider_id: str) -> Provider | None:
    """Return the registered provider for an id (case-insensitive), or None."""
    _ensure_loaded()
    return _REGISTRY.get(str(provider_id or "").strip().lower())


def registered() -> dict[str, Provider]:
    """Return a snapshot copy of all registered providers keyed by id."""
    _ensure_loaded()
    return dict(_REGISTRY)


def default() -> Provider:
    """Return the DeepSeek provider when present, else the first registered."""
    _ensure_loaded()
    if "deepseek" in _REGISTRY:
        return _REGISTRY["deepseek"]
    for provider in _REGISTRY.values():
        return provider
    raise RuntimeError("no providers registered")


def _in_catalogue(provider: Provider, model: str) -> bool:
    """True when ``model`` is in ``provider``'s OWN catalogue, not synthesised.

    The distinction that makes :func:`resolve_model` order-independent: a
    provider whose ``model_spec`` returns a spec for an id it does not list
    (opencode's ``provider/model`` passthrough for whatever the local serve
    instance is authenticated for) must never outrank a provider that actually
    publishes the model. Returns True when the catalogue cannot be read, so a
    provider we cannot question is never demoted on a guess.
    """
    try:
        catalogue = provider.models()
    except Exception:
        return True
    key = str(model or "").strip().lower()
    return any(str(mid).strip().lower() == key for mid in (catalogue or {}))


def resolve_model(model: str,
                  provider_ids: list[str] | None = None) -> tuple[Provider, ModelSpec] | None:
    """Resolve a model id to (provider, spec) across providers, else None.

    Two invariants, both load-bearing:

    * a provider that RAISES from ``model_spec`` is skipped, exactly as its two
      siblings already do (``build_models_list`` and the registry's own
      ``catalogue_tuples_for``); one broken provider must not 500 the chat
      route or the model listing.
    * the answer does not depend on the ORDER of ``provider_ids``. A real
      catalogue hit always wins; a synthesised passthrough spec is only
      returned when no provider in the set actually lists the model. Without
      this, a ``providers`` list with opencode first made its slash-passthrough
      swallow every model id containing a slash, and the same config with
      opencode last resolved them correctly.
    """
    _ensure_loaded()
    if provider_ids is None:
        providers = list(_REGISTRY.values())
    else:
        providers = [p for pid in provider_ids
                     if (p := _REGISTRY.get(str(pid or "").strip().lower())) is not None]
    passthrough: tuple[Provider, ModelSpec] | None = None
    for provider in providers:
        try:
            spec = provider.model_spec(model)
        except Exception:
            continue
        if spec is None:
            continue
        if _in_catalogue(provider, model):
            return provider, spec
        if passthrough is None:
            passthrough = (provider, spec)
    return passthrough


def resolved_ids(provider_ids: list[str] | None) -> list[str]:
    """Registered subset of provider_ids (lowercased, deduped); ['deepseek'] fallback when none resolve."""
    _ensure_loaded()
    out: list[str] = []
    for raw in (provider_ids or []):
        key = str(raw or "").strip().lower()
        if key and key not in out and key in _REGISTRY:
            out.append(key)
    return out or ["deepseek"]


def catalogue_tuples_for(settings) -> dict[str, tuple[bool, bool, str]]:
    """Central MODELS-shaped catalogue for ``settings`` (no MODELS fallback).

    Union of ``tuple_spec()`` for ``resolved_ids(active_provider_ids(...))``.
    Single home for the loop previously inline in ``server.active_models``;
    callers fall back to ``dict(MODELS)`` when this returns empty (or when
    settings is None). Never imports ``deepseaport.server`` (no cycle).
    """
    _ensure_loaded()
    try:
        from . import pools as _pools
        ids = resolved_ids(_pools.active_provider_ids(settings))
    except Exception:
        ids = ["deepseek"]
    out: dict[str, tuple[bool, bool, str]] = {}
    for pid in ids:
        provider = _REGISTRY.get(pid)
        if provider is None:
            continue
        try:
            out.update(provider.tuple_spec())
        except Exception:
            continue
    return out