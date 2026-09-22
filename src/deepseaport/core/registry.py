"""Provider registry: lazy discovery of built-in provider modules.

Modules listed in ``_BUILTIN_MODULES`` are imported lazily and asked for a
``create_provider()`` factory. Import/registration failures are logged and
never raised, so a broken optional provider cannot take the server down.
This module must never import ``deepseaport.server`` (cycle).
"""

from __future__ import annotations

import importlib
import logging

from .provider import ModelSpec, Provider

logger = logging.getLogger("deepseaport.providers")

_BUILTIN_MODULES = ("deepseaport.providers.deepseek", "deepseaport.providers.qwen")

_REGISTRY: dict[str, Provider] = {}
_LOADED = False


def _ensure_loaded() -> None:
    """Import built-in provider modules once, registering their factories."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod_name in _BUILTIN_MODULES:
        try:
            module = importlib.import_module(mod_name)
            factory = getattr(module, "create_provider", None)
            if factory is None:
                logger.warning("provider module %s has no create_provider()", mod_name)
                continue
            provider = factory()
            if provider is not None:
                register(provider)
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider module %s failed to load: %s", mod_name, exc)


def register(provider: Provider) -> Provider:
    """Register a provider keyed by its lowercased id; returns it."""
    key = str(getattr(provider, "id", "") or "").strip().lower()
    if not key:
        raise ValueError("provider.id must be non-empty")
    _REGISTRY[key] = provider
    return provider


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


def resolve_model(model: str,
                  provider_ids: list[str] | None = None) -> tuple[Provider, ModelSpec] | None:
    """Resolve a model id to (provider, spec) across providers, else None."""
    _ensure_loaded()
    if provider_ids is None:
        providers = list(_REGISTRY.values())
    else:
        providers = [p for pid in provider_ids
                     if (p := _REGISTRY.get(str(pid or "").strip().lower())) is not None]
    for provider in providers:
        spec = provider.model_spec(model)
        if spec is not None:
            return provider, spec
    return None


def resolved_ids(provider_ids: list[str] | None) -> list[str]:
    """Registered subset of provider_ids (lowercased, deduped); ['deepseek'] fallback when none resolve."""
    _ensure_loaded()
    out: list[str] = []
    for raw in (provider_ids or []):
        key = str(raw or "").strip().lower()
        if key and key not in out and key in _REGISTRY:
            out.append(key)
    return out or ["deepseek"]