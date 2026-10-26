"""Provider-scoped account pool helpers (provider-neutral).

Pure functions over ``Settings`` that decide which providers are active and
which accounts belong to a provider. Late-imports ``..accounts`` inside
functions so this module never participates in an import cycle and never
imports ``zourcewell.server``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("zourcewell.core.pools")

# Provider ids already reported as ignored (see active_provider_ids).
# active_provider_ids is on the per-request path, so an unguarded warning
# would repeat every call; the config does not change under a running server.
_WARNED_IGNORED_PROVIDERS: set[str] = set()


def _provider_of(account) -> str:
    """Normalized provider id for an account (defaults to deepseek)."""
    return (getattr(account, "provider", "deepseek") or "deepseek").strip().lower()


def selected_provider(settings) -> str:
    """Return the single selected provider id (lowercased, defaults deepseek)."""
    return (getattr(settings, "provider", "") or "deepseek").strip().lower()


def active_provider_ids(settings) -> list[str]:
    """Return the active provider ids in order, deduped and lowercased.

    ``settings.providers`` with two or more non-empty ids selects multi mode;
    otherwise the single ``settings.provider`` is used.

    A ONE-entry ``providers`` list stays ignored, and that is deliberate, not
    an oversight: multi mode is defined as 2+ ids (ARCHITECTURE.md, "Multi
    mode: 2+ ids in Settings.providers") and it is pinned by
    tests/test_multi_provider.py, so honouring one entry would flip
    is_multi() and take the whole per-provider pool path (independent failover,
    union model catalogue) for what reads like single-provider setup. The bug
    was the silence: ``provider: "deepseek"`` + ``providers: ["qwen"]`` ran
    deepseek with no hint that qwen had been written down and dropped. Now it
    is reported once per id, naming the ignored entry and the one in effect.
    """
    ids: list[str] = []
    for raw in (getattr(settings, "providers", None) or []):
        key = str(raw or "").strip().lower()
        if key and key not in ids:
            ids.append(key)
    if len(ids) >= 2:
        return ids
    selected = selected_provider(settings)
    if len(ids) == 1 and ids[0] != selected and ids[0] not in _WARNED_IGNORED_PROVIDERS:
        _WARNED_IGNORED_PROVIDERS.add(ids[0])
        logger.warning(
            "providers=%r lists a single provider, which multi mode ignores "
            "(it needs 2 or more); running provider=%r instead. Set provider=%r, "
            "or list two providers to enable multi mode.",
            ids[0], selected, ids[0])
    return [selected]


def is_multi(settings) -> bool:
    """True when more than one provider is active for this server."""
    return len(active_provider_ids(settings)) > 1


def providers_requiring_accounts(settings) -> list[str]:
    """Active provider ids that need an ``AccountPool`` slot and a token.

    The single source of truth for ``Provider.requires_accounts``. A provider
    with ``requires_accounts = False`` is served happily with zero accounts --
    opencode runs a local CLI that multiplexes its own sessions, has no token,
    and its ``login()`` raises. Anything gating on "does this user need to add
    a credential first?" must ask here.

    Registry unavailable -> every id is treated as needing accounts, so the
    accounts-based screens stay the safe default rather than telling someone
    with a web upstream to go add a credential that is silently unused.
    """
    try:
        from . import registry as _registry

        return [pid for pid in active_provider_ids(settings)
                if getattr(_registry.get(pid), "requires_accounts", True)]
    except Exception:
        return list(active_provider_ids(settings))


def any_provider_needs_accounts(settings) -> bool:
    """True when at least one active provider needs an account/token.

    True when the active set is empty as well: an unconfigured server has not
    been shown to be accountless, and every preflight/hint caller wants the
    conservative answer in that case.
    """
    return bool(providers_requiring_accounts(settings))


def only_accountless_providers(settings) -> bool:
    """True when providers ARE active and NONE of them needs an account.

    This is the "serve can start with an empty pool" case, and it is
    deliberately distinct from :func:`any_provider_needs_accounts`: an empty
    active set must NOT read as accountless, or a misconfigured server would
    sail past preflight and then 503 on the first request.
    """
    ids = active_provider_ids(settings)
    return bool(ids) and not providers_requiring_accounts(settings)


def accounts_for_provider(settings, provider_id) -> list:
    """Return settings accounts bound to ``provider_id`` (case-insensitive)."""
    key = str(provider_id or "").strip().lower()
    out: list = []
    for account in (getattr(settings, "accounts", None) or []):
        if _provider_of(account) == key:
            out.append(account)
    return out


def make_pool(settings, provider_id, current: str = ""):
    """Build an AccountPool limited to one provider's accounts.

    Canonical per-provider constructor (FROZEN seam): server_routes._pool_for
    builds multi-mode pools through this; server_routes._pool (selected
    provider) uses it when no unknown-provider fallback is needed. Keep all
    pool construction here so the two route helpers never diverge.
    """
    from ..accounts import AccountPool
    return AccountPool(accounts_for_provider(settings, provider_id),
                       current=current, provider=provider_id)


def current_for(settings, provider_id) -> str:
    """Return settings.active_account only for the single selected provider.

    Multi mode keeps each provider on independent auto-failover (no shared
    CURRENT) until per-provider current support exists.
    """
    if is_multi(settings):
        return ""
    if str(provider_id or "").strip().lower() != selected_provider(settings):
        return ""
    return getattr(settings, "active_account", "") or ""