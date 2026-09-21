"""Provider-scoped account pool helpers (provider-neutral).

Pure functions over ``Settings`` that decide which providers are active and
which accounts belong to a provider. Late-imports ``..accounts`` inside
functions so this module never participates in an import cycle and never
imports ``deepseaport.server``.
"""

from __future__ import annotations

# Sentinel id used by menus to represent "all active providers" at once.
MULTI_ID = "multi"


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
    """
    ids: list[str] = []
    for raw in (getattr(settings, "providers", None) or []):
        key = str(raw or "").strip().lower()
        if key and key not in ids:
            ids.append(key)
    if len(ids) >= 2:
        return ids
    return [selected_provider(settings)]


def is_multi(settings) -> bool:
    """True when more than one provider is active for this server."""
    return len(active_provider_ids(settings)) > 1


def accounts_for_provider(settings, provider_id) -> list:
    """Return settings accounts bound to ``provider_id`` (case-insensitive)."""
    key = str(provider_id or "").strip().lower()
    out: list = []
    for account in (getattr(settings, "accounts", None) or []):
        if _provider_of(account) == key:
            out.append(account)
    return out


def make_pool(settings, provider_id, current: str = ""):
    """Build an AccountPool limited to one provider's accounts."""
    from ..accounts import AccountPool
    return AccountPool(accounts_for_provider(settings, provider_id), current=current)


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