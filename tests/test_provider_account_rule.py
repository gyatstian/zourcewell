"""The `requires_accounts` rule must have exactly one implementation.

A provider with `requires_accounts = False` (opencode: a local CLI, no pool,
no token, a `login()` that raises) is served happily with zero accounts. Every
preflight check, hint and gate asks "may I tell this user to go add a
credential?" -- and a provider that never uses one makes that advice a dead
end. That rule used to be written out three times (tui, chat_ui, and inlined
in tui._preflight_reason); `core.pools` now owns it, so a fourth provider
inherits it instead of needing it re-implemented per surface.
"""

import inspect

import pytest

from deepseaport.core import pools, registry
from deepseaport.core.provider import Provider


class _Stub(Provider):
    def __init__(self, pid, requires):
        self.id = pid
        self.requires_accounts = requires

    def models(self):
        return {}

    def probe_ban(self, headers, timeout=20):
        return False, None


class _Settings:
    def __init__(self, provider="deepseek", providers=None):
        self.provider = provider
        self.providers = list(providers or [])
        self.accounts = []


@pytest.fixture
def reg(monkeypatch):
    """Install a provider table and return it (mutate to configure)."""
    table = {}
    monkeypatch.setattr(registry, "get", lambda pid: table.get(pid))
    return table


def test_web_provider_needs_accounts(reg):
    reg["deepseek"] = _Stub("deepseek", True)
    s = _Settings(provider="deepseek")
    assert pools.providers_requiring_accounts(s) == ["deepseek"]
    assert pools.any_provider_needs_accounts(s) is True
    assert pools.only_accountless_providers(s) is False


def test_local_provider_needs_nothing(reg):
    reg["opencode"] = _Stub("opencode", False)
    s = _Settings(provider="opencode")
    assert pools.providers_requiring_accounts(s) == []
    assert pools.any_provider_needs_accounts(s) is False
    # The distinction that matters: accountless, but active.
    assert pools.only_accountless_providers(s) is True


def test_mixed_set_needs_accounts(reg):
    reg["deepseek"] = _Stub("deepseek", True)
    reg["opencode"] = _Stub("opencode", False)
    s = _Settings(provider="deepseek", providers=["deepseek", "opencode"])
    assert pools.providers_requiring_accounts(s) == ["deepseek"]
    assert pools.any_provider_needs_accounts(s) is True
    # One local provider must not unblock a web provider's account prompt.
    assert pools.only_accountless_providers(s) is False


def test_unknown_provider_is_treated_as_needing_accounts(reg):
    """An unregistered id must not read as a local provider."""
    s = _Settings(provider="never-registered")
    assert pools.providers_requiring_accounts(s) == ["never-registered"]
    assert pools.only_accountless_providers(s) is False


def test_none_flag_is_read_as_accountless_so_the_contract_must_set_a_bool(reg):
    """Documents the sharp edge: `getattr(p, "requires_accounts", True)`
    only defaults when the attribute is ABSENT. A provider that sets it to
    None is treated as local, which is the dangerous direction -- so the
    Provider contract must always carry a real bool.
    """
    class _NoneFlag:
        id = "weird"
        requires_accounts = None

    reg["weird"] = _NoneFlag()
    s = _Settings(provider="weird")
    assert pools.providers_requiring_accounts(s) == []
    assert pools.only_accountless_providers(s) is True
    # The base class default is the safe one; assert it, so a future edit that
    # flips the class attribute default is caught here rather than in
    # production, where it would unblock a web provider's account prompt.
    assert Provider.requires_accounts is True


def test_registry_failure_falls_back_to_the_conservative_answer(monkeypatch):
    """A broken registry must not make a web provider look accountless.

    `providers_requiring_accounts` swallows the lookup error and returns
    every active id, so gating callers still demand an account.
    """
    def _boom(pid):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(registry, "get", _boom)
    s = _Settings(provider="deepseek")
    assert pools.providers_requiring_accounts(s) == ["deepseek"]
    assert pools.any_provider_needs_accounts(s) is True
    assert pools.only_accountless_providers(s) is False


# --- the surfaces that must not re-implement the rule ----------------------

def _code_without_docstring(fn):
    """Source minus the docstring/comments that legitimately name the flag."""
    src = inspect.getsource(fn)
    try:
        doc = fn.__doc__ or ""
    except Exception:
        doc = ""
    for line in doc.splitlines():
        src = src.replace(line, "")
    return src


def test_ui_surfaces_delegate_instead_of_copying():
    """tui and chat_ui must call the shared helper, not re-derive it.

    Asserted on the source rather than on behaviour: a third copy of the rule
    would still pass every behavioural test above. The point is that there is
    one implementation, so the check has to look at the implementation.
    """
    from deepseaport import chat_ui, tui

    for mod in (tui, chat_ui):
        fn = mod._active_providers_need_accounts
        code = _code_without_docstring(fn)
        assert "any_provider_needs_accounts" in code, (
            f"{mod.__name__}.{fn.__name__} no longer delegates to "
            f"core.pools.any_provider_needs_accounts")
        assert "requires_accounts" not in code, (
            f"{mod.__name__}.{fn.__name__} inspects requires_accounts itself "
            f"-- that is the copy this test exists to prevent")

    # tui._preflight_reason had the rule inlined too.
    pre = _code_without_docstring(tui._preflight_reason)
    assert "only_accountless_providers" in pre
    assert "requires_accounts" not in pre
