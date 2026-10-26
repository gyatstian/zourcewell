"""Phase 4: provider-bound accounts and provider-scoped pools."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from zourcewell.config import AccountConfig, Settings, load_settings
from zourcewell.core import pools, registry
from zourcewell.core.provider import ModelSpec, Provider
from zourcewell.server import create_app
from support import settings
from functools import partial
from zourcewell.server_routes import _pool, _pool_for, _prepare


class _FakeProvider(Provider):
    """Minimal provider with one model; completion deliberately unsupported."""

    id = "fakep"
    display_name = "FakeP"

    def models(self) -> dict[str, ModelSpec]:
        return {"fakep-mini": ModelSpec(id="fakep-mini", owned_by="fakep")}

    def complete(self, *args, **kwargs):
        raise NotImplementedError("fake provider cannot complete")


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_settings = partial(settings, log_level="INFO", warmup_on_startup=True)


# 1. AccountConfig provider default + roundtrip ---------------------------------


def test_account_provider_default_and_roundtrip(tmp_path):
    assert AccountConfig(email="a@x.com").provider == "deepseek"
    cfg = tmp_path / "c.json"
    s = Settings(accounts=[AccountConfig(email="a@x.com", token="t1", provider="qwen")],
                 config_path=str(cfg))
    s.save()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["accounts"][0]["provider"] == "qwen"
    loaded = load_settings(str(cfg))
    assert loaded.accounts[0].provider == "qwen"


def test_config_missing_provider_migrates_to_deepseek(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"accounts": [{"email": "a@x.com", "token": "t1"}]}),
                   encoding="utf-8")
    loaded = load_settings(str(cfg))
    assert loaded.accounts[0].provider == "deepseek"


# 2. pools helpers --------------------------------------------------------------


def test_unknown_provider_entry_falls_back_and_single_is_not_multi():
    """The two verdicts test_config_integrity.py's warning tests do not pin:
    an unrecognised entry is dropped, and single mode is not multi mode."""
    assert pools.active_provider_ids(_settings(providers=["no-such-provider"])) == ["deepseek"]
    assert pools.is_multi(_settings(provider="deepseek")) is False


def test_accounts_for_provider_filters():
    s = _settings(providers=["deepseek", "qwen"], accounts=[
        AccountConfig(email="a@x.com", token="t1"),
        AccountConfig(email="b@x.com", token="t2", provider="qwen"),
        AccountConfig(email="c@x.com", token="t3", provider="QWEN"),
    ])
    assert [a.email for a in pools.accounts_for_provider(s, "deepseek")] == ["a@x.com"]
    assert [a.email for a in pools.accounts_for_provider(s, "qwen")] == ["b@x.com", "c@x.com"]


# 3. _pool filters by selected provider ----------------------------------------


def test_pool_filters_by_selected_provider():
    s = _settings(provider="qwen", accounts=[
        AccountConfig(email="a@x.com", token="t1"),
        AccountConfig(email="b@x.com", token="t2", provider="qwen"),
    ])
    app = create_app(s)
    pool = _pool(app)
    ids = [row["identifier"] for row in pool.status()]
    assert ids == ["b@x.com"]


# 4. multi-provider mode --------------------------------------------------------


def _register_fake() -> _FakeProvider:
    provider = _FakeProvider()
    registry._REGISTRY[registry._key(provider)] = provider
    return provider


def _multi_settings() -> Settings:
    return _settings(providers=["deepseek", "fakep"], accounts=[
        AccountConfig(email="ds@x.com", token="t1"),
        AccountConfig(email="fp@x.com", token="t2", provider="fakep"),
    ])


def test_multi_models_prepare_and_pools():
    _register_fake()
    try:
        s = _multi_settings()
        app = create_app(s)
        client = TestClient(app)

        data = client.get("/v1/models").json()["data"]
        ids = {m["id"] for m in data}
        assert "fakep-mini" in ids
        assert "deepseek-flash" in ids
        owners = {m["id"]: m["owned_by"] for m in data}
        assert owners["fakep-mini"] == "fakep"
        assert owners["deepseek-flash"] == "deepseek"

        prep = _prepare({"model": "fakep-mini",
                         "messages": [{"role": "user", "content": "hi"}]}, s)
        assert prep["provider"] == "fakep"

        with pytest.raises(HTTPException) as exc_info:
            _prepare({"model": "gpt-99",
                      "messages": [{"role": "user", "content": "hi"}]}, s)
        assert exc_info.value.status_code == 404

        fp_pool = _pool_for(app, "fakep")
        assert [r["identifier"] for r in fp_pool.status()] == ["fp@x.com"]
        ds_pool = _pool_for(app, "deepseek")
        assert [r["identifier"] for r in ds_pool.status()] == ["ds@x.com"]
    finally:
        registry._REGISTRY.pop("fakep", None)


# 5. POST /v1/accounts binds provider -------------------------------------------


def test_add_account_binds_provider_field():
    _register_fake()
    try:
        s = _multi_settings()
        app = create_app(s)
        client = TestClient(app)
        r = client.post("/v1/accounts", json={
            "email": "new@x.com", "token": "tok-new", "provider": "fakep"})
        assert r.status_code == 201, r.text
        added = [a for a in s.accounts if a.email == "new@x.com"]
        assert len(added) == 1 and added[0].provider == "fakep"
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_add_account_defaults_to_selected_provider():
    s = _settings(provider="deepseek", accounts=[])
    app = create_app(s)
    r = TestClient(app).post("/v1/accounts",
                             json={"email": "new@x.com", "token": "tok-new"})
    assert r.status_code == 201, r.text
    assert s.accounts[0].provider == "deepseek"


# 5b. an explicit provider must be ACTIVE, not merely registered ----------------


def test_add_account_rejects_an_inactive_registered_provider():
    """qwen is registered but not active here, and _pool_for falls back to the
    selected pool for an id outside the active set (pinned above). Accepting the
    id therefore stored a Qwen token into -- and made CURRENT in -- the DeepSeek
    pool, which then replayed it to chat.deepseek.com."""
    s = _settings(provider="deepseek",
                  accounts=[AccountConfig(email="ds@x.com", token="ds-tok")])
    app = create_app(s)
    pool = _pool(app)
    before = pool.status()

    resp = TestClient(app).post("/v1/accounts",
                                json={"provider": "qwen", "token": "qwen-jwt-abc"})

    assert resp.status_code == 400, resp.text
    assert "active" in resp.json()["detail"]
    assert [a.provider for a in s.accounts] == ["deepseek"]
    assert s.active_account == ""
    assert pool.status() == before


def test_account_mutations_reject_an_inactive_provider():
    """The same guard on the other three identifier routes: PUT/DELETE/unblock
    with a registered-but-inactive id fell through to the selected pool."""
    s = _settings(provider="deepseek",
                  accounts=[AccountConfig(email="ds@x.com", token="ds-tok")])
    app = create_app(s)
    client = TestClient(app)

    unblock = client.post("/v1/accounts/unblock",
                          json={"identifier": "ds@x.com", "provider": "qwen"})
    put = client.put("/v1/accounts/ds@x.com/token?provider=qwen",
                     json={"token": "fresh"})
    delete = client.delete("/v1/accounts/ds@x.com?provider=qwen")

    for resp in (unblock, put, delete):
        assert resp.status_code == 400, resp.text
        assert "active" in resp.json()["detail"]
    assert s.accounts[0].token == "ds-tok"
    assert len(s.accounts) == 1

    # Control: the active provider keeps working.
    ok = client.delete("/v1/accounts/ds@x.com?provider=deepseek")
    assert ok.status_code == 200, ok.text
    assert s.accounts == []


# 5c. the same email on two providers: mutations stay inside one provider ------


def _same_email_two_providers() -> Settings:
    """Two records that share an email: account_identity includes the provider,
    so load/save/dedupe all treat them as distinct accounts."""
    return _settings(providers=["deepseek", "qwen"], accounts=[
        AccountConfig(email="same@x.com", token="ds-tok", provider="deepseek"),
        AccountConfig(email="same@x.com", token="qw-tok", provider="qwen"),
    ])


def test_delete_scopes_to_the_provider_param():
    s = _same_email_two_providers()

    resp = TestClient(create_app(s)).delete(
        "/v1/accounts/same@x.com?provider=qwen")

    assert resp.status_code == 200, resp.text
    assert [(a.email, a.provider, a.token) for a in s.accounts] == [
        ("same@x.com", "deepseek", "ds-tok")], (
        "the settings-wide first match removed the deepseek record")


def test_put_token_scopes_to_the_provider_param():
    s = _same_email_two_providers()

    resp = TestClient(create_app(s)).put(
        "/v1/accounts/same@x.com/token?provider=qwen", json={"token": "fresh"})

    assert resp.status_code == 200, resp.text
    assert {a.provider: a.token for a in s.accounts} == {
        "deepseek": "ds-tok", "qwen": "fresh"}, (
        "the unscoped settings write touched the deepseek record too")


# 6. single-provider seam delegation -------------------------------------------


def test_single_provider_pool_for_delegates_to_server_pool(monkeypatch):
    import zourcewell.server as S

    sentinel = object()
    monkeypatch.setattr(S, "_pool", lambda _app: sentinel)
    app = create_app(_settings(provider="deepseek"))
    assert _pool_for(app, "deepseek") is sentinel
    assert _pool_for(app, "") is sentinel


def test_single_provider_uses_shared_pool_registry():
    """Single mode stores in app.state.pools too (one store, not two)."""
    s = _settings(provider="deepseek",
                  accounts=[AccountConfig(email="a@x.com", token="t1")])
    app = create_app(s)
    pool = _pool(app)
    assert app.state.pools == {"deepseek": pool}
    assert _pool_for(app, "deepseek") is pool


def test_multi_provider_pools_share_one_registry():
    _register_fake()
    try:
        app = create_app(_multi_settings())
        ds = _pool_for(app, "deepseek")
        fp = _pool_for(app, "fakep")
        assert app.state.pools == {"deepseek": ds, "fakep": fp}
        assert _pool(app) is ds  # selected provider is deepseek
    finally:
        registry._REGISTRY.pop("fakep", None)


# 7. a pool is never assembled out of another provider's accounts --------------
#
# `server_routes._pool` is the last place that allowed the borrow: when the
# selected id was unregistered it fell back to `list(settings.accounts)` with
# NO provider filter, so a bad id on a multi-provider config handed DeepSeek a
# pool of Qwen ES256 JWTs and replayed them to chat.deepseek.com as userTokens.
# The account routes, `accounts.usable_provider` and tests/test_live_smoke.py
# all refuse the borrow; these pin that the pool builder does too.


def test_an_unregistered_selected_provider_never_borrows_another_accounts():
    s = settings(provider="nope", accounts=[
        AccountConfig(email="ds@x.com", token="t-ds"),
        AccountConfig(email="qw@x.com", token="t-qw", provider="qwen"),
    ])

    pool = _pool(create_app(s))

    assert pool.status() == [], (
        "the unregistered provider borrowed "
        f"{[r['identifier'] for r in pool.status()]}")
    # The lend was only ever a read: nothing was taken away either.
    assert len(s.accounts) == 2


def test_a_registered_provider_with_no_bound_account_is_also_empty():
    """The sibling half: right shape, right name, still nobody else's token."""
    s = settings(provider="qwen", accounts=[AccountConfig(email="ds@x.com", token="t")])

    assert _pool(create_app(s)).status() == []


def test_the_borrow_never_reaches_the_account_routes():
    """POST /v1/accounts with an unregistered provider is a 400, not a lend."""
    s = settings(provider="nope", accounts=[AccountConfig(email="ds@x.com", token="t")])
    resp = TestClient(create_app(s)).post("/v1/accounts", json={
        "email": "new@x.com", "token": "tok-new", "provider": "glm"})

    assert resp.status_code == 400, resp.text
    assert s.accounts == [s.accounts[0]], "the rejected account was stored anyway"


# 8. needs_waf_jar: who consumes the bridge's harvested jar ---------------------
#
# The startup obscura warmup targets ONE host (CHAT_HOME, DeepSeek by default)
# and plants that host's WAF cookies in bridge.state. Only a provider that
# actually sends those cookies may trigger it: qwen harvests its own acw_tc
# cache through the bridge's engine, opencode has no WAF at all. `needs_bridge`
# stays True for qwen -- one-shot chat still builds the bridge it may use.


def test_needs_waf_jar_defaults_and_overrides():
    from zourcewell.providers.deepseek.provider import DeepSeekProvider
    from zourcewell.providers.opencode.provider import OpencodeProvider
    from zourcewell.providers.qwen.provider import QwenProvider

    assert Provider.needs_waf_jar is True
    assert DeepSeekProvider.needs_waf_jar is True
    assert QwenProvider.needs_waf_jar is False
    assert OpencodeProvider.needs_waf_jar is False
    assert QwenProvider.needs_bridge is True