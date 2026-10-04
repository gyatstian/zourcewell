"""Phase 4: provider-bound accounts and provider-scoped pools."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deepseaport.config import AccountConfig, Settings, load_settings
from deepseaport.core import pools, registry
from deepseaport.core.provider import ModelSpec, Provider
from deepseaport.server import create_app
from support import settings
from functools import partial
from deepseaport.server_routes import _pool, _pool_for, _prepare


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


def test_active_provider_ids_single_vs_multi():
    assert pools.active_provider_ids(_settings(provider="deepseek")) == ["deepseek"]
    assert pools.active_provider_ids(_settings(providers=[])) == ["deepseek"]
    assert pools.active_provider_ids(_settings(providers=["no-such-provider"])) == ["deepseek"]
    assert pools.active_provider_ids(
        _settings(providers=["deepseek", "qwen", "DEEPSEEK", ""])) == ["deepseek", "qwen"]
    assert pools.is_multi(_settings(provider="deepseek")) is False
    assert pools.is_multi(_settings(providers=["deepseek", "qwen"])) is True


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


# 6. single-provider seam delegation -------------------------------------------


def test_single_provider_pool_for_delegates_to_server_pool(monkeypatch):
    import deepseaport.server as S

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