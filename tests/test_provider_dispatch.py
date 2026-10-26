"""Provider dispatch: routes execute through Provider methods (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from zourcewell.config import AccountConfig, Settings
from zourcewell.core import registry
from zourcewell.core.provider import ModelSpec, Provider
from support import settings
from functools import partial
from zourcewell.server import create_app


class _FakeProvider(Provider):
    """Minimal provider overriding the route seam methods."""

    id = "fakep"
    display_name = "FakeP"

    def models(self) -> dict[str, ModelSpec]:
        return {"fakep-x": ModelSpec(id="fakep-x", owned_by="fakep")}

    def complete_request(self, app, pool, prep, item, cancel_event):
        return {"content": "P", "thinking": "", "usage_total": 1}

    async def stream_response(self, app, pool, item, prep):
        yield b'data: {"choices": [{"delta": {"content": "P"}}]}\n\n'
        yield b"data: [DONE]\n\n"


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_settings = partial(settings, log_level="INFO", warmup_on_startup=True)


def _fake_settings() -> Settings:
    return _settings(
        providers=["deepseek", "fakep"],
        accounts=[AccountConfig(email="f@x.com", token="t", provider="fakep")],
    )


def _body(**kw) -> dict:
    body = {"model": "fakep-x", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


def _register_fake() -> None:
    fake = _FakeProvider()
    registry._REGISTRY[registry._key(fake)] = fake


def test_nonstream_dispatches_through_provider(monkeypatch):
    import zourcewell.providers.deepseek.completion_service as CS

    _register_fake()
    try:
        app = create_app(_fake_settings())

        def _boom(*a, **k):
            raise AssertionError("DeepSeek engine must not run for fakep")

        monkeypatch.setattr(CS, "_complete_with_failover_sync", _boom)
        resp = TestClient(app).post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "P"
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_stream_dispatches_through_provider():
    _register_fake()
    try:
        app = create_app(_fake_settings())
        resp = TestClient(app).post("/v1/chat/completions", json=_body(stream=True))
        assert resp.status_code == 200, resp.text
        assert "[DONE]" in resp.text
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_default_deepseek_honors_engine_monkeypatch(monkeypatch):
    import zourcewell.providers.deepseek.completion_service as CS

    app = create_app(_settings(accounts=[AccountConfig(email="a@x.com", token="tok")]))
    monkeypatch.setattr(
        CS, "_complete_with_failover_sync",
        lambda *a, **k: {"content": "hi", "thinking": "", "usage_total": 5},
    )
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_pool_never_borrows_accounts_for_an_unregistered_provider():
    """Defense-in-depth pin, kept though ``load_settings`` makes it unreachable.

    Model resolution still falls back to deepseek (above); the pool must NOT.
    The rule the account routes and ``accounts.usable_provider`` already
    follow: an unregistered provider id never BORROWS another provider's
    accounts. That lend handed provider X a pool of provider Y's credentials
    -- a Qwen ES256 JWT replayed to chat.deepseek.com as a userToken. No
    bound accounts means no pool, not a cross-provider borrow.

    Reaching this state needs a hand-built Settings: ``load_settings`` names
    the bad id and substitutes deepseek at load time
    (tests/test_config_integrity.py). The pin stays because the pool
    construction itself must never re-open the borrow for a caller that skips
    that normalization.
    """
    from zourcewell.server import MODELS
    from zourcewell.server_routes import _pool_for, _prepare

    s = _settings(provider="nope",
                  accounts=[AccountConfig(email="a@x.com", token="t")])
    app = create_app(s)
    data = TestClient(app).get("/v1/models").json()["data"]
    assert sorted(m["id"] for m in data) == sorted(MODELS)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]}, s)
    assert prep["provider"] == "deepseek"
    pool = _pool_for(app, "deepseek")
    assert [r["identifier"] for r in pool.status()] == [], (
        "an unregistered selected provider still borrowed the deepseek account")


def test_resolved_ids():
    assert registry.resolved_ids(["nope"]) == ["deepseek"]
    assert registry.resolved_ids(["deepseek", "nope"]) == ["deepseek"]
    assert "deepseek" in registry.resolved_ids(None)
