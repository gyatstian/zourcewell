"""Provider dispatch: routes execute through Provider methods (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from deepseaport.config import AccountConfig, Settings
from deepseaport.core import registry
from deepseaport.core.provider import ModelSpec, Provider
from deepseaport.server import create_app


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


def _settings(**kw) -> Settings:
    base = dict(accounts=[], provider="deepseek", config_path="")
    base.update(kw)
    return Settings(**base)


def _fake_settings() -> Settings:
    return _settings(
        providers=["deepseek", "fakep"],
        accounts=[AccountConfig(email="f@x.com", token="t", provider="fakep")],
    )


def _body(**kw) -> dict:
    body = {"model": "fakep-x", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


def test_nonstream_dispatches_through_provider(monkeypatch):
    import deepseaport.server as S

    registry.register(_FakeProvider())
    try:
        app = create_app(_fake_settings())

        def _boom(*a, **k):
            raise AssertionError("DeepSeek engine must not run for fakep")

        monkeypatch.setattr(S, "_complete_with_failover_sync", _boom)
        resp = TestClient(app).post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "P"
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_stream_dispatches_through_provider():
    registry.register(_FakeProvider())
    try:
        app = create_app(_fake_settings())
        resp = TestClient(app).post("/v1/chat/completions", json=_body(stream=True))
        assert resp.status_code == 200, resp.text
        assert "[DONE]" in resp.text
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_default_deepseek_honors_engine_monkeypatch(monkeypatch):
    import deepseaport.server as S

    app = create_app(_settings(accounts=[AccountConfig(email="a@x.com", token="tok")]))
    monkeypatch.setattr(
        S, "_complete_with_failover_sync",
        lambda *a, **k: {"content": "hi", "thinking": "", "usage_total": 5},
    )
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_unknown_provider_fallback():
    from deepseaport.server import MODELS
    from deepseaport.server_routes import _pool_for, _prepare

    s = _settings(provider="nope",
                  accounts=[AccountConfig(email="a@x.com", token="t")])
    app = create_app(s)
    data = TestClient(app).get("/v1/models").json()["data"]
    assert sorted(m["id"] for m in data) == sorted(MODELS)
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]}, s)
    assert prep["provider"] == "deepseek"
    pool = _pool_for(app, "deepseek")
    assert [r["identifier"] for r in pool.status()] == ["a@x.com"]


def test_resolved_ids():
    assert registry.resolved_ids(["nope"]) == ["deepseek"]
    assert registry.resolved_ids(["deepseek", "nope"]) == ["deepseek"]
    assert "deepseek" in registry.resolved_ids(None)
