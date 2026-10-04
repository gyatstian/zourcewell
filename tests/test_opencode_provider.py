"""Opencode provider: local CLI via `opencode serve` (no accounts, no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deepseaport.config import AccountConfig, Settings
from deepseaport.core import registry
from deepseaport.providers.opencode import client as OC
from deepseaport.providers.opencode import completion_service as OCS
from deepseaport.providers.opencode import protocol as OP
from deepseaport.providers.opencode.models import DEFAULT_MODEL
from support import settings
from functools import partial
from deepseaport.server import create_app


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_settings = partial(settings, provider="opencode", log_level="INFO",
                     warmup_on_startup=True)


def _patch_serve(monkeypatch, content: str = "hi", thinking: str = "",
                 usage: int = 3):
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_1")
    monkeypatch.setattr(
        OC, "send_message",
        lambda sid, pid, mid, agent, prompt, timeout=180, variant=None,
        **kw: {"content": content, "thinking": thinking,
               "usage_total": usage, "finish": "stop"})
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)


def test_registered_with_default_model():
    prov = registry.get("opencode")
    assert prov is not None
    assert prov.id == "opencode"
    assert prov.default_model() == DEFAULT_MODEL
    assert DEFAULT_MODEL in prov.models()
    assert prov.requires_accounts is False
    assert prov.needs_bridge is False
    assert set(prov.models()) == {
        "opencode/muse-spark-1.3-contributor-free",
        "opencode/muse-spark-1.2-contributor-free",
        "opencode/ling-3.0-flash-fin-free",
        "opencode/space-bunny-free",
        "opencode/mimo-v2.6-flash-free",
        "opencode/nemotron-3-ultra-free",
        "opencode/nemotron-3.5-lightning-free",
        "opencode/big-pickle",
    }


def test_model_spec_exact_and_passthrough():
    prov = registry.get("opencode")
    assert prov.model_spec(DEFAULT_MODEL).id == DEFAULT_MODEL
    assert prov.model_spec("  OPENCODE/MUSE-SPARK-1.3-CONTRIBUTOR-FREE  ") is not None
    # Any provider/model passes through instead of 404ing.
    spec = prov.model_spec("anthropic/claude-sonnet-4")
    assert spec is not None and spec.id == "anthropic/claude-sonnet-4"
    assert prov.accepts_model("openrouter/anthropic/claude-x")
    # Bare ids outside the catalogue stay unknown (typo guard).
    assert prov.model_spec("gpt-99") is None
    assert not prov.accepts_model("gpt-99")
    assert prov.model_spec("") is None
    assert prov.model_spec("   ") is None


def test_split_model():
    assert OP.split_model("opencode/muse-spark-1.3-contributor-free") == (
        "opencode", "muse-spark-1.3-contributor-free")
    # First slash only: model part may contain slashes.
    assert OP.split_model("openrouter/anthropic/claude-x") == (
        "openrouter", "anthropic/claude-x")
    assert OP.split_model("bare-model") == ("opencode", "bare-model")
    assert OP.split_model("") == ("opencode", "muse-spark-1.3-contributor-free")


def test_message_payload_shape():
    body = OP.message_payload("opencode", "m", "build", "hello")
    assert body["model"] == {"providerID": "opencode", "modelID": "m"}
    assert body["agent"] == "build"
    assert body["parts"] == [{"type": "text", "text": "hello"}]


def test_classify_error():
    assert OCS.classify_error(None) is None
    assert OCS.classify_error("") is None
    assert OCS.classify_error("OPENCODE_SERVE_DOWN: refused") is OCS.RetryKind.UPSTREAM
    assert OCS.classify_error("connection refused") is OCS.RetryKind.UPSTREAM
    assert OCS.classify_error("HTTP 429 rate_limit_exceeded") is OCS.RetryKind.RATE_LIMIT
    assert OCS.classify_error("HTTP_422 unprocessable") is OCS.RetryKind.UPSTREAM_INVALID
    assert OCS.classify_error("OPENCODE_EMPTY: empty completion") is OCS.RetryKind.OVERLOAD
    assert OCS.classify_error("OPENCODE_HTTP_500: message failed") is OCS.RetryKind.OVERLOAD
    assert OCS.classify_error("OPENCODE_HTTP_503: message failed") is OCS.RetryKind.OVERLOAD
    assert OCS.classify_error("fetch failed") is OCS.RetryKind.OVERLOAD
    assert OCS.classify_error("upstream timed out") is OCS.RetryKind.OVERLOAD
    assert OCS.classify_error("weird boom") is OCS.RetryKind.UPSTREAM


def test_no_account_seams():
    prov = registry.get("opencode")
    assert prov.probe_ban({}, timeout=5) == (False, None)
    assert prov.normalize_token('{"value":"tok64"}') == "tok64"
    with pytest.raises(NotImplementedError):
        prov.login("a@x.com", "pw", None)


def test_warmup_never_raises(monkeypatch):
    from deepseaport.providers.opencode import managed_serve as MS

    prov = registry.get("opencode")
    # Managed boot is stubbed: unit tests must never spawn real processes.
    monkeypatch.setattr(MS, "ensure", lambda timeout=30.0: None)
    monkeypatch.setattr(OC, "health", lambda timeout=10: False)
    assert prov.warmup(None) is None
    monkeypatch.setattr(OC, "health", lambda timeout=10: True)
    assert prov.warmup(None) is None


def test_preflight_passes_with_zero_accounts():
    from deepseaport.tui import _preflight_reason

    assert _preflight_reason(_settings()) == ""


def test_nonstream_needs_no_pool(monkeypatch):
    _patch_serve(monkeypatch)
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_nonstream_passthrough_model(monkeypatch):
    seen = {}

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None,
                **kw):
        seen.update(pid=pid, mid=mid, prompt=prompt, variant=variant)
        return {"content": "ok", "thinking": "", "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_9")
    monkeypatch.setattr(OC, "send_message", _send)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "anthropic/claude-x",
        "messages": [{"role": "user", "content": "ping"}]})
    assert resp.status_code == 200, resp.text
    assert (seen["pid"], seen["mid"]) == ("anthropic", "claude-x")
    assert "ping" in seen["prompt"]


def test_stream_needs_no_pool(monkeypatch):
    _patch_serve(monkeypatch, content="hello stream")
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}],
        "stream": True})
    assert resp.status_code == 200, resp.text
    assert "hello stream" in resp.text
    assert "[DONE]" in resp.text


def test_200_with_error_envelope_is_502_not_content(monkeypatch):
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_1")
    monkeypatch.setattr(
        OC, "_request",
        lambda method, path, body=None, timeout=20, session_id=None, **kw:
        (200, {"info": {"error": "Connection error.", "finish": "error"},
               "parts": []}))
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502, resp.text
    assert "agent loop failed" in resp.text


def test_serve_down_is_502_with_hint(monkeypatch):
    from deepseaport.providers.opencode import managed_serve as MS

    def _down(timeout=20, **kw):
        raise RuntimeError("OPENCODE_SERVE_DOWN: refused. start serve first")

    monkeypatch.setattr(OC, "create_session", _down)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    # Managed boot unavailable here: the 502 path (not a real spawn) is under test.
    monkeypatch.setattr(MS, "ensure", lambda timeout=30.0: None)
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502, resp.text
    assert "opencode serve down" in resp.text.lower()


def test_models_union_includes_opencode():
    app = create_app(_settings())
    ids = [m["id"] for m in TestClient(app).get("/v1/models").json()["data"]]
    assert DEFAULT_MODEL in ids


def test_bridge_flags():
    from deepseaport.providers.deepseek.provider import DeepSeekProvider
    from deepseaport.providers.qwen.provider import QwenProvider

    assert registry.get("opencode").needs_bridge is False
    assert DeepSeekProvider.needs_bridge is True
    assert QwenProvider.needs_bridge is True


def test_startup_warm_skips_obscura_for_opencode_only(monkeypatch):
    from deepseaport import server_concurrency as SC

    calls = {"bridge": 0, "health": 0}
    monkeypatch.setattr(OC, "health",
                        lambda timeout=10: calls.__setitem__("health", 1) or True)

    class _Bridge:
        def warmup(self, timeout: int = 45):
            calls["bridge"] += 1
            return {}

    SC._startup_warm(_Bridge(), _settings())
    assert calls == {"bridge": 0, "health": 1}


def test_startup_warm_keeps_obscura_for_deepseek(monkeypatch):
    from deepseaport import server_concurrency as SC
    from deepseaport.providers.deepseek.provider import DeepSeekProvider

    calls = {"bridge": 0, "provider": 0}
    monkeypatch.setattr(DeepSeekProvider, "warmup",
                        lambda self, bridge: calls.__setitem__("provider", 1))

    class _Bridge:
        def warmup(self, timeout: int = 45):
            calls["bridge"] += 1
            return {}

    s = _settings(provider="deepseek",
                  accounts=[AccountConfig(email="a@x.com", token="t1")])
    SC._startup_warm(_Bridge(), s)
    assert calls["bridge"] == 1
    assert calls["provider"] == 1
