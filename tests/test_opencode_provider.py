"""Opencode provider: local CLI via `opencode serve` (no accounts, no network)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from deepseaport.config import AccountConfig, Settings
from deepseaport.core import registry
from deepseaport.providers.opencode import client as OC
from deepseaport.providers.opencode import completion_service as OCS
from deepseaport.providers.opencode import protocol as OP
from deepseaport.providers.opencode import managed_serve as MS
from deepseaport.providers.opencode import models
from deepseaport.providers.opencode.models import DEFAULT_MODEL, MODELS_SPEC
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


def test_registered_with_fallback_catalogue():
    prov = registry.get("opencode")
    assert prov is not None
    assert prov.id == "opencode"
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
    assert prov.model_spec("openrouter/anthropic/claude-x") is not None
    # Bare ids outside the catalogue stay unknown (typo guard).
    assert prov.model_spec("gpt-99") is None
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


# --- the catalogue: live from the serve, static as the fallback ------------


@pytest.fixture(autouse=True)
def _isolate_model_cache():
    """The catalogue cache is MODULE state; a faked read must not leak.

    ``live_specs`` caches success AND failure process-wide (``_CACHE_TTL`` for
    a catalogue, ``_FAIL_TTL`` for a miss), so without this a faked catalogue
    would answer every later test in the run (and vice versa: a serve-down
    cached as {} would mask a fake).
    """
    saved = (models._cached, models._cached_at)
    models._cached, models._cached_at = None, 0.0
    yield
    models._cached, models._cached_at = saved


def _fake_catalogue(monkeypatch, result):
    """Stub ``client._request`` (the one HTTP seam ``live_specs`` uses)."""
    seen: list[str] = []

    def _request(method, path, body=None, timeout=20, session_id=None):
        seen.append(f"{method} {path}")
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(OC, "_request", _request)
    return seen


def test_spec_from_entry_drops_what_the_operator_cannot_pick():
    assert models._spec_from_entry("nonsense") is None
    assert models._spec_from_entry({}) is None
    assert models._spec_from_entry({"modelID": "m"}) is None
    assert models._spec_from_entry({"providerID": "p"}) is None
    # Disabled and non-active entries must not be advertised.
    assert models._spec_from_entry(
        {"providerID": "p", "modelID": "m", "enabled": False}) is None
    assert models._spec_from_entry(
        {"providerID": "p", "modelID": "m", "status": "beta"}) is None
    assert models._spec_from_entry(
        {"providerID": "p", "modelID": "m", "enabled": True,
         "status": "active"}) is not None


def test_spec_from_entry_thinking_follows_the_reasoning_field_not_variants():
    """Only a declared reasoning field means the serve can deliver thinking.

    ``variants`` alone must NOT flip it: the free zen models carry variants
    (the reasoning-effort knobs) while ``muse-spark-1.3-contributor-free``
    reports reasoning TOKENS and delivers no thinking text at all.
    """
    base = {"providerID": "zen", "modelID": "m", "status": "active"}
    plain = models._spec_from_entry(dict(base))
    assert plain.thinking is False and plain.owned_by == "zen"
    assert models._spec_from_entry(
        {**base, "compatibility": {"reasoningField": "reasoning"}}).thinking is True
    assert models._spec_from_entry(
        {**base, "compatibility": {}}).thinking is False
    assert models._spec_from_entry(
        {**base, "variants": ["low", "high"]}).thinking is False
    assert models._spec_from_entry(
        {**base, "variants": ["low"],
         "compatibility": {"reasoningField": "reasoning"}}).thinking is True
    spec = models._spec_from_entry({**base, "modelID": "sub/m"})
    assert spec.id == "zen/sub/m"


def test_live_specs_reads_the_serve_and_drops_unusable_entries(monkeypatch):
    seen = _fake_catalogue(monkeypatch, (200, {"data": [
        {"providerID": "zen", "modelID": "muse", "status": "active",
         "compatibility": {"reasoningField": "reasoning"}},
        {"providerID": "zen", "modelID": "off", "enabled": False},
        {"providerID": "zen", "modelID": "beta", "status": "beta"},
        "junk",
    ]}))
    specs = models.live_specs()
    assert seen == ["GET /api/model"], seen
    assert list(specs) == ["zen/muse"], specs
    assert specs["zen/muse"].thinking is True
    # The cached copy is handed out, not the caller's dict.
    specs["zen/muse"] = None
    assert models.live_specs()["zen/muse"] is not None
    assert seen == ["GET /api/model"], "the second read must come from cache"


def test_live_specs_failure_is_cached_and_never_raises(monkeypatch):
    down = _fake_catalogue(monkeypatch, RuntimeError(
        "OPENCODE_SERVE_DOWN: refused"))
    assert models.live_specs() == {}
    assert models.live_specs() == {}
    assert down == ["GET /api/model"], "a failure must be cached for the TTL too"

    # Junk and empty bodies are the same signal, not a partial catalogue.
    for junk in ((200, {}), (200, {"data": []}), (200, {"data": "nope"}),
                 (404, {"data": [{"providerID": "p", "modelID": "m"}]}), (200, [])):
        models._cached, models._cached_at = None, 0.0
        _fake_catalogue(monkeypatch, junk)
        assert models.live_specs() == {}, junk


def test_live_specs_failure_expires_before_a_catalogue_does(monkeypatch):
    """A booting serve must be re-read soon; a healthy one may be trusted.

    ``/api/model`` is slow on the first read of a booting serve, so caching a
    FAILED read for the full catalogue TTL would leave /v1/models on the
    static set for minutes right after the managed instance came up.
    """
    good = [{"providerID": "zen", "modelID": "m", "status": "active"}]
    down = _fake_catalogue(monkeypatch, RuntimeError("OPENCODE_SERVE_DOWN"))
    assert models.live_specs() == {}
    models._cached_at = time.monotonic() - models._FAIL_TTL - 1
    _fake_catalogue(monkeypatch, (200, {"data": good}))
    assert list(models.live_specs()) == ["zen/m"]
    # A real catalogue, by contrast, is reused for the long TTL.
    up = _fake_catalogue(monkeypatch, (200, {"data": good}))
    models._cached_at = time.monotonic() - models._FAIL_TTL - 1
    assert list(models.live_specs()) == ["zen/m"]
    assert up == [], "a cached catalogue must not be re-read inside _CACHE_TTL"


def test_live_specs_boots_the_managed_serve_when_nothing_answers(monkeypatch):
    """The list must not depend on which process asks for it.

    Pre-change, any process that had not already booted a serve -- a fresh
    ``deepseaport models``, the TUI before warmup -- read nothing and fell
    back to the hand-written eight, so the same installation reported
    different ids depending on who asked.
    """
    good = [{"providerID": "zen", "modelID": "m", "status": "active"}]
    reads: list[float] = []

    def _request(method, path, body=None, timeout=20, session_id=None):
        reads.append(timeout)
        if len(reads) == 1:
            raise RuntimeError("OPENCODE_SERVE_DOWN: nothing is listening")
        return 200, {"data": good}

    monkeypatch.setattr(OC, "_request", _request, raising=False)
    monkeypatch.setattr(MS, "ensure", lambda timeout=30.0: "http://127.0.0.1:49999")

    assert list(models.live_specs()) == ["zen/m"]
    assert len(reads) == 2, "the boot must be followed by a second read"
    assert reads[1] >= models._BOOT_READ_TIMEOUT, (
        "a serve that just booted needs the long budget: it builds its model "
        "registry on the first /api/model read")

    # Nothing to boot (operator's own serve is down, or the managed one is
    # disabled): one read, then the static fallback. Never a second attempt.
    models._cached, models._cached_at = None, 0.0
    reads.clear()
    monkeypatch.setattr(MS, "ensure", lambda timeout=30.0: None)
    assert models.live_specs() == {}
    assert len(reads) == 1, reads


def test_provider_models_prefers_live_over_static(monkeypatch):
    prov = registry.get("opencode")
    _fake_catalogue(monkeypatch, (200, {"data": [
        {"providerID": "openrouter", "modelID": "anthropic/claude-x",
         "status": "active", "compatibility": {"reasoningField": "reasoning"}},
    ]}))
    ids = list(prov.models())
    # The live set REPLACES the static eight (upstream order), it is not merged.
    assert ids == ["openrouter/anthropic/claude-x"], ids
    spec = prov.model_spec("openrouter/anthropic/claude-x")
    assert spec is not None and spec.thinking is True and spec.owned_by == "openrouter"
    # The static set is a fallback, not a floor: with a live catalogue that
    # omits the default, the id still resolves -- as a passthrough spec.
    assert prov.model_spec(DEFAULT_MODEL).thinking is False


def test_provider_models_falls_back_when_the_serve_is_down(monkeypatch):
    prov = registry.get("opencode")
    _fake_catalogue(monkeypatch, RuntimeError("OPENCODE_SERVE_DOWN: refused"))
    assert dict(prov.models()) == MODELS_SPEC
    assert prov.model_spec(DEFAULT_MODEL).thinking is True
    # A provider/model the serve cannot advertise still passes through.
    assert prov.model_spec("anthropic/claude-x").id == "anthropic/claude-x"


def test_models_hint_is_in_process_only(monkeypatch):
    """A menu must not read, connect or boot for a list of model ids.

    ``models()`` reads the serve and BOOTS the managed instance when nothing
    answers; a picker row called it once, and the opencode row landed after the
    other two by that boot plus a failed connect to the dead base (a refused
    loopback connect itself costs seconds on some hosts).
    """
    prov = registry.get("opencode")
    boots: list[float] = []

    def _ensure(timeout=30.0):
        boots.append(timeout)
        return None

    monkeypatch.setattr(MS, "ensure", _ensure)

    def _no_io(*a, **k):
        raise AssertionError("models_hint() performed I/O")

    # Nothing read yet: the static fallback, no HTTP, no boot.
    monkeypatch.setattr(OC, "_request", _no_io)
    assert dict(prov.models_hint()) == MODELS_SPEC

    # What another caller already read is what the row shows.
    monkeypatch.setattr(OC, "_request",
                        lambda *a, **k: (200, {"data": [
                            {"providerID": "zen", "modelID": "m",
                             "status": "active"}]}))
    assert list(models.live_specs()) == ["zen/m"]
    monkeypatch.setattr(OC, "_request", _no_io)
    assert list(prov.models_hint()) == ["zen/m"]
    assert boots == [], "a menu row booted the managed serve"


def test_models_hint_never_waits_on_a_read_that_holds_the_lock():
    """The lock is held ACROSS a fetch and a boot, so the hint must not take it.

    A menu drawn while the warmup thread is booting (or a route is reading)
    would otherwise inherit that wait -- up to ``_BOOT_READ_TIMEOUT``. Stale is
    acceptable; blocked is not.
    """
    import threading

    spec = models.ModelSpec(id="zen/m")
    models._cached = {"zen/m": spec}
    models._cache_lock.acquire()
    try:
        got: list = []
        thread = threading.Thread(target=lambda: got.append(models.cached_specs()),
                                  daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "cached_specs() waited on a held lock"
        assert list(got[0]) == ["zen/m"]
    finally:
        models._cache_lock.release()


def test_message_payload_shape():
    """v2 prompt body: the text, and nothing else.

    The model, its reasoning ``variant`` and the agent are SESSION-scoped
    (``POST /api/session/:id/model`` and ``.../agent``) and native tools have
    no per-message map at all, so the message body carries no system, no
    model, no agent and no ``tools`` key to declare.
    """
    assert OP.message_payload("hello") == {"text": "hello"}
    assert set(OP.message_payload("hello")) == {"text"}


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


def test_live_variant_resend(monkeypatch):
    """The live path resends without a rejected variant, never falls back twice.

    A variant refusal is not an SSE failure: when the resend itself fails the
    error is returned as-is (a blocking fallback here would be a third turn on
    a new session); when it succeeds it is the attempt that answers.
    """
    calls: list = []
    blocking: list = []
    fail_resend = {"on": True}

    def _stream(sid, pid, mid, agent, prompt, timeout=180, variant=None, **kw):
        calls.append(variant)
        if variant == "medium":
            raise RuntimeError("OPENCODE_UPSTREAM: Variant unavailable for m")
        if fail_resend["on"]:
            raise RuntimeError("OPENCODE_UPSTREAM: resend failed")
        return {"content": "ok", "thinking": "",
                "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_vl")
    monkeypatch.setattr(OC, "stream_message", _stream)
    monkeypatch.setattr(OC, "send_message",
                        lambda *a, **kw: blocking.append(1) or {
                            "content": "blocking", "thinking": "",
                            "usage_total": 1, "finish": "stop"})
    monkeypatch.setattr(OC, "delete_session", lambda sid, timeout=10, **kw: None)

    content, _think, error, _usage = OCS._attempt_live(
        "hi", DEFAULT_MODEL, 30, variant="medium")
    assert content == "" and "resend failed" in (error or "")
    assert blocking == [] and calls == ["medium", None]

    fail_resend["on"] = False
    calls.clear()
    assert OCS._attempt_live("hi", DEFAULT_MODEL, 30, variant="medium") == (
        "ok", "", None, 1)
    assert calls == ["medium", None]


def test_turn_outcome_only_reads_items_after_the_prompt_id():
    """A resend on the same session must not answer with the previous turn.

    v2's variant-less resend (completion_service re-sends without the
    ``variant``) reuses the session, so the timeline still carries the first
    attempt's completed answer ahead of the new user item. Only what follows
    the id the prompt returned is this turn.
    """
    done = {"id": "msg_a1", "type": "assistant",
            "time": {"created": 1.0, "completed": 2.0},
            "content": [{"type": "text", "text": "FIRST ATTEMPT"}],
            "finish": "stop", "tokens": {"output": 3}}
    user = {"id": "msg_u2", "type": "user", "time": {"created": 3.0}}

    assert OC._turn_outcome([done, user], "msg_u2", "p", "m") == ("pending", "")
    # With no id to anchor on, the newest completed answer is still resolved.
    assert OC._turn_outcome([done], "", "p", "m") == ("done", {
        "content": "FIRST ATTEMPT", "thinking": "", "usage_total": 3,
        "finish": "stop"})


def test_200_with_error_envelope_is_502_not_content(monkeypatch):
    """A turn that fails on the timeline is a 502, never its content.

    v2 answers the prompt 200 at once and runs the agent loop in the
    background, so the upstream failure rides the assistant item rather than
    a status code on the POST.
    """
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_1")
    monkeypatch.setattr(
        OC, "_request",
        lambda method, path, body=None, timeout=20, session_id=None, **kw:
        (200, {"data": {"id": "msg_u"}}) if path.endswith("/prompt") else (204, {}))
    monkeypatch.setattr(
        OC, "get_session_messages",
        lambda sid, timeout=20, **kw: [
            {"id": "msg_u", "type": "user",
             "time": {"created": 1.0, "completed": 1.0}, "text": "hi"},
            {"id": "msg_a", "type": "assistant",
             "time": {"created": 1.0, "completed": 2.0}, "content": [],
             "finish": "error", "tokens": {},
             "error": {"type": "ProviderAuthError", "message": "Connection error."}},
        ])
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
