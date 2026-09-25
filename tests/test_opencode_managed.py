"""Managed isolated opencode serve: auto boot scoped to deepseaport only."""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

from deepseaport.config import Settings
from deepseaport.providers.opencode import managed_serve as MS
from deepseaport.providers.opencode import protocol as OP
from deepseaport.providers.opencode.models import DEFAULT_MODEL, MODELS_SPEC
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare
from deepseaport.providers.opencode import client as OC
from deepseaport.providers.opencode import completion_service as OCS


def _settings(**kw) -> Settings:
    base = dict(accounts=[], provider="opencode", config_path="")
    base.update(kw)
    return Settings(**base)


def test_default_agent_is_passthrough():
    assert OP.DEFAULT_AGENT == MS.MANAGED_AGENT == "deepseaport-passthrough"
    assert OP.agent_name() == "deepseaport-passthrough"


def test_message_payload_carries_no_system_or_tools():
    body = OP.message_payload("opencode", "m", OP.agent_name(), "hi")
    assert set(body) == {"model", "agent", "parts"}
    assert body["agent"] == "deepseaport-passthrough"


def test_managed_config_content_is_minimal():
    cfg = json.loads(MS.managed_config_content())
    assert cfg["instructions"] == []
    assert MANAGED_AGENT_PRESENT(cfg)
    perms = cfg["agent"][MS.MANAGED_AGENT]["permission"]
    # edit stays denied; shell/read stay declared (free-tier gatekeeper
    # rejects tool-less upstream requests).
    assert perms["edit"] == "deny" and perms["read"] == "allow"
    assert perms["bash"] == "allow" and perms["webfetch"] == "deny"


def test_client_headers_identity(monkeypatch):
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_CLIENT", raising=False)
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_PROJECT", raising=False)
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_USER_AGENT", "opencode/1.18.31")
    headers = OP.client_headers("ses_abc123")
    assert headers["User-Agent"] == "opencode/1.18.31"
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-project"] == "global"
    assert headers["X-Session-ID"] == "ses_abc123"
    assert "X-Session-ID" not in OP.client_headers(None)


def test_request_sends_identity_headers(monkeypatch):
    from deepseaport.providers.opencode import client as OC

    seen = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"healthy": true}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        seen.update({k.lower(): v for k, v in req.header_items()})
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_USER_AGENT", "opencode/1.18.31")
    assert OC.health(timeout=2) is True
    assert seen.get("x-opencode-client") == "cli"
    assert seen.get("x-opencode-project") == "global"
    assert seen.get("user-agent") == "opencode/1.18.31"


def test_free_tier_denied_is_terminal(monkeypatch):
    from deepseaport.providers.opencode import client as OC
    from deepseaport.providers.opencode import completion_service as OCS

    gatekeeper = ("Error from provider (Console): OpenCode's free tier can "
                  "only be used from within OpenCode")
    monkeypatch.setattr(
        OC, "_request",
        lambda method, path, body=None, timeout=20, session_id=None:
        (200, {"info": {"error": gatekeeper, "finish": "error"},
               "parts": []}))
    try:
        OC.send_message("ses_1", "opencode", "m", "agent", "hi", timeout=5)
        raise AssertionError("must raise")
    except RuntimeError as exc:
        assert "OPENCODE_FREE_TIER_DENIED" in str(exc)
        assert OCS.classify_error(str(exc)) is OCS.RetryKind.UPSTREAM


def MANAGED_AGENT_PRESENT(cfg: dict) -> bool:
    return MS.MANAGED_AGENT in cfg.get("agent", {})


def test_strip_plugin_replaces_in_place_with_single_entry():
    assert "experimental.chat.system.transform" in MS.STRIP_PLUGIN_JS
    assert "splice(" in MS.STRIP_PLUGIN_JS
    assert "output.system =" not in MS.STRIP_PLUGIN_JS


def test_isolated_files_stay_off_real_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    base = tmp_path / "managed"
    paths = MS.write_isolated_files(base)
    assert (paths["cfg"] / "plugins" / "strip-system.js").exists()
    assert (paths["work"]).is_dir()
    assert not (tmp_path / "fake-home" / ".config").exists()


def test_managed_disabled_flag(monkeypatch):
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_MANAGED", "0")
    assert MS.managed_enabled() is False
    assert MS.ensure() is None


def test_explicit_user_base_skips_managed(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_BASE", "http://127.0.0.1:4999")
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_MANAGED", raising=False)
    # Must not spawn: fail loudly if it tries.
    monkeypatch.setattr("shutil.which", lambda name: (_ for _ in ()).throw(
        AssertionError("must not spawn for user-owned serve")))
    assert MS.ensure() is None


def test_config_hash_tracks_content(monkeypatch):
    assert MS._config_hash() == MS._config_hash() != ""
    monkeypatch.setattr(MS, "managed_config_content", lambda: '{"other": true}')
    assert MS._config_hash() != ""


def test_port_file_roundtrip_with_hash(tmp_path):
    pf = tmp_path / "serve.json"
    MS._write_port_file(pf, "http://127.0.0.1:9", "pw", 1234, "abc123")
    assert MS._read_port_file(pf) == ("http://127.0.0.1:9", "pw", 1234, "abc123")


def test_ensure_reuses_healthy_same_hash_without_spawning(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_BASE", raising=False)
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_MANAGED", raising=False)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    MS._managed_base = ""
    MS._managed_password = ""
    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    MS._write_port_file(tmp_path / "m" / "serve.json",
                        "http://127.0.0.1:9", "pw", 999999, MS._config_hash())
    monkeypatch.setattr(MS, "_probe", lambda base, pw, timeout=2.0: True)
    monkeypatch.setattr("shutil.which", lambda name: (_ for _ in ()).throw(
        AssertionError("healthy reuse must not spawn")))
    assert MS.ensure() == "http://127.0.0.1:9"


def test_ensure_restarts_on_hash_mismatch(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_BASE", raising=False)
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_MANAGED", raising=False)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m2")
    MS._managed_base = ""
    MS._managed_password = ""
    (tmp_path / "m2").mkdir(parents=True, exist_ok=True)
    MS._write_port_file(tmp_path / "m2" / "serve.json",
                        "http://127.0.0.1:9", "pw", 999999, "stale-hash")
    monkeypatch.setattr(MS, "_probe",
                        lambda base, pw, timeout=2.0: base == "http://127.0.0.1:9")
    killed = []
    monkeypatch.setattr(MS, "_kill_pid", lambda pid: killed.append(pid))
    monkeypatch.setattr("shutil.which", lambda name: None)  # no spawn after kill
    assert MS.ensure() is None
    assert killed == [999999]


def test_models_flagged_thinking():
    assert MODELS_SPEC
    assert all(spec.thinking is True for spec in MODELS_SPEC.values())


def test_payload_variant_only_when_set():
    assert "variant" not in OP.message_payload("opencode", "m", "a", "hi")
    body = OP.message_payload("opencode", "m", "a", "hi", variant="medium")
    assert body["variant"] == "medium"


def test_prepare_carries_reasoning_effort():
    body = {"model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "Medium"}
    assert _prepare(body, _settings())["reasoning_effort"] == "medium"
    dict_form = {"model": DEFAULT_MODEL,
                 "messages": [{"role": "user", "content": "hi"}],
                 "reasoning": {"effort": "high"}}
    assert _prepare(dict_form, _settings())["reasoning_effort"] == "high"
    bad = {"model": DEFAULT_MODEL,
           "messages": [{"role": "user", "content": "hi"}],
           "reasoning_effort": "ultra"}
    prep = _prepare(bad, _settings())
    assert prep["reasoning_effort"] is None
    assert prep["unsupported_params"]["reasoning_effort"] == "ultra"


def test_end_to_end_variant_forwarded(monkeypatch):
    seen = {}

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None,
              **kw):
        seen["variant"] = variant
        return {"content": "ok", "thinking": "r",
                "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_v")
    monkeypatch.setattr(OC, "send_message", _send)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    app = create_app(_settings())
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "medium"})
    assert resp.status_code == 200, resp.text
    assert seen["variant"] == "medium"
    assert resp.json()["choices"][0]["message"]["reasoning_content"] == "r"


def test_variant_rejection_falls_back_without(monkeypatch):
    calls = []

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None,
              **kw):
        calls.append(variant)
        if variant == "medium":
            raise RuntimeError("OPENCODE_HTTP_400: unknown variant medium")
        return {"content": "ok", "thinking": "",
                "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "create_session",
                        lambda timeout=20, **kw: "ses_vf")
    monkeypatch.setattr(OC, "send_message", _send)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    content, think, error, usage = OCS._attempt(
        "hi", DEFAULT_MODEL, 30, variant="medium")
    assert error is None and content == "ok"
    assert calls == ["medium", None]


def test_ensure_without_binary_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_BASE", raising=False)
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_MANAGED", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    assert MS.ensure() is None
    assert os.environ.get("DEEPSEAPORT_OPENCODE_BASE", "") == ""


def test_native_tools_omitted_without_pi_tools():
    assert OCS._native_tools_for({}) is None
    assert OCS._native_tools_for({"tools": []}) is None
    assert OCS._native_tools_for(None) is None
    minimal = OCS._native_tools_for({"tools": [{"type": "function"}]})
    assert isinstance(minimal, dict) and minimal.get("bash") is True
    assert minimal.get("read") is True and minimal.get("edit") is False


def test_send_message_omits_tools_key_when_none(monkeypatch):
    seen = {}

    def _fake_request(method, path, body=None, timeout=20, session_id=None,
                      **kw):
        seen["body"] = body
        return 200, {"info": {"finish": "stop"},
                     "parts": [{"type": "text", "text": "hi"}]}

    monkeypatch.setattr(OC, "_request", _fake_request)
    out = OC.send_message("ses_1", "opencode", "m", "agent", "hi",
                          timeout=5, tools=None)
    assert out["content"] == "hi"
    assert "tools" not in seen["body"]
    out2 = OC.send_message("ses_1", "opencode", "m", "agent", "hi",
                           timeout=5)
    assert "tools" in seen["body"] and seen["body"]["tools"]["bash"] is True


def test_attempt_uses_omit_for_plain_chat(monkeypatch):
    seen = {}
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_o")

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None,
              tools=None, **kw):
        seen["tools"] = tools
        return {"content": "ok", "thinking": "",
                "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "send_message", _send)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    app = create_app(_settings())
    prep = {"prompt": "hi", "model": DEFAULT_MODEL, "tools": [],
            "reasoning_effort": None}
    out = OCS._run_completion_core(app, None, prep)
    assert out["content"] == "ok" and seen["tools"] is None
    prep_tools = {"prompt": "hi", "model": DEFAULT_MODEL,
                  "tools": [{"type": "function"}], "reasoning_effort": None}
    OCS._run_completion_core(app, None, prep_tools)
    assert isinstance(seen["tools"], dict) and seen["tools"]["bash"] is True


def test_managed_prompt_forbids_native_and_allows_dsml():
    cfg = json.loads(MS.managed_config_content())
    prompt = cfg["agent"][MS.MANAGED_AGENT]["prompt"]
    assert "never execute native" in prompt.lower()
    assert "DSML" in prompt
    assert "harness" in prompt.lower()
