"""Managed isolated opencode serve: auto boot scoped to deepseaport only."""

from __future__ import annotations

import json
import os
import subprocess
import threading

import pytest
from fastapi.testclient import TestClient

from deepseaport.config import Settings
from deepseaport.providers.opencode import managed_serve as MS
from deepseaport.providers.opencode import protocol as OP
from deepseaport.providers.opencode.models import DEFAULT_MODEL, MODELS_SPEC
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare
from deepseaport.providers.opencode import client as OC
from support import settings
from functools import partial
from deepseaport.providers.opencode import completion_service as OCS


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_settings = partial(settings, provider="opencode", log_level="INFO",
                     warmup_on_startup=True)


def test_default_agent_is_passthrough():
    assert OP.DEFAULT_AGENT == MS.MANAGED_AGENT == "deepseaport-passthrough"
    assert OP.agent_name() == "deepseaport-passthrough"


def _fake_v2_serve(monkeypatch, items: list, prompt_id: str = "msg_u") -> list:
    """v2 `_request` fake: session-scoped select + prompt; timeline via `items`.

    Returns the ``(path, body)`` log of everything that went out, so a test can
    assert WHAT rides the wire (v2 moved the model, its reasoning ``variant``
    and the agent off the message body and onto the session).
    """
    bodies: list = []

    def _fake_request(method, path, body=None, timeout=20, session_id=None, **kw):
        bodies.append((path, body))
        if path.endswith("/prompt"):
            return 200, {"data": {"id": prompt_id}}
        return 204, {}

    monkeypatch.setattr(OC, "_request", _fake_request)
    monkeypatch.setattr(OC, "get_session_messages",
                        lambda sid, timeout=20, **kw: list(items))
    return bodies


def _assistant_item(text: str = "hi", thinking: str = "", usage: int = 3,
                    error=None) -> dict:
    """One v2 timeline assistant item (content parts, tokens, error)."""
    item = {"id": "msg_a", "type": "assistant", "agent": OP.agent_name(),
            "time": {"created": 1.0, "streamed": 2.0, "completed": 3.0},
            "content": [{"type": "reasoning", "text": thinking},
                        {"type": "text", "text": text}],
            "finish": "error" if error else "stop",
            "tokens": {"input": 2, "output": usage}}
    if error is not None:
        item["error"] = error
    return item


def test_managed_config_content_is_minimal():
    cfg = json.loads(MS.managed_config_content())
    assert cfg["instructions"] == []
    assert MANAGED_AGENT_PRESENT(cfg)
    perms = cfg["agent"][MS.MANAGED_AGENT]["permission"]
    # every native tool is denied; the map is the only tool boundary on v2.
    assert perms["read"] == "deny" and perms["edit"] == "deny"
    assert perms["bash"] == "deny" and perms["webfetch"] == "deny"
    assert perms["external_directory"] == "deny"
    # The agent-level map mirrors the top-level one (both are sent).
    assert cfg["permission"] == perms


def test_managed_config_denies_native_shell():
    """The map granted bash while the comment claimed it was unused.

    The child is an agentic CLI rooted at the operator's cwd; a
    prompt-injection chain must not get native shell access. On opencode v2
    this map is also the ONLY thing bounding native tools -- there is no
    per-message override left to grant one back.
    """
    cfg = json.loads(MS.managed_config_content())
    for scope in (cfg["permission"], cfg["agent"][MS.MANAGED_AGENT]["permission"]):
        assert scope["bash"] == "deny"
        assert scope["read"] == "deny"


def test_managed_config_denies_read_like_tools():
    """#17: read sat at "allow" while the session directory is the operator's
    cwd -- the repo root or dist/ -- where config.json holds plaintext
    passwords and bearer tokens. ``subagent`` (alias ``task``) is denied
    because a spawned opencode agent carries its own permission set (the
    built-in explore agent allows read/grep/glob), which would bypass the map.
    """
    cfg = json.loads(MS.managed_config_content())
    expected = ("read", "glob", "grep", "subagent", "edit", "bash",
                "webfetch", "external_directory")
    for scope in (cfg["permission"], cfg["agent"][MS.MANAGED_AGENT]["permission"]):
        for key in expected:
            assert scope[key] == "deny", f"{key} is not denied: {scope}"


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
    # v2 answers the prompt at once and fails the TURN: the error rides the
    # assistant timeline item, not a status code on the POST.
    _fake_v2_serve(monkeypatch, [_assistant_item(error={
        "type": "ProviderAuthError", "message": gatekeeper})])
    try:
        OC.send_message("ses_1", "opencode", "m", "agent", "hi", timeout=5)
        raise AssertionError("must raise")
    except RuntimeError as exc:
        assert "OPENCODE_FREE_TIER_DENIED" in str(exc)
        assert OCS.classify_error(str(exc)) is OCS.RetryKind.UPSTREAM


def test_probe_rejects_an_html_200_body(monkeypatch):
    """A status-only probe calls any local web server healthy.

    An unknown GET path on an opencode v2 serve falls back to the web-app
    index and answers 200 with HTML, so ``_probe`` has to require a JSON dict
    body before it calls the instance (and its basic-auth password) live.
    """
    seen = {}

    class _Resp:
        status = 200

        def __init__(self, body: bytes):
            self._body = body

        def read(self, n=-1):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(req, timeout=None):
        seen["url"] = getattr(req, "full_url", "")
        return _Resp(b"<!doctype html><title>opencode</title>")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    assert MS._probe("http://127.0.0.1:4096", "pw", timeout=1.0) is False
    assert seen["url"] == "http://127.0.0.1:4096/api/info"


def test_probe_accepts_a_json_info_body(monkeypatch):
    """The same probe on a real v2 ``/api/info``: 200 + JSON dict is healthy."""
    class _Resp:
        status = 200

        def read(self, n=-1):
            return json.dumps({"version": "2.4.0", "hostname": "localhost"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp())
    assert MS._probe("http://127.0.0.1:4096/", "", timeout=1.0) is True


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


# --- child env hygiene (secrets must not ride into the agentic CLI) ---------


def test_child_env_strips_secret_prefixes():
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/op",
        "KEEP_ME": "harmless",
        "DEEPSEEK_PASSWORD": "hunter2",
        "DEEPSEEK_EMAIL": "a@x.com",
        "DEEPSEAPORT_KEYS": "sk-live-abc",
        "DEEPSEAPORT_TOKEN": "tok",
        "OPENCODE_CONFIG": "/home/op/.config/opencode/opencode.json",
        "OPENCODE_API_KEY": "oc-key",
    }
    env = MS._child_env(parent)
    for gone in ("DEEPSEEK_PASSWORD", "DEEPSEEK_EMAIL", "DEEPSEAPORT_KEYS",
                 "DEEPSEAPORT_TOKEN", "OPENCODE_CONFIG", "OPENCODE_API_KEY"):
        assert gone not in env
    assert "hunter2" not in json.dumps(env)
    # Unrelated process env (PATH/HOME/proxies) is inherited untouched.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/op"
    assert env["KEEP_ME"] == "harmless"
    # The source mapping is never mutated.
    assert parent["DEEPSEEK_PASSWORD"] == "hunter2"


def test_ensure_child_env_has_no_parent_secrets(monkeypatch, tmp_path):
    """F4: the child inherited the whole parent env verbatim.

    .env.example documents DEEPSEAPORT_KEYS / DEEPSEEK_EMAIL /
    DEEPSEEK_PASSWORD / DEEPSEEK_TOKEN, so a sourced .env lands in a child
    that is an agentic CLI with native read/bash, rooted at the operator's cwd.
    """
    # setenv (not delenv) so monkeypatch restores the pre-test value: ensure()
    # writes DEEPSEAPORT_OPENCODE_* into this process env on success.
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_BASE", "")
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_MANAGED", "1")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "hunter2-s3cret")
    monkeypatch.setenv("DEEPSEEK_EMAIL", "op@example.com")
    monkeypatch.setenv("DEEPSEAPORT_KEYS", "sk-live-abc")
    monkeypatch.setenv("DEEPSEAPORT_TOKEN", "tok-abc")
    monkeypatch.setenv("OPENCODE_CONFIG", "/home/op/.config/opencode.json")
    monkeypatch.setenv("KEEP_ME", "harmless")

    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m3")
    monkeypatch.setattr(MS, "_probe", lambda base, pw, timeout=2.0: True)
    monkeypatch.setattr(MS, "_find_free_port", lambda: 45999)
    monkeypatch.setattr("shutil.which", lambda name: "opencode")
    monkeypatch.setattr(MS.subprocess, "run", lambda *a, **k: _Completed())
    monkeypatch.setattr(MS.subprocess, "Popen", _capturing_popen(captured := {}))
    saved = (MS._managed_base, MS._managed_password, MS._proc)
    MS._managed_base = ""
    MS._managed_password = ""
    try:
        assert MS.ensure(timeout=1.0) == "http://127.0.0.1:45999"
    finally:
        (MS._managed_base, MS._managed_password, MS._proc) = saved
    env = captured["env"]
    for gone in ("DEEPSEEK_PASSWORD", "DEEPSEEK_EMAIL", "DEEPSEAPORT_KEYS",
                 "DEEPSEAPORT_TOKEN", "OPENCODE_CONFIG"):
        assert gone not in env
    assert "hunter2-s3cret" not in json.dumps(env)
    # Everything the child legitimately needs is set explicitly after the strip.
    assert env["KEEP_ME"] == "harmless"
    assert env["PATH"]
    assert env["OPENCODE_CONFIG_DIR"].endswith("cfg")
    assert env["XDG_CONFIG_HOME"].endswith("home")
    assert env["OPENCODE_SERVER_PASSWORD"]
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["bash"] == "deny"
    # The parent process env is untouched: only the child copy is stripped.
    assert os.environ["DEEPSEEK_PASSWORD"] == "hunter2-s3cret"


def _Completed():
    return subprocess.CompletedProcess(["opencode", "--version"], 0, "1.18.32", "")


def _capturing_popen(captured: dict):
    class _Proc:
        pid = 4242
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    def _popen(argv, **kw):
        captured["argv"] = list(argv)
        captured["env"] = dict(kw.get("env") or {})
        captured["kwargs"] = kw
        return _Proc()

    return _popen


def test_payload_variant_only_when_set(monkeypatch):
    """v2 made the model, its ``variant`` and the agent SESSION-scoped.

    They ride ``POST /api/session/:id/model`` and ``.../agent`` now, and the
    prompt body carries the text and nothing else (no system, no model, no
    tools key to declare).
    """
    bodies = _fake_v2_serve(monkeypatch, [_assistant_item("hi")])
    assert OC.send_message("ses_1", "opencode", "m", OP.agent_name(), "hi",
                           timeout=5)["content"] == "hi"
    assert next(b for p, b in bodies if p.endswith("/model")) == {
        "model": {"id": "m", "providerID": "opencode"}}
    assert next(b for p, b in bodies if p.endswith("/agent")) == {
        "agent": "deepseaport-passthrough"}
    assert next(b for p, b in bodies if p.endswith("/prompt")) == {"text": "hi"}

    bodies.clear()
    OC.send_message("ses_1", "opencode", "m", OP.agent_name(), "hi",
                    timeout=5, variant="medium")
    assert next(b for p, b in bodies if p.endswith("/model"))["model"] == {
        "id": "m", "providerID": "opencode", "variant": "medium"}


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


def test_variant_resend_forwards_the_cancel_event(monkeypatch):
    """Both send_message calls of a resend carry the cancellation token.

    The variant-less resend must not be the hole a mid-turn cancel leaks
    through: cancellation is honoured per poll, so both calls need the token.
    """
    seen: list = []
    cancel = threading.Event()

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None,
              cancel_event=None, **kw):
        seen.append(cancel_event)
        if variant == "medium":
            raise RuntimeError("OPENCODE_HTTP_400: unknown variant medium")
        return {"content": "ok", "thinking": "",
                "usage_total": 1, "finish": "stop"}

    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: "ses_vf")
    monkeypatch.setattr(OC, "send_message", _send)
    monkeypatch.setattr(OC, "delete_session",
                        lambda sid, timeout=10, **kw: None)
    content, _think, error, _usage = OCS._attempt(
        "hi", DEFAULT_MODEL, 30, variant="medium", cancel_event=cancel)

    assert error is None and content == "ok"
    assert seen == [cancel, cancel], seen


def test_ensure_without_binary_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_BASE", raising=False)
    monkeypatch.delenv("DEEPSEAPORT_OPENCODE_MANAGED", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    assert MS.ensure() is None
    assert os.environ.get("DEEPSEAPORT_OPENCODE_BASE", "") == ""


def test_send_message_cancel_event_aborts_the_session(monkeypatch):
    """Buffered turns were un-cancellable: no abort, no early exit.

    ``stream_message`` already aborted the session on cancel; the blocking
    prompt-then-poll path ignored the token entirely, so a client disconnect
    let the turn run to its deadline.
    """
    bodies = _fake_v2_serve(monkeypatch, [_assistant_item("hi")])
    polls: list[str] = []
    monkeypatch.setattr(
        OC, "get_session_messages",
        lambda sid, timeout=20, **kw: polls.append(sid) or [_assistant_item("hi")])
    cancel = threading.Event()
    cancel.set()
    from deepseaport.server_concurrency import _StreamCancelled

    with pytest.raises(_StreamCancelled):
        OC.send_message("ses_1", "opencode", "m", OP.agent_name(), "hi",
                        timeout=5, cancel_event=cancel)

    assert any(p.endswith("/interrupt") for p, _b in bodies), bodies
    assert polls == [], "cancel must be seen before the first poll"


def test_managed_prompt_forbids_native_and_allows_dsml():
    cfg = json.loads(MS.managed_config_content())
    prompt = cfg["agent"][MS.MANAGED_AGENT]["prompt"]
    assert "never execute native" in prompt.lower()
    assert "DSML" in prompt
    assert "harness" in prompt.lower()
