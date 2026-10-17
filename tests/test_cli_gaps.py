"""Offline gap tests for deepseaport.cli (no network, no browser, no server).

Covers cmd_login / cmd_chat / cmd_waf / cmd_models / cmd_accounts,
_scan_free_ports / _next_free_port / _ensure_free_port / _run_server /
_watch_stop_keys, and main() dispatch. All config goes to tmp_path.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import types

import pytest

from support import on_disk, write_config

from deepseaport.cli import (
    _ensure_free_port,
    _next_free_port,
    _run_server,
    _scan_free_ports,
    _watch_stop_keys,
    cmd_accounts,
    main as cli_main,
)


def _load(path):
    from deepseaport.config import load_settings

    return load_settings(path)


# cmd_login -------------------------------------------------------------------

def test_login_success_saves_token_and_active(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok123", False, ""))
    rc = cli_main(["--config", cfg, "login", "--email", "a@x.com", "--password", "pw"])
    assert rc == 0
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "tok123"
    assert s.accounts[0].email == "a@x.com"
    assert s.active_account == "a@x.com"


def test_login_banned_with_token_returns_3_and_persists_banned(tmp_path, monkeypatch, capsys):
    import deepseaport.accounts as acc_mod
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok-banned", True, "suspended"))
    monkeypatch.setattr(acc_mod, "check_ban_for_token", lambda token, timeout=12: (True, None))
    rc = cli_main(["--config", cfg, "login", "--email", "b@x.com", "--password", "pw"])
    assert rc == 3
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "tok-banned"
    assert s.accounts[0].banned is True
    out = capsys.readouterr().out
    assert "BANNED" in out


def test_login_no_token_returns_3(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: (None, False, ""))
    rc = cli_main(["--config", cfg, "login", "--email", "c@x.com", "--password", "pw"])
    assert rc == 3
    assert _load(cfg).accounts == []


def test_login_banned_no_token_returns_3(tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: (None, True, "suspended detail"))
    rc = cli_main(["--config", cfg, "login", "--email", "d@x.com", "--password", "pw"])
    assert rc == 3
    assert _load(cfg).accounts == []
    assert "BANNED" in capsys.readouterr().out


def test_login_no_token_prints_the_ban_wording_hint(tmp_path, monkeypatch, capsys):
    """A single ban-wording sighting no longer sets login_banned, but the driver
    reports it in the detail; the user still has to hear it."""
    import deepseaport.cli as cli_mod

    hint = "captcha on the signin page [ban wording seen: 'suspended until']"
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: (None, False, hint))
    rc = cli_main(["--config", cfg, "login", "--email", "b@x.com", "--password", "pw"])
    assert rc == 3
    assert _load(cfg).accounts == []  # nothing persisted
    out = capsys.readouterr().out
    assert "Evidence:" in out
    assert "ban wording seen" in out


def _patch_banned_login(monkeypatch, expires_at: float):
    import deepseaport.accounts as acc_mod
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok-banned", True, "suspended"))
    monkeypatch.setattr(acc_mod, "check_ban_for_token",
                        lambda token, timeout=12: (True, expires_at))


def test_login_expiry_line_omitted_when_formatter_returns_empty(
        tmp_path, monkeypatch, capsys):
    """format_ban_datetime may legitimately have nothing to render (a garbage
    ban timestamp); "Expires:" with an empty tail is not a message."""
    import deepseaport.providers.deepseek.protocol as proto_mod

    _patch_banned_login(monkeypatch, 1789555791.11)
    monkeypatch.setattr(proto_mod, "format_ban_datetime", lambda ts: "")
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_main(["--config", cfg, "login", "--email", "b@x.com",
                     "--password", "pw"]) == 3
    out = capsys.readouterr().out
    assert "BANNED" in out
    assert "Expires:" not in out


def test_login_expiry_line_kept_when_formatter_returns_text(
        tmp_path, monkeypatch, capsys):
    """Control: a renderable expiry still prints with the historical wording."""
    import deepseaport.providers.deepseek.protocol as proto_mod

    _patch_banned_login(monkeypatch, 1789555791.11)
    monkeypatch.setattr(proto_mod, "format_ban_datetime",
                        lambda ts: "16 September 2026 12:49")
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_main(["--config", cfg, "login", "--email", "b@x.com",
                     "--password", "pw"]) == 3
    assert "Expires: 16 September 2026 12:49" in capsys.readouterr().out


# save failures (finding 14) ---------------------------------------------------
#
# Settings.save() raises by design (an unparsable config.json, an unwritable
# location). It used to escape as a traceback -- the whole UX of a one-shot
# CLI -- and in cmd_login it also discarded the token the ~30s browser
# authorization had just produced.


def test_login_save_failure_reports_and_keeps_the_captured_token(
        tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod

    bad = tmp_path / "config.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("tok-captured", False, ""))

    rc = cli_main(["--config", str(bad), "login", "--email", "a@x.com",
                   "--password", "pw"])

    assert rc != 0
    out = capsys.readouterr().out
    assert "could not write" in out
    assert "tok-captured" in out, "the captured token was thrown away"
    assert "Traceback" not in out


def test_banned_login_save_failure_also_keeps_the_token(
        tmp_path, monkeypatch, capsys):
    """The banned branch had its own unguarded save()."""
    _patch_banned_login(monkeypatch, 0.0)
    bad = tmp_path / "config.json"
    bad.write_text("{not json", encoding="utf-8")

    rc = cli_main(["--config", str(bad), "login", "--email", "b@x.com",
                   "--password", "pw"])

    assert rc == 3
    out = capsys.readouterr().out
    assert "could not write" in out
    assert "tok-banned" in out


@pytest.mark.parametrize("argv", [
    ["accounts", "add", "--email", "new@x.com", "--token", "tok-new"],
    ["accounts", "select", "a@x.com"],
    ["accounts", "remove", "a@x.com"],
    ["accounts", "set-token", "a@x.com", "--token", "fresh"],
    ["accounts", "unblock"],
])
def test_accounts_commands_survive_an_unwritable_config(
        tmp_path, monkeypatch, capsys, argv):
    """Every mutating accounts command reports the write failure and exits 1
    instead of raising through the entry point."""
    from deepseaport.config import Settings

    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok-a"}]})

    def _boom(self):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Settings, "save", _boom)

    rc = cli_main(["--config", cfg, *argv])

    assert rc == 1
    out = capsys.readouterr().out
    assert "could not write" in out
    assert "Traceback" not in out


# cmd_chat --------------------------------------------------------------------

def _patch_chat(monkeypatch, thinking="think-stuff", content="hello"):
    import deepseaport.accounts as acc_mod
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.providers.deepseek.completion_service as ds_cs
    import deepseaport.server as srv_mod

    fake_app = types.SimpleNamespace(state=types.SimpleNamespace(bridge=None))
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: fake_app)

    class _FakeObscuraBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire", lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(ds_cs, "_complete_with_failover_sync",
                        lambda app, pool, prep, item, **k: {"content": content, "thinking": thinking})


def test_chat_returns_0_and_prints_thinking(tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [{"email": "a@x.com", "token": "tok123"}]})
    _patch_chat(monkeypatch, thinking="think-stuff", content="hello")
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "THINKING:" in out
    assert "think-stuff" in out
    assert "hello" in out


def test_chat_no_thinking_prints_content_only(tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [{"email": "a@x.com", "token": "tok123"}]})
    _patch_chat(monkeypatch, thinking="", content="just-content")
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "THINKING" not in out
    assert "just-content" in out


def test_chat_dispatches_through_the_selected_provider_not_deepseeks_engine(
        tmp_path, monkeypatch, capsys):
    """Finding 6. `chat --provider qwen` ran DeepSeek's whole code path.

    Pre-change ``cmd_chat`` imported DeepSeek's
    ``_complete_with_failover_sync`` unconditionally and called it for EVERY
    provider. That function closes over DeepSeek's module global
    ``_run_completion_core`` and never reads ``prep["provider"]``, so a Qwen
    account was selected, validated, pooled -- and its ES256 JWT was then sent
    to chat.deepseek.com as a ``userToken``.

    The two tests above could never see it: they monkeypatch DeepSeek's engine
    global, which is exactly the seam the bug hid behind. So this asserts
    WHICH callable ran -- each provider's own ``Provider.complete`` -- and
    makes DeepSeek's engine raise if it is entered at all.
    """
    import deepseaport.accounts as acc_mod
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.providers.deepseek.completion_service as ds_cs
    import deepseaport.providers.deepseek.provider as ds_prov
    import deepseaport.providers.qwen.provider as qw_prov
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))

    class _FakeObscuraBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())

    ran: list[str] = []

    def _qwen_complete(self, app, pool, prep, first_item, cancel_event=None,
                       on_content=None, on_thinking=None):
        ran.append(("qwen", first_item, prep.get("provider")))
        return {"content": "qwen-answer", "thinking": ""}

    def _ds_complete(self, *a, **k):
        ran.append(("deepseek", None, None))
        return {"content": "deepseek-answer", "thinking": ""}

    def _ds_engine_entered(*a, **k):
        raise AssertionError("DeepSeek's engine ran for a --provider qwen chat")

    monkeypatch.setattr(qw_prov.QwenProvider, "complete", _qwen_complete)
    monkeypatch.setattr(ds_prov.DeepSeekProvider, "complete", _ds_complete)
    monkeypatch.setattr(ds_cs, "_complete_with_failover_sync", _ds_engine_entered)

    cfg = write_config(tmp_path, {
        "keys": [],
        "accounts": [{"email": "a@x.com", "token": "qwen-jwt",
                      "provider": "qwen"}]})

    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 0
    # Which callable ran, in one assertion -- not "some engine was reached".
    assert [name for name, _item, _pp in ran] == ["qwen"]
    assert ran[0][2] == "qwen", "prep was not resolved to the qwen provider"
    assert "qwen-answer" in capsys.readouterr().out


def test_chat_qwen_one_shot_skips_the_deepseek_host_warmup(tmp_path, monkeypatch, capsys):
    """needs_waf_jar=False: build and register the bridge, skip the warmup.

    The one-shot bridge is qwen's engine/UA source (WAF recovery), but the
    warmup fetches CHAT_HOME -- chat.deepseek.com -- which a qwen-only request
    has no use for: qwen's cookies come from its own acw_tc jar.
    """
    import deepseaport.accounts as acc_mod
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.providers.qwen.provider as qw_prov
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))

    calls: dict = {"warmup": 0}

    class _FakeObscuraBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            calls["warmup"] += 1
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(qw_prov.QwenProvider, "complete",
                        lambda self, app, pool, prep, item, **k: {"content": "ok", "thinking": ""})

    cfg = write_config(tmp_path, {
        "keys": [],
        "accounts": [{"email": "a@x.com", "token": "qwen-jwt",
                      "provider": "qwen"}]})

    rc = cli_main(["--config", cfg, "chat", "--provider", "qwen",
                   "--model", "qwen3.7-plus", "--prompt", "hi"])

    assert rc == 0
    assert calls["warmup"] == 0, (
        "one-shot chat booted obscura against chat.deepseek.com for qwen")
    capsys.readouterr()


def test_chat_ban_failover_survives_the_provider_dispatch(tmp_path, monkeypatch):
    """The dispatch must not cost cmd_chat its same-request ban failover.

    The replaced line existed to get ban failover (a banned CURRENT transparently
    retries on the next unbanned account). Routing through ``Provider.complete``
    keeps it only because every provider delegates to its OWN
    _complete_with_failover_sync -- so this asserts Qwen's engine, not DeepSeek's,
    is the one that owns the retry.
    """
    import deepseaport.accounts as acc_mod
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.providers.qwen.completion_service as qw_cs
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))

    class _FakeObscuraBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())

    seen: dict = {}

    def _engine(app, pool, prep, item, cancel_event=None, **k):
        seen["engine"] = "qwen"
        seen["cancel_event"] = cancel_event
        return {"content": "after-failover", "thinking": ""}

    monkeypatch.setattr(qw_cs, "_complete_with_failover_sync", _engine)

    cfg = write_config(tmp_path, {
        "keys": [],
        "accounts": [{"email": "a@x.com", "token": "qwen-jwt",
                      "provider": "qwen"}]})

    assert cli_main(["--config", cfg, "chat", "--provider", "qwen",
                     "--model", "qwen3.7-plus", "--prompt", "hi"]) == 0
    assert seen == {"engine": "qwen", "cancel_event": None}


def test_chat_reaches_the_accountless_provider_without_a_pool(tmp_path, monkeypatch,
                                                             capsys):
    """Finding 6, other half: the accountless branch must not touch DeepSeek.

    ``chat --provider opencode`` skips the pool entirely. Asserting opencode's
    own provider ran with pool=None/item=None proves both branches dispatch.
    """
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.providers.deepseek.completion_service as ds_cs
    import deepseaport.providers.opencode.provider as oc_prov
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))
    monkeypatch.setattr(bridge_mod, "ObscuraBridge",
                        lambda *a, **k: pytest.fail("opencode needs no WAF bridge"))
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)

    seen: dict = {}

    def _oc_complete(self, app, pool, prep, first_item, cancel_event=None,
                     on_content=None, on_thinking=None):
        seen.update(pool=pool, item=first_item, provider=prep.get("provider"))
        return {"content": "local-answer", "thinking": ""}

    def _ds_engine_entered(*a, **k):
        raise AssertionError("DeepSeek's engine ran for --provider opencode")

    monkeypatch.setattr(oc_prov.OpencodeProvider, "complete", _oc_complete)
    monkeypatch.setattr(ds_cs, "_complete_with_failover_sync", _ds_engine_entered)

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_main(["--config", cfg, "chat", "--provider", "opencode",
                   "--model", "opencode/muse-spark-1.3-contributor-free",
                   "--prompt", "hi"])

    assert rc == 0
    assert seen == {"pool": None, "item": None, "provider": "opencode"}
    assert "local-answer" in capsys.readouterr().out


def test_chat_refuses_instead_of_raising_when_no_provider_resolves(
        tmp_path, monkeypatch, capsys):
    """The account-holding branch needs a provider; ``None`` must not crash.

    ``cmd_chat`` resolves the provider defensively, so it can be None. The
    branch below it has no provider to dispatch to, and a one-shot CLI prints a
    raw traceback as its whole UX -- so it takes the same _chat_fail path every
    other failure there takes, and exits 1.
    """
    import deepseaport.accounts as acc_mod
    import deepseaport.core.obscura_bridge as bridge_mod
    import deepseaport.core.registry as reg_mod
    import deepseaport.server as srv_mod

    monkeypatch.setattr(srv_mod, "create_app",
                        lambda settings=None: types.SimpleNamespace(
                            state=types.SimpleNamespace(bridge=None)))

    class _FakeObscuraBridge:
        def __init__(self, *a, **k):
            self.state = types.SimpleNamespace(user_agent="ua", cookies={})

        def warmup(self, timeout=45):
            return {}

    monkeypatch.setattr(bridge_mod, "ObscuraBridge", _FakeObscuraBridge)
    monkeypatch.setattr(bridge_mod, "register_default_bridge", lambda *a, **k: None)
    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(reg_mod, "get", lambda pid: None)

    cfg = write_config(tmp_path, {
        "keys": [], "accounts": [{"email": "a@x.com", "token": "tok"}]})

    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "chat failed" in out
    assert on_disk(cfg)["accounts"][0]["token"] == "tok"  # nothing else touched


# cmd_waf / cmd_models ----------------------------------------------------------

class _FakeWafBridge:
    def __init__(self, has_token: bool):
        self._has = has_token
        self.binary = "fake-obscura"
        self.state = types.SimpleNamespace(user_agent="fake-ua", cookies={})

    def version(self):
        return "9.9-test"

    def warmup(self, timeout=45):
        return {"aws-waf-token": "x"} if self._has else {}

    def has_waf_token(self):
        return self._has


def _patch_waf(monkeypatch, has_token: bool):
    import deepseaport.core.obscura_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "ObscuraBridge",
                        lambda binary="", profile="": _FakeWafBridge(has_token))


def test_waf_warmup_true_returns_0(tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    _patch_waf(monkeypatch, True)
    assert cli_main(["--config", cfg, "waf"]) == 0
    out = capsys.readouterr().out
    assert "fake-obscura" in out
    assert "cookies:" in out


def test_waf_no_token_returns_2(tmp_path, monkeypatch):
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    _patch_waf(monkeypatch, False)
    assert cli_main(["--config", cfg, "waf"]) == 2


def test_models_prints_4_ids(tmp_path, monkeypatch, capsys):
    from deepseaport.server import MODELS

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "models"]) == 0
    out = capsys.readouterr().out
    ids = json.loads(out)
    assert set(ids) == set(MODELS)
    assert len(ids) == 4


# cmd_accounts ------------------------------------------------------------------

def test_accounts_list_ban_label_and_missing_hint(tmp_path, monkeypatch, capsys):
    import deepseaport.accounts as acc_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok-with-token-1234567890"},
        {"email": "b@x.com", "token": ""},
    ]})
    monkeypatch.setattr(acc_mod, "collect_ban_labels", lambda accounts, **k: {"a@x.com": None})
    assert cli_main(["--config", cfg, "accounts", "list"]) == 0
    out = capsys.readouterr().out
    assert "(BANNED)" in out
    assert "MISSING" in out
    assert "missing token: b@x.com" in out


def test_accounts_select_clear(tmp_path):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ], "active_account": "a@x.com"})
    assert cli_main(["--config", cfg, "accounts", "select", "--clear"]) == 0
    assert _load(cfg).active_account == ""


def test_accounts_add_missing_id_returns_2(tmp_path, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add"]) == 2
    assert "need --email" in capsys.readouterr().out


def test_accounts_set_token_missing_args_returns_2(tmp_path):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ]})
    # Empty token value -> usage error.
    assert cli_main(["--config", cfg, "accounts", "set-token", "a@x.com", "--token", ""]) == 2
    # Missing identifier -> usage error (direct call; argparse would SystemExit).
    args = argparse.Namespace(config=cfg, accounts_action="set-token", identifier="", token="x")
    assert cmd_accounts(args) == 2


def test_accounts_remove_missing_returns_1(tmp_path, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"},
    ]})
    assert cli_main(["--config", cfg, "accounts", "remove", "missing@x.com"]) == 1
    assert "not found" in capsys.readouterr().out


def test_accounts_remove_no_identifier_returns_2(tmp_path):
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, accounts_action="remove", identifier="")
    assert cmd_accounts(args) == 2


def test_accounts_unblock_clears_persisted_ban(tmp_path):
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok123", "banned": True, "banned_until": 0.0},
    ]})
    assert cli_main(["--config", cfg, "accounts", "unblock"]) == 0
    s = _load(cfg)
    assert s.accounts[0].banned is False
    assert float(s.accounts[0].banned_until or 0.0) == 0.0


def test_accounts_unknown_action_returns_2(tmp_path, capsys):
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, accounts_action="bogus")
    assert cmd_accounts(args) == 2
    assert "unknown accounts action" in capsys.readouterr().out


# ports -------------------------------------------------------------------------

def test_scan_free_ports_bad_start_falls_back():
    ports = _scan_free_ports("127.0.0.1", "bad", 1, limit=5, check=lambda h, p: True)
    assert ports == [5001]


def test_scan_free_ports_limit_exhausted_short_list():
    ports = _scan_free_ports("127.0.0.1", 5000, 5, limit=2, check=lambda h, p: True)
    assert ports == [5000, 5001]


def test_scan_free_ports_check_raises_skipped():
    def _boom(host, port):
        raise OSError("busy")

    assert _scan_free_ports("127.0.0.1", 5000, 2, limit=5, check=_boom) == []


def test_next_free_port_none_when_all_busy(monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_is_port_free", lambda host, port: False)
    assert _next_free_port("127.0.0.1", 5000, limit=3) is None


def test_ensure_free_port_no_free_returns_original(tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod
    from deepseaport.config import Settings

    s = Settings(port=5000, config_path=str(tmp_path / "c.json"))
    monkeypatch.setattr(cli_mod, "_is_port_free", lambda host, port: False)
    monkeypatch.setattr(cli_mod, "_next_free_port", lambda host, port, limit=50: None)
    assert _ensure_free_port(s, "127.0.0.1", 5000) == 5000
    assert "no free port" in capsys.readouterr().out


def test_run_server_single_worker_uses_uvicorn_run(tmp_path, monkeypatch):
    """The supported path is unchanged: uvicorn.Server on the resolved port."""
    import deepseaport.cli as cli_mod
    import deepseaport.server as srv_mod
    import uvicorn

    from deepseaport.config import Settings

    captured: dict = {}

    class _FakeConfig:
        def __init__(self, app, **kw):
            captured.update(kw)

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: captured.update(run_called=True))
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: object())
    monkeypatch.setattr(srv_mod, "apply_log_level", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_ensure_free_port", lambda s, h, p: p)

    s = Settings(port=5001, log_level="INFO", config_path=str(tmp_path / "c.json"))
    args = argparse.Namespace(host="", port=0, workers=1)
    assert _run_server(s, args) == 0
    assert captured.get("ran") is True
    assert captured.get("port") == 5001
    assert captured.get("host") == "127.0.0.1"
    assert "run_called" not in captured  # the multi-worker uvicorn.run path is gone


def test_watch_stop_keys_never_raises_with_exited_server():
    srv = types.SimpleNamespace(should_exit=True)
    _watch_stop_keys(srv)  # must not raise
    assert srv.should_exit is True


def test_watch_stop_keys_broken_server_never_raises():
    class _Boom:
        @property
        def should_exit(self):
            raise RuntimeError("boom")

    _watch_stop_keys(_Boom())  # must not raise


# main dispatch -------------------------------------------------------------------

def test_main_no_subcommand_defaults_to_serve(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    seen = {}
    def _fake_serve(args):
        seen["cmd"] = args.cmd
        return 42
    monkeypatch.setattr(cli_mod, "cmd_serve", _fake_serve)
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg]) == 42
    assert seen["cmd"] == "serve"


def test_main_login_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "cmd_login", lambda args: 7)
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg, "login", "--email", "a@x.com"]) == 7


def test_main_chat_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(cli_mod, "cmd_chat", lambda args: 8)
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.main(["--config", cfg, "chat", "--prompt", "hi"]) == 8


def test_main_unknown_cmd_returns_1(monkeypatch):
    import deepseaport.cli as cli_mod

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, argv=None: argparse.Namespace(config=None, cmd="bogus"))
    try:
        assert cli_mod.main([]) == 1
    finally:
        # restore is handled by monkeypatch undo
        pass


def test_main_waf_models_accounts_dispatch(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "cmd_waf", lambda args: 11)
    assert cli_mod.main(["--config", cfg, "waf"]) == 11
    monkeypatch.setattr(cli_mod, "cmd_models", lambda args: 12)
    assert cli_mod.main(["--config", cfg, "models"]) == 12
    monkeypatch.setattr(cli_mod, "cmd_accounts", lambda args: 13)
    assert cli_mod.main(["--config", cfg, "accounts", "list"]) == 13


def test_ensure_free_port_free_returns_same(tmp_path):
    import deepseaport.cli as cli_mod
    from deepseaport.config import Settings

    s = Settings(port=5010, config_path=str(tmp_path / "c.json"))
    # Loopback high ports are almost always free; but stub to be deterministic.
    orig = cli_mod._is_port_free
    try:
        cli_mod._is_port_free = lambda host, port: True
        assert _ensure_free_port(s, "127.0.0.1", 5010) == 5010
    finally:
        cli_mod._is_port_free = orig


def test_scan_free_ports_negative_start_clamped():
    ports = _scan_free_ports("127.0.0.1", -5, 1, limit=10, check=lambda h, p: True)
    assert ports == [1]


def test_port_flag_rejects_out_of_range_and_zero(tmp_path, capsys):
    """Pre-change a raw type=int accepted 70000, ``_ensure_free_port`` handed it
    to uvicorn and bind() died with an uncaught OverflowError."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    for bad in ("70000", "0", "-1"):
        with pytest.raises(SystemExit) as exc:
            cli_main(["--config", cfg, "serve", "--port", bad])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "1-65535" in err
        assert bad in err


def test_port_flag_accepts_a_bindable_port(tmp_path, monkeypatch):
    import deepseaport.cli as cli_mod

    seen: dict = {}
    monkeypatch.setattr(cli_mod, "_run_server",
                        lambda s, a: seen.update(port=s.port) or 0)
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "serve", "--no-tui",
                     "--port", "5123"]) == 0
    assert seen == {"port": 5123}


@pytest.mark.parametrize("bad", [70000, 0, -1, "abc", None])
def test_ensure_free_port_out_of_range_falls_back_to_default(
        bad, tmp_path, monkeypatch, caplog):
    """The guard must never return an out-of-range port: uvicorn's bind dies on
    it with OverflowError, which _run_server does not catch."""
    import deepseaport.cli as cli_mod
    from deepseaport.config import Settings

    s = Settings(port=5001, config_path=str(tmp_path / "c.json"))
    monkeypatch.setattr(cli_mod, "_is_port_free",
                        lambda host, port: port == 5001)
    monkeypatch.setattr(cli_mod, "_next_free_port",
                        lambda host, port, limit=50: None)
    with caplog.at_level(logging.WARNING, logger="deepseaport.server_runner"):
        assert _ensure_free_port(s, "127.0.0.1", bad) == 5001
    assert any("is not a usable listen port" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_run_server_garbage_settings_port_uses_default(tmp_path, monkeypatch):
    """A hand-built Settings with a non-numeric port must fall back, not raise
    ValueError out of _run_server's pre-flight line."""
    import deepseaport.cli as cli_mod
    import deepseaport.server as srv_mod
    import uvicorn

    from deepseaport.config import Settings

    captured: dict = {}

    class _FakeConfig:
        def __init__(self, app, **kw):
            captured.update(kw)

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: object())
    monkeypatch.setattr(srv_mod, "apply_log_level", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_is_port_free", lambda host, port: port == 5001)

    s = Settings(port="not-a-port", log_level="INFO",
                 config_path=str(tmp_path / "c.json"))
    args = argparse.Namespace(host="", port=0, workers=1)
    assert _run_server(s, args) == 0
    assert captured.get("ran") is True
    assert captured.get("port") == 5001


def test_login_upsert_path_uses_tmp_config(tmp_path, monkeypatch):
    """Second login for same email updates token, keeps single account."""
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("first-tok", False, ""))
    assert cli_main(["--config", cfg, "login", "--email", "e@x.com", "--password", "pw"]) == 0
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda email, password, settings: ("second-tok", False, ""))
    assert cli_main(["--config", cfg, "login", "--email", "e@x.com", "--password", "pw"]) == 0
    s = _load(cfg)
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "second-tok"
    # Token persists across reload.
    assert _load(cfg).accounts[0].token == "second-tok"


def test_unblock_single_identifier_only(tmp_path):
    future = time.time() + 3600
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1", "banned": True, "banned_until": future},
        {"email": "b@x.com", "token": "t2", "banned": True, "banned_until": future},
    ]})
    assert cli_main(["--config", cfg, "accounts", "unblock", "a@x.com"]) == 0
    s = _load(cfg)
    by_email = {a.email: a for a in s.accounts}
    assert by_email["a@x.com"].banned is False
    assert by_email["b@x.com"].banned is True


# secrets on the command line (C9) ------------------------------------------------

def test_secret_flag_value_warns_about_the_process_list(tmp_path, capsys):
    """--token/--password expose the value to every same-user process."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "a@x.com", "--token", "tok123"]) == 0
    out = capsys.readouterr().out
    assert "process list" in out
    assert "shell history" in out


def test_login_password_flag_value_warns(tmp_path, monkeypatch, capsys):
    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("tok123", False, ""))
    assert cli_main(["--config", cfg, "login",
                     "--email", "a@x.com", "--password", "pw"]) == 0
    assert "process list" in capsys.readouterr().out


def test_secret_flags_keep_working_for_scripting(tmp_path):
    """Non-interactive callers must not be blocked by the prompt path."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add", "--email", "a@x.com",
                     "--password", "pw", "--token", "tok123"]) == 0
    acc = _load(cfg).accounts[0]
    assert (acc.password, acc.token) == ("pw", "tok123")


def test_missing_secret_flag_prompts_without_echo_on_a_tty(tmp_path, monkeypatch):
    """No flag + a console -> getpass, so the secret never hits the process list."""
    import deepseaport.cli as cli_mod

    prompted: list[str] = []
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda p="": prompted.append(p) or "tok-typed")
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add", "--email", "a@x.com"]) == 0
    assert _load(cfg).accounts[0].token == "tok-typed"
    assert any("Token" in p for p in prompted)


def test_missing_secret_flag_does_not_prompt_when_headless(tmp_path, monkeypatch):
    """Pipes/CI: no flag and no console means no prompt, as before."""
    import deepseaport.cli as cli_mod

    called: list[str] = []
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr("getpass.getpass", lambda p="": called.append(p) or "nope")
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add", "--email", "a@x.com"]) == 0
    assert called == []
    assert _load(cfg).accounts[0].token == ""


def test_password_and_token_help_mention_the_risk(capsys):
    """--password had no help at all; both must document the exposure."""
    import deepseaport.cli as cli_mod

    seen = []
    for argv in (["accounts", "add", "--help"], ["login", "--help"]):
        with pytest.raises(SystemExit) as exc:
            cli_mod.main(argv)
        assert exc.value.code == 0
        seen.append(capsys.readouterr().out)
    add_help, login_help = seen
    assert "process list" in add_help and "shell history" in add_help
    assert "process list" in login_help  # --password used to have no help at all


# headless tty probe (C10) --------------------------------------------------------

def test_cmd_serve_survives_missing_stdin(monkeypatch, tmp_path):
    """sys.stdin is None under pythonw.exe: the old bare .isatty() raised."""
    import sys as _sys

    import deepseaport.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"}]})
    monkeypatch.setattr(_sys, "stdin", None)
    served = {}

    def _serve(s, a):
        served["ran"] = True
        return 0

    monkeypatch.setattr(cli_mod, "_run_server", _serve)
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=False, cmd="serve",
                              provider=None, providers=None)
    assert cli_mod.cmd_serve(args) == 0
    # No console to draw on -> straight to the server, no AttributeError.
    assert served == {"ran": True}


def test_cmd_serve_still_opens_tui_on_a_real_tty(monkeypatch, tmp_path):
    import sys as _sys

    import deepseaport.cli as cli_mod
    import deepseaport.tui as tui_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1"}]})

    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(_sys, "stdin", _Tty())
    monkeypatch.setattr(tui_mod, "provider_menu", lambda s: None)
    monkeypatch.setattr(tui_mod, "main_menu", lambda *a, **k: 0)
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=False, cmd="serve",
                              provider=None, providers=None)
    assert cli_mod.cmd_serve(args) == 0
