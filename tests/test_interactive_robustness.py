"""Interactive-path robustness regressions (C1-C7): data loss, Ctrl+C, wrong row.

Every test here reproduces a specific pre-change failure:

* C1  truncated config.json + _refresh_from_disk + save() erased every account
* C2  Ctrl+C during _wait_for_server leaked the uvicorn thread and pinned
     every logger (and settings.log_level) at ERROR
* C3  deleting row 2 of two accounts sharing an email removed the FIRST
     match (the other provider's account) instead of the confirmed one
* C4  uvicorn's SystemExit(STARTUP_FAILURE) on bind failure killed the TUI
* C5  multi-chat refused to start on an accountless provider (opencode)
* C6  auto-login ban state was applied to an account that did not exist yet
* C7  `chat` leaked a traceback (plus "Press Enter to close...") on the
     empty-pool / unknown-model / bad-messages paths

No network, no browser, no real server: uvicorn, the WAF bridge, the ban
probes and the REPL are all faked.
"""

from __future__ import annotations

import json
import logging
import types

import pytest

from deepseaport import chat_ui
from deepseaport.config import AccountConfig, Settings
from functools import partial

from support import JoinCountingThread, settings, write_config

# Default kwargs differ per file; the shared base dict lives in support.py.
_fresh = partial(settings, config_path="", log_level="INFO",
                  warmup_on_startup=True)


def _answers(monkeypatch, seq):
    """Drive input() and record the prompts it was called with.

    The prompts have to be captured: a monkeypatched input() never writes its
    prompt to stdout, so capsys cannot see the delete confirmation.
    """
    it = iter(seq)
    seen: list[str] = []

    def _inp(prompt=""):
        seen.append(prompt)
        return next(it)

    monkeypatch.setattr("builtins.input", _inp)
    return seen


# --- C1: _refresh_from_disk must not destroy accounts -------------------------

def test_refresh_truncated_config_then_save_keeps_accounts(tmp_path, capsys):
    """End-to-end: half-written config + refresh + save() must keep both accounts.

    Pre-change this persisted "accounts": [] -- the whole credential pool gone
    because a transiently truncated file was read as "no accounts".
    """
    from deepseaport.config import load_settings
    from deepseaport.tui import _refresh_from_disk

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "keys": ["sk-1"],
        "accounts": [
            {"email": "a@x.com", "password": "pa", "token": "t1"},
            {"email": "b@x.com", "password": "pb", "token": "t2"},
        ],
        "active_account": "a@x.com",
    }), encoding="utf-8")
    s = load_settings(str(cfg))
    assert len(s.accounts) == 2

    # What the Windows non-atomic save fallback (or an editor mid-write) leaves.
    whole = json.dumps({"keys": ["sk-1"], "accounts": [
        {"email": "a@x.com", "password": "pa", "token": "t1"}]})
    cfg.write_text(whole[:len(whole) // 2], encoding="utf-8")

    _refresh_from_disk(s)
    # Any later toggle / key add / port change persists from memory.
    s.save()

    on_disk = json.loads(cfg.read_text(encoding="utf-8"))
    assert [a["email"] for a in on_disk["accounts"]] == ["a@x.com", "b@x.com"]
    assert [a["password"] for a in on_disk["accounts"]] == ["pa", "pb"]
    assert [a["token"] for a in on_disk["accounts"]] == ["t1", "t2"]
    assert "not valid JSON" in capsys.readouterr().out


def test_refresh_rejects_non_object_json(tmp_path, capsys):
    """A JSON array/string is not a config: keep memory, say why."""
    from deepseaport.config import load_settings
    from deepseaport.tui import _refresh_from_disk

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"accounts": [{"email": "a@x.com", "token": "t1"}]}),
                   encoding="utf-8")
    s = load_settings(str(cfg))
    cfg.write_text("[1, 2, 3]", encoding="utf-8")
    _refresh_from_disk(s)
    assert [a.email for a in s.accounts] == ["a@x.com"]
    assert "not a JSON object" in capsys.readouterr().out


def test_refresh_still_applies_a_valid_file(tmp_path):
    """The non-destructive guard must not stop the API hand-off it exists for."""
    from deepseaport.config import load_settings
    from deepseaport.tui import _refresh_from_disk

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"accounts": [{"email": "a@x.com", "token": "t1"}]}),
                   encoding="utf-8")
    s = load_settings(str(cfg))
    cfg.write_text(json.dumps({
        "accounts": [{"email": "a@x.com", "token": "t1"},
                     {"email": "z@x.com", "token": "t9"}],
        "active_account": "z@x.com"}), encoding="utf-8")
    _refresh_from_disk(s)
    assert [a.email for a in s.accounts] == ["a@x.com", "z@x.com"]
    assert s.active_account == "z@x.com"


# --- C2: Ctrl+C must not leak the server or the log levels -------------------

class _Srv:
    def __init__(self):
        self.should_exit = False


def _fake_server(monkeypatch, srv=None, thr=None):
    srv = srv if srv is not None else _Srv()
    thr = thr if thr is not None else JoinCountingThread()
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr(chat_ui, "_start_background_server",
                        lambda app, host, port: (srv, thr))
    return srv, thr


def test_chat_server_ctrl_c_in_wait_restores_logs_and_stops_server(monkeypatch):
    """KeyboardInterrupt in the _wait_for_server window is the common Ctrl+C.

    Pre-change it skipped every `except Exception`, so the uvicorn thread kept
    running, should_exit stayed False, and settings.log_level plus the root
    logger stayed at ERROR for the rest of the TUI session.
    """
    srv, thr = _fake_server(monkeypatch)
    monkeypatch.setattr(chat_ui, "_wait_for_server", _raise_ctrl_c)
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")],
               log_level="INFO")
    root_level = logging.getLogger().level
    quiet_level = logging.getLogger("deepseaport").level

    with pytest.raises(KeyboardInterrupt):
        chat_ui.run_chat_server(s)

    assert srv.should_exit is True
    assert thr.joined == 1
    assert s.log_level == "INFO"
    assert logging.getLogger().level == root_level
    assert logging.getLogger("deepseaport").level == quiet_level


def _raise_ctrl_c(*a, **k):
    raise KeyboardInterrupt


def test_multi_chat_ctrl_c_in_wait_restores_logs(monkeypatch):
    """Same hole in run_multi_chat_servers: levels stayed at ERROR forever."""
    srv, _thr = _fake_server(monkeypatch)
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live",
                        lambda st: ([AccountConfig(email="a@x.com", token="t1")], 0, 0))
    monkeypatch.setattr(chat_ui, "_wait_for_server", _raise_ctrl_c)
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")], port=54321)
    root_level = logging.getLogger().level

    with pytest.raises(KeyboardInterrupt):
        chat_ui.run_multi_chat_servers(s, count=1)

    assert srv.should_exit is True
    assert logging.getLogger().level == root_level


def test_chat_server_normal_exit_stops_server_exactly_once(monkeypatch):
    """The cleanup guard must not double-stop on the ordinary REPL exit path."""
    srv, thr = _fake_server(monkeypatch)
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(chat_ui.chat_repl, "run_repl", lambda **k: None)
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")], log_level="INFO")

    assert chat_ui.run_chat_server(s) == 0
    assert srv.should_exit is True
    assert thr.joined == 1
    assert s.log_level == "INFO"


# --- C3: row-index addressing when two providers share an email ---------------

def test_delete_row_two_keeps_the_same_email_other_provider(monkeypatch, capsys):
    """Row 2 is the qwen account; the survivor must be the deepseek one.

    Pre-change the confirm named no provider and the removal went through
    pool.remove()/remove_account(), which match by identifier and take the
    FIRST hit -- so the deepseek account was destroyed instead.
    """
    from deepseaport.tui import accounts_menu

    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        active_account="", provider="deepseek", providers=["deepseek", "qwen"])
    prompts = _answers(monkeypatch, ["d", "2", "y", "q"])
    accounts_menu(s)

    # The confirmation must name the provider: "a@x.com" alone cannot.
    assert any("Delete 'a@x.com' [qwen]" in p for p in prompts)
    assert len(s.accounts) == 1
    assert s.accounts[0].provider == "deepseek"
    assert s.accounts[0].token == "t1"


def test_delete_by_identifier_still_works(monkeypatch):
    """The typed-identifier fallback is unchanged: first match goes."""
    from deepseaport.tui import accounts_menu

    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1"),
                         AccountConfig(email="b@x.com", token="t2")])
    _answers(monkeypatch, ["d", "b@x.com", "y", "q"])
    accounts_menu(s)
    assert [a.email for a in s.accounts] == ["a@x.com"]


def test_delete_shared_email_keeps_active_selection_valid(monkeypatch):
    """Deleting one of two same-email accounts must not orphan CURRENT."""
    from deepseaport.tui import accounts_menu

    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        active_account="a@x.com", provider="deepseek", providers=["deepseek", "qwen"])
    _answers(monkeypatch, ["d", "1", "y", "q"])
    accounts_menu(s)
    assert len(s.accounts) == 1
    # The surviving qwen account still satisfies the selection by identifier.
    assert s.active_account == "a@x.com"


def test_set_token_row_two_targets_that_provider(monkeypatch):
    """`t` by row number must not overwrite the other provider's token."""
    from deepseaport.tui import accounts_menu

    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        active_account="", provider="deepseek", providers=["deepseek", "qwen"])
    _answers(monkeypatch, ["t", "2", "fresh-token-2", "q"])
    accounts_menu(s)
    assert [a.token for a in s.accounts] == ["t1", "fresh-token-2"]


# --- C4: SystemExit from a serve path must return to the menu ----------------

def test_main_menu_survives_systemexit_from_serve(monkeypatch, capsys):
    """uvicorn calls sys.exit(3) when bind() fails: a BaseException that killed
    the whole TUI with no menu. Pre-change SystemExit escaped main_menu.
    """
    from deepseaport.tui import main_menu

    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")],
               provider="deepseek", providers=[])

    def _serve(_s):
        raise SystemExit(3)  # uvicorn STARTUP_FAILURE

    # Quit lives in the provider chooser: "q. Go back" opens it, "q. Quit" quits.
    _answers(monkeypatch, ["1", "q", "q"])
    assert main_menu(s, _serve) == 0
    out = capsys.readouterr().out
    assert "code 3" in out
    assert "Port already in use" in out


def test_main_menu_clean_systemexit_is_not_an_error(monkeypatch, capsys):
    """SystemExit(0) is a normal stop: no error text, no exit code in the log."""
    from deepseaport.tui import main_menu

    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")],
               provider="deepseek", providers=[])

    def _serve(_s):
        raise SystemExit(0)

    _answers(monkeypatch, ["1", "q", "q"])
    assert main_menu(s, _serve) == 0
    out = capsys.readouterr().out
    assert "back to menu" in out
    assert "exited with code" not in out


def test_main_menu_systemexit_from_chat_multi_and_dry(monkeypatch, capsys):
    """All four serve blocks are guarded, not just option 1.

    Distinct exit codes per block so each guard is proved individually.
    """
    from deepseaport.tui import main_menu

    def _mk(code):
        def _fn(_s):
            raise SystemExit(code)
        return _fn

    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")],
               provider="deepseek", providers=[])
    # Menu with chat: 1 serve, 2 chat, 3 accounts, 4 settings, 5 multi, 6 dry.
    # Quit lives in the provider chooser: q ("Go back"), then q ("Quit").
    _answers(monkeypatch, ["2", "5", "6", "q", "q"])
    assert main_menu(s, _mk(1), _mk(2), _mk(3), _mk(4)) == 0
    out = capsys.readouterr().out
    assert "Chat server exited with code 2" in out
    assert "Multi-server exited with code 3" in out
    assert "Dry run exited with code 4" in out


# --- C5: accountless providers must not be gated on accounts -----------------

class _Boom(Exception):
    pass


def test_multi_chat_starts_without_accounts_for_accountless_provider(monkeypatch, capsys):
    """opencode runs a local CLI: an empty pool is not a reason to refuse.

    Pre-change this printed "No accounts yet. Add one first." and returned 1,
    contradicting tui._preflight_reason which correctly allows the start.
    """
    srv, _thr = _fake_server(monkeypatch)
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live",
                        lambda st: ([AccountConfig(email="a@x.com", token="t1")], 0, 0))
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)

    def _repl(*a, **k):
        raise _Boom

    monkeypatch.setattr(chat_ui.chat_repl, "run_repl", _repl)
    s = _fresh(accounts=[], provider="opencode", providers=[], port=54444)

    with pytest.raises(_Boom):  # reached the REPL => the guard let us through
        chat_ui.run_multi_chat_servers(s, count=1)
    assert "No accounts yet" not in capsys.readouterr().out
    assert srv.should_exit is True  # and the finally still cleaned up


def test_multi_chat_still_gates_a_provider_that_needs_accounts(monkeypatch, capsys):
    """The gate must survive for DeepSeek, which really does need a token."""
    s = _fresh(accounts=[], provider="deepseek", providers=[])
    assert chat_ui.run_multi_chat_servers(s, count=1) == 1
    assert "No accounts yet" in capsys.readouterr().out


# --- C6: the ban state must land on the account that is actually added -------

def test_add_account_persists_ban_reported_by_auto_login(monkeypatch, capsys):
    """Auto-login ran BEFORE the AccountConfig was appended, so its by-email
    ban write matched nothing: the UI said "Account list will show the ban"
    while the saved account kept banned=False.
    """
    from deepseaport.tui import accounts_menu

    monkeypatch.setattr("deepseaport.providers.deepseek.auth.obscura_login_token",
                        lambda email, password, settings:
                        ("banned-tok-123", True, "suspended until 16 September 2030"))
    s = _fresh(accounts=[], active_account="", provider="deepseek", providers=[])
    _answers(monkeypatch, ["a", "new@x.com", "pw", "y", "q"])
    accounts_menu(s)

    out = capsys.readouterr().out
    assert "BANNED" in out
    assert len(s.accounts) == 1
    assert s.accounts[0].token == "banned-tok-123"
    assert s.accounts[0].banned is True
    assert s.accounts[0].banned_until > 0.0


def test_add_account_ban_survives_a_save_reload(tmp_path, monkeypatch):
    """The ban has to be on disk, not just in memory."""
    from deepseaport.config import load_settings
    from deepseaport.tui import accounts_menu

    monkeypatch.setattr("deepseaport.providers.deepseek.auth.obscura_login_token",
                        lambda email, password, settings:
                        ("banned-tok-123", True, "suspended until 16 September 2030"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"keys": [], "accounts": []}), encoding="utf-8")
    s = load_settings(str(cfg))
    _answers(monkeypatch, ["a", "new@x.com", "pw", "y", "q"])
    accounts_menu(s)

    on_disk = json.loads(cfg.read_text(encoding="utf-8"))["accounts"]
    assert len(on_disk) == 1
    assert on_disk[0]["banned"] is True
    assert on_disk[0]["banned_until"] > 0.0


# --- C7: `chat` must report failures, not traceback --------------------------

def _patch_bridge(monkeypatch):
    """No browser: stub the WAF bridge so warmup cannot launch Obscura."""
    import deepseaport.core.obscura_bridge as bridge_mod
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
    return fake_app


def test_chat_empty_pool_reports_short_error_not_traceback(tmp_path, monkeypatch, capsys):
    """Pre-change ValueError("no DeepSeek accounts in pool") escaped main().

    The frozen exe then appended "Press Enter to close..." to the traceback.
    """
    from deepseaport.cli import main as cli_main

    _patch_bridge(monkeypatch)
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "chat", "--prompt", "hi"]) == 1
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert "no DeepSeek accounts in pool" in out
    assert "userToken needed per account" in out


def test_chat_unknown_model_reports_http_status(tmp_path, monkeypatch, capsys):
    """_prepare raises HTTPException(404) for an unknown model: also a traceback."""
    from deepseaport.cli import main as cli_main

    _patch_bridge(monkeypatch)
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok123"}]})
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi",
                   "--model", "no-such-model-xyz"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Traceback" not in out
    assert "404" in out


def test_chat_acquire_timeout_reports_short_error(tmp_path, monkeypatch, capsys):
    """A TimeoutError from a saturated pool must be a line, not a stack dump."""
    import deepseaport.accounts as acc_mod
    from deepseaport.cli import main as cli_main

    _patch_bridge(monkeypatch)

    def _timeout(self, timeout=30, allow_failover=True):
        raise TimeoutError("all slots busy")

    monkeypatch.setattr(acc_mod.AccountPool, "acquire", _timeout)
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok123"}]})
    rc = cli_main(["--config", cfg, "chat", "--prompt", "hi"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Traceback" not in out
    assert "timed out" in out


def test_chat_happy_path_output_is_unchanged(tmp_path, monkeypatch, capsys):
    """The guard must not touch the success output or its return code."""
    from deepseaport.cli import main as cli_main

    _patch_bridge(monkeypatch)
    import deepseaport.accounts as acc_mod
    import deepseaport.providers.deepseek.completion_service as ds_cs

    monkeypatch.setattr(acc_mod.AccountPool, "acquire",
                        lambda self, timeout=30, allow_failover=True: object())
    monkeypatch.setattr(ds_cs, "_complete_with_failover_sync",
                        lambda app, pool, prep, item, **k: {"content": "hello",
                                                            "thinking": "think-stuff"})
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "tok123"}]})
    assert cli_main(["--config", cfg, "chat", "--prompt", "hi"]) == 0
    assert capsys.readouterr().out == "THINKING:\nthink-stuff\n\nhello\n"
