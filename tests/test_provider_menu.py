"""Phase 5: provider gate, flags, and account binding (no network)."""

from __future__ import annotations

import argparse
import json

from deepseaport.config import AccountConfig, Settings, load_settings
from deepseaport.tui import _preflight_reason, main_menu, provider_menu


def _fresh(**kw):
    kw.setdefault("config_path", "")
    return Settings(**kw)


def _cfg(tmp_path, payload: dict) -> str:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


# 1. provider_menu single select + save + EOF keeps current --------------------

def test_provider_menu_single_select_saves(monkeypatch, tmp_path):
    cfg = tmp_path / "c.json"
    s = Settings(provider="deepseek", providers=[], config_path=str(cfg))
    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    provider_menu(s)
    assert s.provider == "deepseek"
    assert s.providers == []
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["provider"] == "deepseek"


def test_provider_menu_eof_keeps_current(monkeypatch):
    s = _fresh(provider="deepseek", providers=[])

    def _boom(*a, **k):
        raise EOFError

    monkeypatch.setattr("builtins.input", _boom)
    provider_menu(s)  # must not raise
    assert s.provider == "deepseek"


# 2. invalid input re-prompts ---------------------------------------------------

def test_provider_menu_invalid_reprompts(monkeypatch):
    s = _fresh(provider="deepseek", providers=[])
    inputs = iter(["zzz", "1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    provider_menu(s)  # must not raise, ends on q
    assert s.provider == "deepseek"


def test_provider_menu_multi_select(monkeypatch):
    from deepseaport.core import registry

    s = _fresh(provider="deepseek", providers=[])
    ids = sorted(registry.registered().keys())
    assert ids, "need at least one registered provider"
    inputs = iter(["m", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    provider_menu(s)
    assert sorted(s.providers) == ids
    assert s.provider == ids[0]


# 3. cmd_serve --provider flag + unknown ----------------------------------------

def test_cmd_serve_provider_flag_skips_gate_and_saves(tmp_path, monkeypatch, capsys):
    import sys

    import deepseaport.cli as cli_mod
    import deepseaport.tui as tui_mod

    cfg = _cfg(tmp_path, {"keys": [], "accounts": []})
    calls = {"menu": 0, "gate": 0}
    monkeypatch.setattr(tui_mod, "main_menu",
                        lambda st, *a, **k: calls.__setitem__("menu", 1) or 0)
    monkeypatch.setattr(tui_mod, "provider_menu",
                        lambda st: calls.__setitem__("gate", 1))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=False, cmd="serve",
                              provider="deepseek", providers=None)
    assert cli_mod.cmd_serve(args) == 0
    assert calls == {"menu": 1, "gate": 0}
    assert load_settings(cfg).provider == "deepseek"
    capsys.readouterr()


def test_cmd_serve_unknown_provider_returns_2(tmp_path, capsys):
    import deepseaport.cli as cli_mod

    cfg = _cfg(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=True, cmd="serve",
                              provider="nope", providers=None)
    assert cli_mod.cmd_serve(args) == 2
    assert "unknown provider: nope" in capsys.readouterr().out


def test_cmd_serve_gate_runs_when_no_flag(tmp_path, monkeypatch):
    import sys

    import deepseaport.cli as cli_mod
    import deepseaport.tui as tui_mod

    cfg = _cfg(tmp_path, {"keys": [], "accounts": []})
    calls = {"menu": 0, "gate": 0}
    monkeypatch.setattr(tui_mod, "main_menu",
                        lambda st, *a, **k: calls.__setitem__("menu", 1) or 0)
    monkeypatch.setattr(tui_mod, "provider_menu",
                        lambda st: calls.__setitem__("gate", 1))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=False, cmd="serve",
                              provider=None, providers=None)
    assert cli_mod.cmd_serve(args) == 0
    assert calls == {"menu": 1, "gate": 1}


# 4. CLI accounts add binds provider ---------------------------------------------

def test_accounts_add_binds_selected_provider_default(tmp_path):
    from deepseaport.cli import main as cli_main

    cfg = _cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "a@x.com", "--token", "tok123"]) == 0
    assert load_settings(cfg).accounts[0].provider == "deepseek"


def test_accounts_add_provider_flag_unregistered_ok(tmp_path):
    from deepseaport.cli import main as cli_main

    cfg = _cfg(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "g@x.com", "--token", "tok123",
                     "--provider", "glm"]) == 0
    assert load_settings(cfg).accounts[0].provider == "glm"


# 5. TUI add binds ----------------------------------------------------------------

def test_tui_add_binds_selected_provider(monkeypatch):
    from deepseaport.tui import accounts_menu

    s = Settings(accounts=[], active_account="", config_path="",
                 provider="deepseek")
    inputs = iter(["a", "new@x.com", "pw", "tok123", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert len(s.accounts) == 1
    assert s.accounts[0].provider == "deepseek"


# 6. preflight ----------------------------------------------------------------------

def test_preflight_blocks_empty_tokenless_passes_token():
    assert _preflight_reason(_fresh(accounts=[])) != ""
    r = _preflight_reason(_fresh(accounts=[AccountConfig(email="a@x.com", token="")]))
    assert r != "" and "token" in r.lower()
    assert _preflight_reason(
        _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])) == ""


# 7. main_menu header -----------------------------------------------------------------

def test_main_menu_shows_provider_line(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    monkeypatch.setattr("builtins.input", lambda *a, **k: "q")
    assert main_menu(s, lambda st: 0) == 0
    out = capsys.readouterr().out
    assert "1. Start server" in out
    assert "Provider" in out
    # "1. Start server" stays the first menu item.
    assert out.index("1. Start server") < out.index("q. Quit")
