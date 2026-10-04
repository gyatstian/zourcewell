"""Phase 5: provider gate, flags, and account binding (no network)."""

from __future__ import annotations

import argparse
import json

from functools import partial
from support import settings, write_config
from deepseaport.config import AccountConfig, Settings, load_settings
from deepseaport.tui import (
    _login_banner,
    _preflight_reason,
    _provider_label,
    main_menu,
    provider_menu,
)


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_fresh = partial(settings, config_path="",
                  log_level="INFO", warmup_on_startup=True)


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

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
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

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, host="", port=0, workers=1,
                              no_tui=True, cmd="serve",
                              provider="nope", providers=None)
    assert cli_mod.cmd_serve(args) == 2
    assert "unknown provider: nope" in capsys.readouterr().out


def test_cmd_serve_gate_runs_when_no_flag(tmp_path, monkeypatch):
    import sys

    import deepseaport.cli as cli_mod
    import deepseaport.tui as tui_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
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

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "a@x.com", "--token", "tok123"]) == 0
    assert load_settings(cfg).accounts[0].provider == "deepseek"


def test_accounts_add_provider_flag_unregistered_ok(tmp_path):
    from deepseaport.cli import main as cli_main

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "g@x.com", "--token", "tok123",
                     "--provider", "future-provider"]) == 0
    assert load_settings(cfg).accounts[0].provider == "future-provider"


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


# 8. provider-aware login prompts (C8) ------------------------------------------------

def test_provider_label_and_login_banner_follow_the_registry():
    # DeepSeek keeps its historical strings; other providers stop claiming to
    # be DeepSeek and stop promising a browser they never launch.
    assert _provider_label("deepseek") == "DeepSeek"
    assert _provider_label("qwen") == "Qwen"
    assert _login_banner("deepseek") == "Running Obscura login (browser, ~30s)..."
    assert "Obscura" not in _login_banner("qwen")
    assert "browser" not in _login_banner("qwen")
    assert "Qwen" in _login_banner("qwen")
    # Unknown id must still produce something printable, never an empty label.
    assert _provider_label("nope-xyz") == "nope-xyz"


def test_tui_auto_login_prompt_names_the_selected_provider(monkeypatch, capsys):
    """Pre-change the TUI asked for "DeepSeek password" and printed
    "Running Obscura login (browser, ~30s)" even with --provider qwen.
    """
    import deepseaport.providers.deepseek.auth as _ds
    from deepseaport.tui import accounts_menu

    prompts: list[str] = []

    def _gp(p=""):
        # Skip only the outer add-form password ("Password (Enter=skip): ") so
        # the auto-login step has to ask for one itself: that inner prompt's
        # wording is what is under test.
        prompts.append(p)
        return "" if p == "Password (Enter=skip): " else "pw"

    monkeypatch.setattr("getpass.getpass", _gp)
    monkeypatch.setattr(_ds, "obscura_login_token",
                        lambda e, p, s: "tok-from-login")
    s = _fresh(accounts=[], active_account="", provider="qwen", providers=[])
    inputs = iter(["a", "new@x.com", "y", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert any("Qwen password" in p for p in prompts)
    assert not any("DeepSeek password" in p for p in prompts)
    out = capsys.readouterr().out
    assert "Running Obscura login (browser" not in out
    assert "Qwen" in out


def _run_login(monkeypatch, tmp_path, argv_extra, provider):
    """Drive cmd_login and return (prompts, exit code)."""
    import deepseaport.cli as cli_mod
    from deepseaport.core import registry as reg

    prompts: list[str] = []

    def _inp(prompt=""):
        prompts.append(prompt)
        return "a@x.com"

    class _Prov:
        id = provider
        # Real registry display names: DeepSeek (not "Deepseek").
        display_name = "DeepSeek" if provider == "deepseek" else "Qwen"
        requires_accounts = True

        def login(self, email, password, settings):
            return ("tok1", False, "")

    monkeypatch.setattr("builtins.input", _inp)
    monkeypatch.setattr("getpass.getpass", lambda p="": prompts.append(p) or "pw")
    # C9: the secret is only prompted for when a console is available.
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(reg, "get", lambda pid: _Prov())
    monkeypatch.setattr(cli_mod, "_unpack_login_token_result",
                        lambda raw: raw if isinstance(raw, tuple) else (None, False, ""))
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("tok1", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, email=None, password=None,
                              provider=provider)
    return prompts, cli_mod.cmd_login(args), argv_extra


def test_cli_login_prompts_stay_byte_identical_for_deepseek(monkeypatch, tmp_path):
    """Existing callers pin these exact strings: do not reword DeepSeek."""
    prompts, rc, _ = _run_login(monkeypatch, tmp_path, [], "deepseek")
    assert prompts[0] == "DeepSeek email: "
    assert prompts[1] == "DeepSeek password: "
    assert rc == 0


def test_cli_login_prompts_name_a_non_deepseek_provider(monkeypatch, tmp_path):
    """Pre-change both prompts said "DeepSeek" regardless of --provider."""
    prompts, rc, _ = _run_login(monkeypatch, tmp_path, [], "qwen")
    assert prompts[0] == "Qwen email: "
    assert prompts[1] == "Qwen password: "
    assert rc == 0


# Finding 8: `login --provider <unregistered>` drove the DeepSeek browser -------

def _run_login_unknown_provider(monkeypatch, tmp_path):
    """Drive cmd_login with a SHAPE-valid but unregistered --provider.

    ``_valid_provider_id`` is ``^[a-z0-9-]+$``, so ``glm`` passes it. Unlike
    ``_run_login`` above this does NOT stub ``registry.get`` into always
    answering -- the whole point is that it must answer None here.
    """
    import deepseaport.cli as cli_mod
    from deepseaport.core import registry as reg

    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "a@x.com")
    monkeypatch.setattr("getpass.getpass", lambda p="": prompts.append(p) or "pw")
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)

    entered: list[tuple] = []
    monkeypatch.setattr(reg, "get", lambda pid: None)
    # Both login drivers, so "a browser was NOT opened" is a real assertion.
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: entered.append(("obscura", e, p))
                        or ("deepseek-usertoken", False, ""))
    monkeypatch.setattr(
        "deepseaport.providers.deepseek.auth.obscura_login_token",
        lambda e, p, s: entered.append(("real_browser", e, p))
        or ("deepseek-usertoken", False, ""))

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, email=None, password=None,
                              provider="glm")
    return prompts, entered, cli_mod.cmd_login(args), cfg


def test_cli_login_refuses_an_unregistered_provider(tmp_path, monkeypatch, capsys):
    """Pre-change: opened a ~30s DeepSeek Obscura window, then stored DeepSeek's
    userToken under provider="glm" -- an account no UI lists and no request can
    serve. Now: exit 2, the same refusal serve/chat already print, no driver."""
    from support import on_disk

    prompts, entered, rc, cfg = _run_login_unknown_provider(monkeypatch, tmp_path)

    assert rc == 2
    assert entered == [], f"a DeepSeek login driver was invoked: {entered!r}"
    assert prompts == [], f"credentials were prompted for anyway: {prompts!r}"
    out = capsys.readouterr().out
    assert "unknown provider: glm" in out
    # The available list is what makes the refusal actionable.
    for pid in ("deepseek", "qwen"):
        assert pid in out
    assert on_disk(cfg)["accounts"] == [], "an account was persisted anyway"


def test_tui_login_refuses_an_unregistered_provider_and_stays_alive(
        monkeypatch, capsys):
    """The TUI half of finding 8, through its real entry point.

    ``tui.py:1241-1276`` is a byte-identical shape to the CLI one: an id the
    registry does not know returned ``None``, so the else branch drove the
    DeepSeek Obscura browser -- while ``_login_banner`` had already promised
    "Running glm login (direct API signin, ~5s)". The TUI cannot exit 2, so the
    refusal is in-app: no browser, no token persisted, and the menu stays alive.
    """
    from deepseaport.core import registry as reg
    from deepseaport.tui import accounts_menu

    # Only "glm" is unknown; everything else resolves normally, so the refusal
    # is about the id and not a stubbed registry.
    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "glm" else real_get(pid))

    entered: list[tuple] = []
    monkeypatch.setattr(
        "deepseaport.providers.deepseek.auth.obscura_login_token",
        lambda e, p, s: entered.append(("real_browser", e, p))
        or ("deepseek-usertoken", False, ""))

    s = _fresh(provider="glm",
               accounts=[AccountConfig(email="a@x.com", password="pw",
                                       token="", provider="glm")])
    # t=set token, email, ""=no token -> auto-login, ""=confirm, b=back.
    inputs = iter(["t", "a@x.com", "", "", "b"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

    accounts_menu(s)

    assert entered == [], f"a DeepSeek login driver was invoked: {entered!r}"
    assert s.accounts[0].token == "", "a DeepSeek token was stored under glm"
    out = capsys.readouterr().out
    assert "unknown provider: glm" in out
    # The refusal lands BEFORE the banner, so it never promises a browser.
    assert "Running glm login" not in out


def test_cli_login_still_reaches_deepseek_and_qwen_browsers(
        monkeypatch, tmp_path, capsys):
    """Control for the refusal: the default must keep driving Obscura.

    ``deepseek`` is exempt from the membership check because its login IS the
    direct browser path, not a registry entry point -- so the gate must not
    have broken the default that every existing caller uses.
    """
    import deepseaport.cli as cli_mod
    from deepseaport.core import registry as reg

    seen: dict = {}
    monkeypatch.setattr("builtins.input", lambda p="": "a@x.com")
    monkeypatch.setattr("getpass.getpass", lambda p="": "pw")
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: seen.update(provider="deepseek",
                                                    driver="obscura")
                        or ("tok-ds", False, ""))

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "glm" else real_get(pid))

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="pw", provider="deepseek")) == 0
    assert seen == {"provider": "deepseek", "driver": "obscura"}
