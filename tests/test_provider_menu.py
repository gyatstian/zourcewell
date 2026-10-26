"""Phase 5: provider gate, flags, and account binding (no network)."""

from __future__ import annotations

import argparse
import json

from functools import partial
from support import settings, write_config
from zourcewell.accounts import provider_label
from zourcewell.config import AccountConfig, Settings, load_settings
from zourcewell.tui import (
    _login_banner,
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
    provider_menu(s)  # must not raise; the valid pick ends the loop
    assert s.provider == "deepseek"


def test_provider_menu_multi_select(monkeypatch):
    from zourcewell.core import registry

    s = _fresh(provider="deepseek", providers=[])
    ids = sorted(registry.registered().keys())
    assert ids, "need at least one registered provider"
    inputs = iter(["m", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    provider_menu(s)
    assert sorted(s.providers) == ids
    assert s.provider == ids[0]


# 2b. menu rows draw no catalogue and no counts --------------------------------


def _probe_provider():
    """Provider that records catalogue calls, with models() rigged to explode.

    ``models()`` may be a live read that BOOTS a local process (opencode's
    managed serve), so a menu either asks ``models_hint()`` -- the in-process
    view -- or does not ask at all.
    """
    from zourcewell.core.provider import ModelSpec, Provider

    class _Probe(Provider):
        id = "hintonly"
        display_name = "HintOnly"
        hint_calls = 0

        def models(self):
            raise AssertionError("a menu called models(), which may boot")

        def models_hint(self):
            type(self).hint_calls += 1
            return {"hintonly/one": ModelSpec(id="hintonly/one",
                                              owned_by="hintonly")}

    return _Probe()


def test_provider_menu_row_is_a_label_only(monkeypatch, capsys):
    """A row is a label: no id, no model count, no account/token count.

    The id is already on the mode line above; the counts were derived values
    nobody picks a provider by, and the model count cost a catalogue read
    per row.
    """
    from zourcewell.core import registry

    fake = _probe_provider()
    registry._REGISTRY[registry._key(fake)] = fake
    try:
        s = _fresh(provider="deepseek", providers=[])
        monkeypatch.setattr("builtins.input", lambda *a, **k: "q")
        provider_menu(s)
    finally:
        registry._REGISTRY.pop("hintonly", None)
    row = next(l for l in capsys.readouterr().out.splitlines() if "HintOnly" in l)
    assert row.split(". ", 1)[1] == "HintOnly"
    assert fake.hint_calls == 0, "the row read a catalogue"


# 3. cmd_serve --provider flag + unknown ----------------------------------------

def test_cmd_serve_provider_flag_skips_gate_and_saves(tmp_path, monkeypatch, capsys):
    import sys

    import zourcewell.cli as cli_mod
    import zourcewell.tui as tui_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    calls = {"menu": 0, "gate": 0}
    monkeypatch.setattr(tui_mod, "main_menu",
                        lambda st, *a, **k: calls.__setitem__("menu", 1) or 0)
    monkeypatch.setattr(tui_mod, "provider_menu",
                        lambda st: calls.__setitem__("gate", 1))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    args = argparse.Namespace(config=cfg, host="", port=0,
                              no_tui=False, cmd="serve",
                              provider="deepseek", providers=None)
    assert cli_mod.cmd_serve(args) == 0
    assert calls == {"menu": 1, "gate": 0}
    assert load_settings(cfg).provider == "deepseek"
    capsys.readouterr()


def test_cmd_serve_unknown_provider_returns_2(tmp_path, capsys):
    import zourcewell.cli as cli_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    args = argparse.Namespace(config=cfg, host="", port=0,
                              no_tui=True, cmd="serve",
                              provider="nope", providers=None)
    assert cli_mod.cmd_serve(args) == 2
    assert "unknown provider: nope" in capsys.readouterr().out


def test_cmd_serve_gate_runs_when_no_flag(tmp_path, monkeypatch):
    import sys

    import zourcewell.cli as cli_mod
    import zourcewell.tui as tui_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    calls = {"menu": 0, "gate": 0}
    monkeypatch.setattr(tui_mod, "main_menu",
                        lambda st, *a, **k: calls.__setitem__("menu", 1) or 0)
    monkeypatch.setattr(tui_mod, "provider_menu",
                        lambda st: calls.__setitem__("gate", 1))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    args = argparse.Namespace(config=cfg, host="", port=0,
                              no_tui=False, cmd="serve",
                              provider=None, providers=None)
    assert cli_mod.cmd_serve(args) == 0
    assert calls == {"menu": 1, "gate": 1}


def test_cmd_serve_chooser_quit_exits_cleanly(tmp_path, monkeypatch):
    """The startup chooser's "q. Quit" quits the serve: rc 0, no main menu."""
    import sys

    import zourcewell.cli as cli_mod
    import zourcewell.tui as tui_mod

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    calls = {"menu": 0}
    monkeypatch.setattr(tui_mod, "main_menu",
                        lambda st, *a, **k: calls.__setitem__("menu", 1) or 0)
    monkeypatch.setattr(tui_mod, "provider_menu", lambda st: False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    args = argparse.Namespace(config=cfg, host="", port=0,
                              no_tui=False, cmd="serve",
                              provider=None, providers=None)
    assert cli_mod.cmd_serve(args) == 0
    assert calls["menu"] == 0


# 4. CLI accounts add binds provider ---------------------------------------------

def test_accounts_add_binds_selected_provider_default(tmp_path):
    from zourcewell.cli import main as cli_main

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "a@x.com", "--token", "tok123"]) == 0
    assert load_settings(cfg).accounts[0].provider == "deepseek"


def test_accounts_add_provider_flag_unregistered_ok(tmp_path):
    from zourcewell.cli import main as cli_main

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_main(["--config", cfg, "accounts", "add",
                     "--email", "g@x.com", "--token", "tok123",
                     "--provider", "future-provider"]) == 0
    assert load_settings(cfg).accounts[0].provider == "future-provider"


# 5. TUI add binds ----------------------------------------------------------------

def test_tui_add_binds_selected_provider(monkeypatch):
    from zourcewell.tui import accounts_menu

    s = Settings(accounts=[], active_account="", config_path="",
                 provider="deepseek")
    inputs = iter(["a", "new@x.com", "pw", "tok123", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    accounts_menu(s)
    assert len(s.accounts) == 1
    assert s.accounts[0].provider == "deepseek"


# 6. main_menu header -----------------------------------------------------------------

def test_main_menu_shows_provider_in_header(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    # Quit is the provider chooser's own "q. Quit" (Go back here, then q there).
    inputs = iter(["q", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    assert main_menu(s, lambda st: 0) == 0
    out = capsys.readouterr().out
    # The header names the active provider, the row reopens the chooser.
    assert "=== deepseek ===" in out
    assert "1. Start server" in out
    assert "q. Go back" in out
    # "1. Start server" stays the first menu item.
    assert out.index("1. Start server") < out.index("q. Go back")


# 7. provider-aware login prompts (C8) ------------------------------------------------

def test_provider_label_and_login_banner_follow_the_registry():
    # DeepSeek keeps its historical strings; other providers stop claiming to
    # be DeepSeek and stop promising a browser they never launch.
    assert provider_label("deepseek") == "DeepSeek"
    assert provider_label("qwen") == "Qwen"
    assert _login_banner("deepseek") == "Running Obscura login (browser, ~30s)..."
    assert "Obscura" not in _login_banner("qwen")
    assert "browser" not in _login_banner("qwen")
    assert "Qwen" in _login_banner("qwen")
    # Unknown id must still produce something printable, never an empty label.
    assert provider_label("nope-xyz") == "nope-xyz"


def test_tui_auto_login_prompt_names_the_selected_provider(monkeypatch, capsys):
    """Pre-change the TUI asked for "DeepSeek password" and printed
    "Running Obscura login (browser, ~30s)" even with --provider qwen.
    """
    import zourcewell.providers.deepseek.auth as _ds
    from zourcewell.tui import accounts_menu

    prompts: list[str] = []

    def _inp(prompt=""):
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr(_ds, "obscura_login_token",
                        lambda e, p, s: "tok-from-login")
    s = _fresh(accounts=[], active_account="", provider="qwen", providers=[])
    # a=add, email, ""=skip the add-form password so the auto-login step has
    # to ask for one itself (that inner prompt's wording is what is under
    # test), y=auto-token, pw=the inner password answer, q=back.
    inputs = iter(["a", "new@x.com", "", "y", "pw", "q"])
    monkeypatch.setattr("builtins.input", _inp)
    accounts_menu(s)
    # Both password fields are visible prompts now, so the wording comes
    # through builtins.input rather than getpass.
    assert any("Qwen password" in p for p in prompts)
    assert not any("DeepSeek password" in p for p in prompts)
    out = capsys.readouterr().out
    assert "Running Obscura login (browser" not in out
    assert "Qwen" in out


def _run_login(monkeypatch, tmp_path, argv_extra, provider):
    """Drive cmd_login and return (prompts, exit code)."""
    import zourcewell.cli as cli_mod
    from zourcewell.core import registry as reg

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
    # 4th arg = the provider hint the explicit --provider flag now carries.
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s, prov=None: ("tok1", False, ""))
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
    import zourcewell.cli as cli_mod
    from zourcewell.core import registry as reg

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
        "zourcewell.providers.deepseek.auth.obscura_login_token",
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
    # The available list is what makes the refusal actionable, and it IS
    # `sorted(registered().keys())` (cli._unknown_provider_exit). A hand-copied
    # pair drifts every time a provider lands -- opencode already had.
    from zourcewell.core import registry

    known = sorted(registry.registered().keys())
    assert known, "need a registered provider to assert the list against"
    for pid in known:
        assert pid in out
    assert on_disk(cfg)["accounts"] == [], "an account was persisted anyway"


def test_tui_login_refuses_an_unregistered_provider_and_stays_alive(
        monkeypatch, capsys):
    """The TUI half of finding 8, through its real entry point.

    ``tui.py:1096-1099`` is a byte-identical shape to the CLI one: an id the
    registry does not know returned ``None``, so the else branch drove the
    DeepSeek Obscura browser -- while ``_login_banner`` had already promised
    "Running glm login (direct API signin, ~5s)". The TUI cannot exit 2, so the
    refusal is in-app: no browser, no token persisted, and the menu stays alive.
    """
    from zourcewell.core import registry as reg
    from zourcewell.tui import accounts_menu

    # Only "glm" is unknown; everything else resolves normally, so the refusal
    # is about the id and not a stubbed registry.
    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "glm" else real_get(pid))

    entered: list[tuple] = []
    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token",
        lambda e, p, s: entered.append(("real_browser", e, p))
        or ("deepseek-usertoken", False, ""))

    s = _fresh(provider="glm",
               accounts=[AccountConfig(email="a@x.com", password="pw",
                                       token="", provider="glm")])
    # t=set token, email, ""=no token -> the auto-login offer refuses in-app
    # (unknown provider) before it can consume anything, q=back.
    inputs = iter(["t", "a@x.com", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

    accounts_menu(s)

    assert entered == [], f"a DeepSeek login driver was invoked: {entered!r}"
    assert s.accounts[0].token == "", "a DeepSeek token was stored under glm"
    out = capsys.readouterr().out
    assert "unknown provider: glm" in out
    # The refusal lands BEFORE the banner, so it never promises a browser.
    assert "Running glm login" not in out


# TUI login parity: the two steps cli.cmd_login does after a successful login --


def _jwt(exp: int) -> str:
    """Minimal unsigned JWT: only the ``exp`` claim matters to jwt_exp."""
    import base64

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_b64({'alg': 'none'})}.{_b64({'exp': exp})}.sig"


def _tui_qwen_login(monkeypatch, token: str, banned: bool = False,
                    detail: str = "") -> None:
    """Registry login seam for qwen, with the DeepSeek driver rigged to explode."""
    import zourcewell.core.registry as reg

    class _FakeQwen:
        def login(self, email, password, settings):
            return (token, banned, detail)

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))
    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the DeepSeek driver ran for a qwen login")))


def test_tui_added_qwen_account_keeps_the_signin_expiry(monkeypatch):
    """cli.cmd_login recovers the signin expiry from the token (jwt_exp, because
    the registry ``login()`` seam drops ``login_with_expiry``'s own value) and
    passes it into ``_upsert_account``. The TUI built its AccountConfig without
    it, so a Qwen account added here kept ``token_expires_at`` 0.0 and the
    documented proactive re-signin could never fire for it."""
    from zourcewell.tui import accounts_menu

    token = _jwt(1789555791)
    _tui_qwen_login(monkeypatch, token)

    s = _fresh(provider="qwen", providers=[], accounts=[])
    inputs = iter(["a", "q@x.com", "pw", "y", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

    accounts_menu(s)

    assert [(a.provider, a.token, a.token_expires_at) for a in s.accounts] == [
        ("qwen", token, 1789555791.0)]


def test_tui_set_token_qwen_keeps_the_signin_expiry(monkeypatch):
    """The same guarantee on the set-token path, the TUI's other token write."""
    from zourcewell.tui import accounts_menu

    token = _jwt(1789555791)
    _tui_qwen_login(monkeypatch, token)

    s = _fresh(provider="qwen", providers=[], accounts=[
        AccountConfig(email="q@x.com", password="pw", token="old-tok",
                      provider="qwen")])
    # t = set token, email, "" = no token -> auto-login offer, Enter accepts, q.
    inputs = iter(["t", "q@x.com", "", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

    accounts_menu(s)

    assert (s.accounts[0].token, s.accounts[0].token_expires_at) == (
        token, 1789555791.0)


def test_tui_banned_qwen_login_keeps_the_probe_expiry(monkeypatch):
    """cli refines a banned login's expiry through ``check_ban_for_token``
    (Provider.probe_ban, so a Qwen token is probed as Qwen) instead of trusting
    the login page's wording alone. The TUI took only the wording, so a ban with
    no parseable date was persisted with ``banned_until`` 0.0."""
    import zourcewell.accounts as acc_mod
    from zourcewell.tui import accounts_menu

    _tui_qwen_login(monkeypatch, _jwt(1789555791), banned=True,
                    detail="suspended")
    seen: list = []

    def _probe(token, timeout=12, provider="deepseek"):
        seen.append(provider)
        return True, 4000000000.0

    monkeypatch.setattr(acc_mod, "check_ban_for_token", _probe)

    s = _fresh(provider="qwen", providers=[], accounts=[])
    inputs = iter(["a", "q@x.com", "pw", "y", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

    accounts_menu(s)

    assert seen == ["qwen"], "the ban probe was not scoped to the login provider"
    assert (s.accounts[0].banned, s.accounts[0].banned_until) == (
        True, 4000000000.0)


def test_cli_login_still_reaches_deepseek_and_qwen_browsers(
        monkeypatch, tmp_path, capsys):
    """Control for the refusal: the default must keep driving Obscura.

    ``deepseek`` is exempt from the membership check because its login IS the
    direct browser path, not a registry entry point -- so the gate must not
    have broken the default that every existing caller uses.
    """
    import zourcewell.cli as cli_mod
    from zourcewell.core import registry as reg

    seen: dict = {}
    monkeypatch.setattr("builtins.input", lambda p="": "a@x.com")
    monkeypatch.setattr("getpass.getpass", lambda p="": "pw")
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s, hint=None: seen.update(provider="deepseek",
                                                               driver="obscura",
                                                               hint=hint)
                        or ("tok-ds", False, ""))

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "glm" else real_get(pid))

    cfg = write_config(tmp_path, {"keys": [], "accounts": []})
    assert cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="pw", provider="deepseek")) == 0
    assert seen == {"provider": "deepseek", "driver": "obscura",
                    "hint": "deepseek"}


# Finding 2: --provider must pin BOTH the driver that runs and the provider
# the token is stored under -------------------------------------------------------

def test_login_provider_flag_deepseek_pins_the_driver_hint(monkeypatch, tmp_path):
    """Pre-change: settings.provider=qwen + ``--provider deepseek`` fell into
    the flag-less branch, ``_obscura_login_token`` resolved settings.provider
    and drove the QWEN signin, then stored the Qwen JWT under deepseek."""
    import zourcewell.cli as cli_mod

    seen: list = []

    def _driver(e, p, s, prov=None):
        seen.append(prov)
        return ("ds-tok", False, "")

    monkeypatch.setattr(cli_mod, "_obscura_login_token", _driver)
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "provider": "qwen"})

    assert cli_mod.main(["--config", cfg, "login", "--email", "a@x.com",
                         "--password", "pw", "--provider", "deepseek"]) == 0

    assert seen == ["deepseek"], "the explicit provider never reached the driver"
    acc = load_settings(cfg).accounts[0]
    assert (acc.token, acc.provider) == ("ds-tok", "deepseek")


def test_login_provider_flag_deepseek_never_reaches_configured_qwen(
        monkeypatch, tmp_path):
    """The shipped wrapper half: the Qwen registry login must not run for
    ``--provider deepseek`` even when the config selects qwen."""
    import zourcewell.cli as cli_mod
    from zourcewell.core import registry as reg

    calls: list = []
    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token",
        lambda e, p, s: calls.append(("deepseek", e, p)) or ("ds-tok", False, ""))
    qwen_calls: list = []

    class _FakeQwen:
        def login(self, email, password, settings):
            qwen_calls.append(email)
            return ("qwen-jwt", False, "")

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "provider": "qwen"})

    assert cli_mod.main(["--config", cfg, "login", "--email", "a@x.com",
                         "--password", "pw", "--provider", "deepseek"]) == 0

    assert qwen_calls == [], "the Qwen signin ran for --provider deepseek"
    assert calls == [("deepseek", "a@x.com", "pw")]
    assert load_settings(cfg).accounts[0].provider == "deepseek"


def test_login_flagless_uses_and_stamps_the_configured_qwen(monkeypatch, tmp_path):
    """Flag-less login resolves settings.provider (qwen) and must persist the
    SAME provider the token came from, not the DeepSeek default."""
    import zourcewell.cli as cli_mod
    from zourcewell.core import registry as reg

    class _FakeQwen:
        def login(self, email, password, settings):
            return ("qwen-jwt", False, "")

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))

    def _no_ds(*a, **k):
        raise AssertionError("the DeepSeek driver ran for a qwen login")

    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token", _no_ds)
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "provider": "qwen"})

    assert cli_mod.main(["--config", cfg, "login", "--email", "q@x.com",
                         "--password", "pw"]) == 0

    acc = load_settings(cfg).accounts[0]
    assert (acc.token, acc.provider) == ("qwen-jwt", "qwen")


# Finding 2: the upsert must not cross providers (same email, two records) ------


def _two_provider_config(tmp_path) -> str:
    """One email on both providers: account_identity includes the provider, so
    both records are legitimate and distinct."""
    return write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "same@x.com", "token": "ds-old", "provider": "deepseek"},
        {"email": "same@x.com", "token": "qw-old", "provider": "qwen"},
    ]})


def _qwen_login(monkeypatch, token: str) -> None:
    import zourcewell.core.registry as reg

    class _FakeQwen:
        def login(self, email, password, settings):
            return (token, False, "")

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: _FakeQwen() if pid == "qwen" else real_get(pid))


def test_login_provider_flag_qwen_updates_only_the_qwen_record(
        tmp_path, monkeypatch):
    """Pre-change the email-only match stamped the Qwen JWT onto the
    provider:deepseek record -- a token the DeepSeek pool would then replay."""
    import zourcewell.cli as cli_mod

    _qwen_login(monkeypatch, "qwen-jwt")
    cfg = _two_provider_config(tmp_path)

    assert cli_mod.main(["--config", cfg, "login", "--email", "same@x.com",
                         "--password", "pw", "--provider", "qwen"]) == 0

    assert {a.provider: a.token for a in load_settings(cfg).accounts} == {
        "deepseek": "ds-old", "qwen": "qwen-jwt"}


def test_login_provider_flag_deepseek_updates_only_the_deepseek_record(
        tmp_path, monkeypatch):
    """Mirror case: --provider deepseek must not touch the qwen record."""
    import zourcewell.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s, prov=None: ("ds-new", False, ""))
    cfg = _two_provider_config(tmp_path)

    assert cli_mod.main(["--config", cfg, "login", "--email", "same@x.com",
                         "--password", "pw", "--provider", "deepseek"]) == 0

    assert {a.provider: a.token for a in load_settings(cfg).accounts} == {
        "deepseek": "ds-new", "qwen": "qw-old"}


def test_login_flagless_updates_only_the_settings_provider_record(
        tmp_path, monkeypatch):
    """The flag-less form resolves the configured provider (qwen here), so the
    qwen record is the one it updates -- pinned so the scoping did not simply
    move the bug to the other side."""
    import zourcewell.cli as cli_mod

    _qwen_login(monkeypatch, "qwen-jwt")
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "same@x.com", "token": "ds-old", "provider": "deepseek"},
        {"email": "same@x.com", "token": "qw-old", "provider": "qwen"},
    ], "provider": "qwen"})

    assert cli_mod.main(["--config", cfg, "login", "--email", "same@x.com",
                         "--password", "pw"]) == 0

    assert {a.provider: a.token for a in load_settings(cfg).accounts} == {
        "deepseek": "ds-old", "qwen": "qwen-jwt"}
