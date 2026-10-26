"""TUI menu guards: failed saves stay in the menu, deletes stay in scope.

* F15  ``_select_provider`` wrapped ``settings.save()`` in EOFError/
       KeyboardInterrupt only, so an unparsable config (RuntimeError) or an
       unwritable one (OSError) killed the whole TUI when picking a provider --
       including the startup gate -- where the CLI twin (``cli._save``) prints
       one line and exits. The pick now reports and the menu survives.
* F11  ``d``/``t`` on a typed identifier called the unscoped removal / token
       write: with one email on two providers it hit the OTHER provider's
       record (the delete destroyed it, the token overwrote it).
* F13  a banned auto-login on the set-token path said only "(saved)"; it now
       names the persisted ban and points at ``accounts unblock``.
* F29  ``tui_editor`` counted combining marks as one cell, desyncing the
       cursor. This last test is here only because this module is the one the
       change was allowed to add; move it next to the other editor tests when
       one owns ``_visual_pos``.

The menus are driven through a scripted stdin (``builtins.input``), the way a
person types at a prompt, and assertions are on the screen and the settings
object -- not on internal calls.
"""

from __future__ import annotations

from functools import partial

from support import AccountConfig, settings
from zourcewell import tui, tui_editor
from zourcewell.config import Settings, load_settings


_fresh = partial(settings, config_path="", log_level="INFO",
                 warmup_on_startup=True)


def _answers(monkeypatch, seq):
    """Feed ``seq`` to every input() the menus ask for."""
    it = iter(seq)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


# --- F15: a refused/failed save must not kill the provider chooser -------------

def _unparsable(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    return p


def test_provider_pick_survives_an_unparsable_config(monkeypatch, capsys, tmp_path):
    """``load_settings`` falls back to empty in-memory settings and latches the
    file, so ``save()`` RAISES to protect the accounts on disk. That raise used
    to escape ``_select_provider`` (and its serve startup gate) as a traceback.
    """
    p = _unparsable(tmp_path)
    s = load_settings(str(p))
    _answers(monkeypatch, ["m"])

    assert tui._select_provider(s) is True  # no exception, menu stays alive

    out = capsys.readouterr().out
    assert "Provider not saved" in out
    assert "(saved)" not in out
    assert p.read_text(encoding="utf-8") == "{not json"


def test_provider_pick_survives_an_unwritable_config(monkeypatch, capsys, tmp_path):
    """The OSError side: a path component that is a file, so the write cannot
    even create the config. Both save sites share the one guard.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    s = _fresh(config_path=str(blocker / "config.json"))
    _answers(monkeypatch, ["1"])

    assert tui._select_provider(s) is True

    out = capsys.readouterr().out
    assert "Provider not saved" in out
    assert "(saved)" not in out
    assert blocker.read_text(encoding="utf-8") == "not a directory"


def test_provider_pick_single_site_also_reports_a_failed_save(
        monkeypatch, capsys, tmp_path):
    """The single-provider branch has its own copy of the save call."""
    p = _unparsable(tmp_path)
    s = load_settings(str(p))
    _answers(monkeypatch, ["1"])

    assert tui._select_provider(s) is True
    assert "Provider not saved" in capsys.readouterr().out


def test_provider_pick_keyboardinterrupt_still_cancels_quietly(
        monkeypatch, capsys):
    """Control: Ctrl+C during the save is not an error -- it cancels the screen
    exactly as before (no failure line).
    """
    s = _fresh()

    def _interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(Settings, "save", _interrupt)
    _answers(monkeypatch, ["m"])

    assert tui._select_provider(s) is True
    out = capsys.readouterr().out
    assert "Provider not saved" not in out
    assert "(saved)" not in out


# --- F11: deletion by typed identifier is provider-scoped ----------------------

def test_delete_typed_email_only_removes_the_scoped_providers_record(
        monkeypatch, capsys):
    """Single-provider scope (qwen): the deepseek record with the same email
    must survive. Pre-change the unscoped first match destroyed it.
    """
    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        provider="qwen", providers=[])
    _answers(monkeypatch, ["d", "a@x.com", "y", "q"])

    tui.accounts_menu(s)

    assert [a.provider for a in s.accounts] == ["deepseek"]
    assert s.accounts[0].token == "t1"


def test_delete_typed_email_in_multi_mode_keeps_the_first_match(monkeypatch):
    """Control: multi mode has no single provider to scope to, so the typed
    path keeps its historical first-match behavior (a row number disambiguates).
    """
    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        provider="deepseek", providers=["deepseek", "qwen"])
    _answers(monkeypatch, ["d", "a@x.com", "y", "q"])

    tui.accounts_menu(s)

    assert [a.provider for a in s.accounts] == ["qwen"]


# --- F11b: the typed-identifier TOKEN path is provider-scoped too ---------------

def test_set_token_typed_email_only_touches_the_scoped_providers_record(
        monkeypatch, capsys):
    """Single-provider scope (qwen): the deepseek record with the same email
    keeps its token. Pre-change the unscoped first match overwrote it.
    """
    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        provider="qwen", providers=[])
    _answers(monkeypatch, ["t", "a@x.com", "fresh", "q"])

    tui.accounts_menu(s)

    assert [(a.provider, a.token) for a in s.accounts] == [("deepseek", "t1"),
                                                           ("qwen", "fresh")]


def test_set_token_typed_email_in_multi_mode_keeps_the_first_match(monkeypatch):
    """Control: multi mode has no single provider to scope to, so the typed
    path keeps its historical first-match behavior (a row number disambiguates).
    """
    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", token="t1", provider="deepseek"),
                  AccountConfig(email="a@x.com", token="t2", provider="qwen")],
        provider="deepseek", providers=["deepseek", "qwen"])
    _answers(monkeypatch, ["t", "a@x.com", "fresh", "q"])

    tui.accounts_menu(s)

    assert [(a.provider, a.token) for a in s.accounts] == [("deepseek", "fresh"),
                                                           ("qwen", "t2")]


def test_token_auto_login_typed_email_uses_the_scoped_record(monkeypatch):
    """Empty token + typed email under scope qwen: the auto-login offer must
    read the QWEN record's credentials. Pre-change the unscoped lookup found
    the deepseek record first (no password of its own here), and the token it
    produced was then written onto the qwen record -- a deepseek credential in
    a qwen row.
    """
    from zourcewell.core import registry as _reg

    qwen_seen: list[tuple[str, str]] = []
    deepseek_seen: list[str] = []
    monkeypatch.setattr(_reg.get("qwen"), "login",
                        lambda email, password, settings:
                        (qwen_seen.append((email, password)) or ("tok-q", False, "")))
    monkeypatch.setattr("zourcewell.providers.deepseek.auth.obscura_login_token",
                        lambda email, password, settings:
                        (deepseek_seen.append(email) or ("tok-ds", False, "")))
    s = _fresh(
        accounts=[AccountConfig(email="a@x.com", password="pw-ds", token="t1",
                                provider="deepseek"),
                  AccountConfig(email="a@x.com", password="pw-q", token="t2",
                                provider="qwen")],
        provider="qwen", providers=[])
    _answers(monkeypatch, ["t", "a@x.com", "", "y", "q"])

    tui.accounts_menu(s)

    assert qwen_seen == [("a@x.com", "pw-q")]
    assert deepseek_seen == []
    assert [a.token for a in s.accounts] == ["t1", "tok-q"]


# --- F13: a persisted ban is named, never auto-cleared -------------------------

def _banned_login(monkeypatch, token="banned-tok-9"):
    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token",
        lambda email, password, settings:
        (token, True, "suspended until 16 September 2030"))


def test_set_token_banned_auto_login_names_the_ban_and_the_way_out(
        monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", password="pw",
                                       token="", provider="deepseek")],
               provider="deepseek", providers=[])
    _banned_login(monkeypatch)
    _answers(monkeypatch, ["t", "a@x.com", "", "y", "q"])

    tui.accounts_menu(s)

    out = capsys.readouterr().out
    assert "accounts unblock" in out
    assert out.index("Token updated") < out.index("accounts unblock")
    # The ban is persisted on the account, and this screen did not clear it.
    assert s.accounts[0].banned is True
    assert s.accounts[0].banned_until > 0.0
    assert s.accounts[0].token == "banned-tok-9"


def test_set_token_unbanned_auto_login_says_nothing_about_unblock(
        monkeypatch, capsys):
    """Control: the unblock line belongs only to a banned account."""
    s = _fresh(accounts=[AccountConfig(email="a@x.com", password="pw",
                                       token="", provider="deepseek")],
               provider="deepseek", providers=[])
    monkeypatch.setattr(
        "zourcewell.providers.deepseek.auth.obscura_login_token",
        lambda email, password, settings: ("tok-fresh", False, ""))
    _answers(monkeypatch, ["t", "a@x.com", "", "y", "q"])

    tui.accounts_menu(s)

    assert "accounts unblock" not in capsys.readouterr().out
    assert s.accounts[0].token == "tok-fresh"
    assert s.accounts[0].banned is False


# --- F29: combining marks take no cell -----------------------------------------

def test_combining_mark_takes_no_cell():
    assert tui_editor._char_cell("\u0301", 0) == 0
    # "e" + combining acute + "x": 2 cells, not the 3 code points.
    assert tui_editor._visual_pos("e\u0301x", 80) == (0, 2)
