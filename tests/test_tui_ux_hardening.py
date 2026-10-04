"""TUI interactive-UX hardening: honest saves, hidden secrets, 80x24 budgets.

Every test here drives the real menu functions with a scripted stdin, the way
a person types at a prompt, and asserts on what lands on the screen and on
disk -- not on internal calls.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import time
import types

import pytest

from functools import partial
from support import settings
from deepseaport import tui
from deepseaport.config import AccountConfig, Settings


# --- helpers ---------------------------------------------------------------------

# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.
_fresh = partial(settings, config_path="",
                  log_level="INFO", warmup_on_startup=True)


def _busy_port() -> tuple[int, socket.socket]:
    """An ephemeral port with a live listener on it, plus the holder socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock.getsockname()[1], sock


def _free_ephemeral() -> int:
    """An ephemeral port nothing is listening on."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _script(monkeypatch, answers):
    """Feed ``answers`` to every input() the menus ask for."""
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


def _as_console(monkeypatch, tty: bool = True):
    """Pretend stdin is (not) the real console, the way getpass branches."""
    monkeypatch.setattr(sys, "stdin",
                        types.SimpleNamespace(isatty=lambda: tty, readline=lambda: ""))


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# --- Task 1: declining the busy-port auto-fix -------------------------------------

def test_edit_port_declined_busy_port_is_neither_saved_nor_reported_saved(
        monkeypatch, capsys, tmp_path):
    """The user said "I'll handle it": the port the app just proved is taken
    must not be written to config.json, and the line must not claim success.
    """
    from deepseaport.config import load_settings

    port, holder = _busy_port()
    try:
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"port": port}), encoding="utf-8")
        s = load_settings(str(cfg))
        assert s.port == port
        _as_console(monkeypatch, tty=True)
        # port, then the auto-fix offer
        _script(monkeypatch, [str(port), "n"])
        tui._edit_port(s)

        assert s.port == port
        assert json.loads(cfg.read_text(encoding="utf-8"))["port"] == port
        out = capsys.readouterr().out
        assert "ok: Port" not in out
        assert "(saved)" not in out
        assert "in use" in out
    finally:
        holder.close()


def test_edit_port_declined_keeps_the_PREVIOUS_port_on_disk(monkeypatch, capsys, tmp_path):
    """The decline case that actually rewrote config.json: the old port was
    different from the (busy) one the user typed.
    """
    from deepseaport.config import load_settings

    port, holder = _busy_port()
    try:
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"port": 5001}), encoding="utf-8")
        s = load_settings(str(cfg))
        _as_console(monkeypatch, tty=True)
        _script(monkeypatch, [str(port), "n"])
        tui._edit_port(s)
        assert s.port == 5001
        assert json.loads(cfg.read_text(encoding="utf-8"))["port"] == 5001
        out = capsys.readouterr().out
        assert "ok: Port" not in out and "(saved)" not in out
    finally:
        holder.close()


def test_edit_port_free_candidate_still_saves_and_says_saved(monkeypatch, capsys):
    """The success path must keep working: a genuinely free port is saved."""
    port, holder = _busy_port()
    try:
        free = _free_ephemeral()
        s = _fresh(port=port)
        _script(monkeypatch, [str(free)])
        tui._edit_port(s)
        assert s.port == free
        out = capsys.readouterr().out
        assert f"ok: Port -> {free} (saved)" in out
    finally:
        holder.close()


def test_edit_port_accepted_auto_fix_still_saves_the_new_port(monkeypatch, capsys):
    port, holder = _busy_port()
    try:
        s = _fresh(port=port)
        _as_console(monkeypatch, tty=True)
        _script(monkeypatch, [str(port), "y"])
        tui._edit_port(s)
        assert s.port != port
        out = capsys.readouterr().out
        assert f"ok: Port -> {s.port} (saved)" in out
    finally:
        holder.close()


# --- Task 2: the token is not echoed while the password is hidden -----------------

def test_add_token_is_read_through_getpass_on_a_console(monkeypatch):
    """On a real console the token goes down the hidden path -- the password
    three fields earlier already did.
    """
    seen: list[str] = []

    def _gp(prompt=""):
        seen.append(prompt)
        return "tok-from-secret"

    monkeypatch.setattr("getpass.getpass", _gp)
    _as_console(monkeypatch, tty=True)
    s = _fresh()
    _script(monkeypatch, ["a", "new@x.com", "pw", "b"])
    tui.accounts_menu(s)
    assert [a.token for a in s.accounts] == ["tok-from-secret"]
    assert any("Token" in p for p in seen)


def test_add_token_y_still_triggers_auto_login_on_a_console(monkeypatch):
    """Hidden entry must not break the "y = auto-token" shortcut."""
    monkeypatch.setattr("deepseaport.providers.deepseek.auth.obscura_login_token",
                        lambda email, password, settings: "auto-tok-123")
    calls: list[str] = []

    def _gp(prompt=""):
        calls.append(prompt)
        if "Token" in prompt:
            return "y"
        return "pw"

    monkeypatch.setattr("getpass.getpass", _gp)
    _as_console(monkeypatch, tty=True)
    s = _fresh()
    _script(monkeypatch, ["a", "new@x.com", "b"])
    tui.accounts_menu(s)
    assert s.accounts[0].token == "auto-tok-123"


def test_set_token_is_read_through_getpass_on_a_console(monkeypatch):
    def _gp(prompt=""):
        return "tok-from-secret" if "token" in prompt.lower() else "pw"

    monkeypatch.setattr("getpass.getpass", _gp)
    _as_console(monkeypatch, tty=True)
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="old")])
    _script(monkeypatch, ["t", "a@x.com", "b"])
    tui.accounts_menu(s)
    assert s.accounts[0].token == "tok-from-secret"


def test_token_prompt_falls_back_to_visible_input_and_says_so(monkeypatch, capsys):
    """No console -> no echo control. Same announcement _ask_password makes;
    the value still arrives (that is what piped callers drive).
    """
    _as_console(monkeypatch, tty=False)
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="old")])
    _script(monkeypatch, ["t", "a@x.com", "tok-via-input", "b"])
    tui.accounts_menu(s)
    assert s.accounts[0].token == "tok-via-input"
    assert "echoed" in capsys.readouterr().out




# --- Task 3: Settings 19 and the main menu agree on the provider mode ------------

def test_settings_item_19_renders_the_multi_provider_list(monkeypatch, capsys):
    s = _fresh(provider="deepseek", providers=["deepseek", "qwen"])
    _script(monkeypatch, ["q"])
    tui.settings_menu(s)
    line = next(ln for ln in _lines(capsys) if ln.strip().startswith("19."))
    assert "deepseek + qwen (multi)" in line


def test_settings_19_and_main_menu_never_disagree(monkeypatch, capsys):
    s = _fresh(provider="deepseek", providers=["deepseek", "qwen"],
               accounts=[AccountConfig(email="a@x.com", token="t1")])
    _script(monkeypatch, ["q"])
    tui.settings_menu(s)
    settings_out = capsys.readouterr().out
    _script(monkeypatch, ["q"])
    tui.main_menu(s, lambda st: 0)
    main_out = capsys.readouterr().out

    disp = tui._provider_display(s)
    assert disp == "deepseek + qwen (multi)"
    assert f"19. Provider" in settings_out and disp in settings_out
    assert f"Provider  : {disp}" in main_out


def test_provider_display_single_mode_is_just_the_id():
    s = _fresh(provider="qwen", providers=[])
    assert tui._provider_display(s) == "qwen"


# --- Task 4: accountless providers are not sent to add accounts ------------------

def test_main_menu_setup_hint_waived_for_an_accountless_provider(monkeypatch, capsys):
    s = _fresh(provider="opencode", providers=["opencode"], accounts=[])
    assert tui._preflight_reason(s) == ""          # serve would work fine
    _script(monkeypatch, ["q"])
    tui.main_menu(s, lambda st: 0)
    out = capsys.readouterr().out
    assert "no accounts yet" not in out
    assert "needs no accounts" in out


def test_main_menu_setup_hint_still_shown_for_an_accounted_provider(monkeypatch, capsys):
    s = _fresh(provider="deepseek", providers=[], accounts=[])
    _script(monkeypatch, ["q"])
    tui.main_menu(s, lambda st: 0)
    out = capsys.readouterr().out
    assert "no accounts yet" in out and "Accounts (3)" in out


def test_accounts_screen_empty_hint_respects_accountless_provider(monkeypatch, capsys):
    s = _fresh(provider="opencode", providers=["opencode"], accounts=[])
    _script(monkeypatch, ["b"])
    tui.accounts_menu(s)
    out = capsys.readouterr().out
    assert "use 'a' to add one" not in out
    assert "needs none" in out


def test_accounts_screen_empty_hint_unchanged_for_accounted_provider(monkeypatch, capsys):
    s = _fresh(provider="deepseek", providers=[], accounts=[])
    _script(monkeypatch, ["b"])
    tui.accounts_menu(s)
    assert "use 'a' to add one" in capsys.readouterr().out


# --- Task 5: the table fits 80 columns, cooldowns read like a human -------------

@pytest.mark.parametrize("secs,expect", [
    (0, "0s"), (30, "30s"), (120, "120s"), (300, "300s"), (540, "540s"),
    (600, "10m"), (2700, "45m"), (7200, "2h"), (5400, "1.5h"), (86400, "1d"),
    (2592000, "30d"),
])
def test_fmt_duration_is_human_scale(secs, expect):
    assert tui._fmt_duration(secs) == expect


def test_fmt_duration_never_renders_raw_seconds_for_a_long_mute():
    assert len(tui._fmt_duration(2592000)) <= 6      # fits the 6-wide column


def _status_rows(rows):
    def _status(self):
        return [dict(r) for r in rows]
    return _status


def _draw_accounts(monkeypatch, capsys, s, rows, banned_ident=None):
    monkeypatch.setattr("deepseaport.accounts.AccountPool.status", _status_rows(rows))
    until = time.time() + 30 * 86400
    monkeypatch.setattr("deepseaport.accounts.get_cached_ban_labels",
                        lambda accts: ({banned_ident: until} if banned_ident else {}))
    _script(monkeypatch, ["b"])
    tui.accounts_menu(s)
    return _lines(capsys)


def _row(ident, **kw):
    base = {"identifier": ident, "email": ident, "busy": False,
            "cooldown_remaining": 0, "uses": 0, "banned": False,
            "banned_until": 0.0, "current": False}
    base.update(kw)
    return base


def test_account_rows_fit_an_eighty_column_console(monkeypatch, capsys):
    """The 84/98-column table: long address + 30-day mute + ban date."""
    ident = "a-very-long-account-address@example-domain.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1", banned=True,
                                       banned_until=time.time() + 30 * 86400)])
    rows = [_row(ident, cooldown_remaining=2592000, uses=12345, current=True)]
    out = _draw_accounts(monkeypatch, capsys, s, rows, banned_ident=ident)
    body = [ln for ln in out if re.match(r"^    1\s+\S", ln)]
    assert body, out
    for ln in body:
        assert len(ln) <= 80, (len(ln), ln)
    # The information is kept, just scaled: 30 days, not 2592000 seconds.
    assert "30d" in body[0]
    assert "2592000" not in body[0]
    assert "(BANNED:" in body[0]


def test_account_rows_keep_the_provider_tag_in_multi_mode(monkeypatch, capsys):
    a, b = "a@x.com", "b@x.com"
    s = _fresh(provider="deepseek", providers=["deepseek", "qwen"],
               accounts=[AccountConfig(email=a, token="t1", provider="deepseek"),
                         AccountConfig(email=b, token="t2", provider="qwen")])
    rows = [_row(a), _row(b)]
    out = _draw_accounts(monkeypatch, capsys, s, rows)
    assert any("[deepseek]" in ln for ln in out)
    assert any("[qwen]" in ln for ln in out)
    assert all(len(ln) <= 80 for ln in out if "@" in ln)


def test_ban_date_and_provider_tag_outrank_the_column_budget(monkeypatch, capsys):
    """Worst case: a 30-day ban date AND a multi-provider tag. The tail is
    the information, so the row may exceed 80 by a little -- but the tag and
    the date are both still readable, which is the trade the floor documents.
    """
    ident = "you@example.com"
    s = _fresh(provider="deepseek", providers=["deepseek", "qwen"],
               accounts=[AccountConfig(email=ident, token="t1", provider="deepseek")])
    rows = [_row(ident, cooldown_remaining=2592000, current=True)]
    out = _draw_accounts(monkeypatch, capsys, s, rows, banned_ident=ident)
    row = next(ln for ln in out if re.match(r"^    1\s+\S", ln))
    assert "(BANNED:" in row and "[deepseek]" in row
    assert ident in row          # not elided: the budget had 13 columns for it
    assert len(row) <= 80 + 4   # documented floor, not a silent overflow


def test_account_rows_drop_the_redundant_provider_tag_in_single_mode(monkeypatch, capsys):
    ident = "a@x.com"
    s = _fresh(provider="qwen", providers=[],
               accounts=[AccountConfig(email=ident, token="t1", provider="qwen")])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    row = next(ln for ln in out if ident in ln)
    assert "[qwen]" not in row          # "Provider scope: qwen" already says it
    assert any("Provider scope: qwen" in ln for ln in out)


def test_long_identifiers_stay_distinguishable_at_the_floor(monkeypatch, capsys):
    """Two addresses that differ only in the middle must not render the same.

    The ban date eats the budget down to the 16-column floor, and both
    addresses share their first 11 characters and their domain.
    """
    a = "firstname.lastname+tag1@some-quite-long-corporate-domain.example"
    b = "firstname.lastname+tag2@some-quite-long-corporate-domain.example"
    until = time.time() + 30 * 86400
    s = _fresh(accounts=[
        AccountConfig(email=a, token="t1", banned=True, banned_until=until),
        AccountConfig(email=b, token="t2", provider="deepseek")])
    rows = [_row(a, cooldown_remaining=2592000), _row(b)]
    out = _draw_accounts(monkeypatch, capsys, s, rows, banned_ident=a)
    body = [ln for ln in out if re.match(r"^    \d\s+\S", ln)]
    assert len(body) == 2
    shown = [re.match(r"^    \d\s+(\S+)", ln).group(1) for ln in body]
    assert shown[0] != shown[1]
    assert "tag1" in shown[0] and "tag2" in shown[1]
    # Paying for that with a wrapped row is the documented trade: a couple of
    # columns over beats two accounts the table cannot tell apart.
    assert max(len(ln) for ln in body) <= 82


def test_identifiers_that_fit_are_printed_verbatim(monkeypatch, capsys):
    ident = "a@x.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1")])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    assert any(ident in ln for ln in out)
    assert "..." not in "\n".join(out)


# --- Task 5b: the menus fit a 24-row console -------------------------------------

def test_settings_screen_clears_and_compacts_on_a_short_console(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui, "_term_size", lambda: (80, 24))
    s = _fresh()
    _script(monkeypatch, ["q"])
    tui.settings_menu(s)
    out = _lines(capsys)
    assert len(out) <= 24, len(out)
    assert "19. Provider" in "\n".join(out)
    assert not any("Request handling" in ln for ln in out)   # sections dropped
    # The prompt is written by input() at the cursor, i.e. the row right after
    # the last drawn line -- which must still be on screen.
    assert [ln for ln in out if ln.strip()][-1] == "  20. Back (q)"


def test_settings_screen_keeps_its_sections_when_piped(monkeypatch, capsys):
    """Piped output (tests, logs) keeps the grouped screen it is read from."""
    s = _fresh()
    _script(monkeypatch, ["q"])
    tui.settings_menu(s)
    out = "\n".join(_lines(capsys))
    assert "Request handling" in out and "Provider" in out


def test_settings_screen_redraws_the_reject_reason_after_clearing(monkeypatch, capsys):
    """The clear must not eat the explanation the user needs."""
    s = _fresh()
    _script(monkeypatch, ["99", "q"])
    tui.settings_menu(s)
    out = capsys.readouterr().out
    assert out.count("Unknown option: type 1-19") == 1
    # ... and it belongs to the redrawn screen, not the one that was cleared.
    assert out.index("Unknown option") > out.rindex("=== Settings")


def test_accounts_screen_redraws_the_reject_reason_after_clearing(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    _script(monkeypatch, ["zzz", "b"])
    tui.accounts_menu(s)
    out = capsys.readouterr().out
    assert out.count("Unknown option. Number selects, d deletes.") == 1
    assert out.index("Unknown option") > out.rindex("=== Accounts")


# --- Task 6: main_menu guards its submenus; expanduser cannot kill the TUI ------

def test_main_menu_survives_a_settings_menu_crash(monkeypatch, capsys):
    s = _fresh()
    monkeypatch.setattr(tui, "settings_menu",
                        lambda st: (_ for _ in ()).throw(RuntimeError("boom")))
    _script(monkeypatch, ["2", "q"])
    assert tui.main_menu(s, lambda st: 0) == 0
    out = capsys.readouterr().out
    assert "Settings failed" in out and "boom" in out


def test_main_menu_survives_an_accounts_menu_crash(monkeypatch, capsys):
    s = _fresh()
    monkeypatch.setattr(tui, "accounts_menu",
                        lambda st: (_ for _ in ()).throw(RuntimeError("kaboom")))
    _script(monkeypatch, ["3", "q"])
    assert tui.main_menu(s, lambda st: 0) == 0
    assert "Accounts failed" in capsys.readouterr().out


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError, SystemExit])
def test_main_menu_does_not_swallow_control_flow_from_a_submenu(monkeypatch, exc):
    s = _fresh()
    monkeypatch.setattr(tui, "settings_menu",
                        lambda st: (_ for _ in ()).throw(exc("stop")))
    _script(monkeypatch, ["2", "q"])
    with pytest.raises(exc):
        tui.main_menu(s, lambda st: 0)


def test_edit_obscura_bin_survives_an_unexpandable_home(monkeypatch, capsys, tmp_path):
    """No USERPROFILE and no HOMEDRIVE/HOMEPATH: expanduser() raises
    RuntimeError, which used to end the process with a traceback.
    """
    import pathlib

    exe = tmp_path / "obscura.exe"
    exe.write_text("MZ", encoding="utf-8")
    monkeypatch.setattr(pathlib.Path, "expanduser",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("Could not determine home directory.")))
    s = _fresh()
    _script(monkeypatch, [str(exe)])
    tui._edit_obscura_bin(s)
    out = capsys.readouterr().out
    assert "Could not expand" in out
    assert s.obscura_bin == str(exe)
    assert "ok: Obscura binary" in out


# --- Task 7: a typo never reports success ----------------------------------------

def test_set_token_typo_reports_not_found_and_no_success(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    _script(monkeypatch, ["t", "typo@x.com", "sometok", "b"])
    tui.accounts_menu(s)
    out = capsys.readouterr().out
    assert "Not found: typo@x.com" in out
    assert "Token updated" not in out
    assert [a.token for a in s.accounts] == ["t1"]


def test_delete_typo_reports_not_found_and_keeps_the_account(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    _script(monkeypatch, ["d", "typo@x.com", "y", "b"])
    tui.accounts_menu(s)
    out = capsys.readouterr().out
    assert "Not found" in out and "nothing deleted" in out
    assert "Removed" not in out
    assert [a.email for a in s.accounts] == ["a@x.com"]


def test_set_token_row_number_out_of_range_changes_nothing(monkeypatch, capsys):
    s = _fresh(accounts=[AccountConfig(email="a@x.com", token="t1")])
    _script(monkeypatch, ["t", "9", "sometok", "b"])
    tui.accounts_menu(s)
    assert "Out of range" in capsys.readouterr().out
    assert [a.token for a in s.accounts] == ["t1"]


# --- Task 8a: token_expires_at is visible ----------------------------------------

def test_expired_token_no_longer_renders_as_plain_yes(monkeypatch, capsys):
    ident = "a@x.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1",
                                       token_expires_at=time.time() - 60)])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    row = next(ln for ln in out if ident in ln)
    assert "expired" in row
    assert "yes" not in row


def test_far_future_expiry_renders_as_plain_yes(monkeypatch, capsys):
    ident = "a@x.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1")])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    row = next(ln for ln in out if ident in ln)
    assert "yes" in row and "expired" not in row


def test_a_token_valid_for_a_year_gets_no_day_hint(monkeypatch, capsys):
    """The Token cell is 7 wide: a hint that overflows it would push the row
    out of the column budget again.
    """
    ident = "a@x.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1",
                                       token_expires_at=time.time() + 400 * 86400)])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    row = next(ln for ln in out if ident in ln)
    assert re.search(r"\byes\b", row)
    assert "400d" not in row
    assert len(row) <= 80


def test_near_expiry_renders_a_day_hint_that_fits(monkeypatch, capsys):
    ident = "a@x.com"
    s = _fresh(accounts=[AccountConfig(email=ident, token="t1",
                                       token_expires_at=time.time() + 3 * 86400)])
    out = _draw_accounts(monkeypatch, capsys, s, [_row(ident)])
    row = next(ln for ln in out if ident in ln)
    assert "yes 3d" in row
    assert len(row) <= 80


# --- Task 8b: the four unreachable settings are reported, not silently added ----

def test_settings_menu_numbering_contract_is_unchanged(monkeypatch, capsys):
    """Items 1-19 plus "20. Back" are API (tests drive them by number), so the
    four unreachable settings (obscura_profile / browser_bin / browser_headless
    / tool_args_max_chars) are a written recommendation, not a renumbering.
    """
    s = _fresh()
    _script(monkeypatch, ["q"])
    tui.settings_menu(s)
    out = capsys.readouterr().out
    numbers = re.findall(r"^\s+(\d+)\.\s", out, re.M)
    assert numbers == [str(i) for i in range(1, 21)]
    assert "20. Back" in out
    for field in ("obscura_profile", "browser_bin", "browser_headless",
                  "tool_args_max_chars"):
        assert field not in out      # still absent -> reported, not added
