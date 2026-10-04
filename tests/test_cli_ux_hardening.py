"""Seven verified defects in the CLI/chat/dry-run front doors.

One test (or a tight group) per task, each with the pre-change failure named in
the docstring. Offline: no browser, no network, no real uvicorn socket, no
credentials. All uvicorn/thread seams are faked at the same seams the existing
tests use (tests/test_cli_gaps.py, tests/test_chatui_gaps.py,
tests/test_auth_hardening.py).

* **T1** the Esc/q stop-key watcher outlived a failed bind (uvicorn's
  ``sys.exit(3)`` never sets ``should_exit``), kept draining CONIN$ every
  100 ms and silently ate every main-menu keystroke that was not Esc/q/Q.
* **T2** ``deepseaport login`` handed an empty password to the real browser
  (the emptiness check in providers/deepseek/auth.py runs *after* the
  real-browser call) and, with a piped secret and no ``--email``, read the
  PASSWORD as the email.
* **T3** only 2 of 4 serve paths printed the non-loopback / auth-disabled
  notice, and the chat paths printed it at a log level they had just silenced.
* **T4** the chat REPL looped on ``should_exit`` alone, so a dead server
  thread produced an endless ``Connection refused`` prompt.
* **T5** ``accounts unblock <typo>`` printed "cleared for 0 account(s)" and
  exited 0 while the account stayed banned.
* **T6** ``serve``/``waf``/``models``/``login``/``chat`` had no subcommand
  help, so ``--help`` rendered them as a bare name list.
* **T7** ``_is_port_free`` set SO_REUSEADDR on Windows, where that makes the
  check blind to any other server that sets it (node, ``python -m
  http.server``, waitress) - and asyncio, which is what uvicorn binds with,
  does not set it there.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import threading
import time
import types
from collections import deque

import pytest

import deepseaport.cli as cli_mod
import deepseaport.server_runner as server_runner
from support import (
    AliveThread,
    DeadThreadNoStart,
    FakeUvicornConfig,
    FakeUvicornServer,
    JoinCountingThread,
    on_disk,
    settings,
    write_config,
)
from deepseaport import chat_repl, chat_ui, dry_run as dry_run_mod
from deepseaport.config import AccountConfig, Settings

# --------------------------------------------------------------------------- #
# shared fakes                                                                   #
# --------------------------------------------------------------------------- #


class _PipedStdin(io.StringIO):
    """A non-tty stdin (``< password.txt``); isatty() False is the point."""

    def isatty(self) -> bool:
        return False


class _FakeMsvcrt:
    """Console input stand-in that records every key the watcher takes.

    ``keys`` is the initial supply; an empty queue means "nothing typed", which
    is what the watcher must treat as a reason to go back to waiting (not to
    spin). ``feed`` is the user typing in the main menu after the serve call
    returned - anything recorded from there on is a stolen keystroke.
    """

    def __init__(self, keys: bytes = b""):
        self.queue = deque(keys)
        self.consumed: list[bytes] = []
        self.polls = 0

    def feed(self, keys: bytes) -> None:
        self.queue.extend(keys)

    def kbhit(self) -> bool:
        self.polls += 1
        return bool(self.queue)

    def getch(self) -> bytes:
        self.consumed.append(self.queue.popleft())
        return self.consumed[-1]


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.


def _settings(tmp_path, **kw) -> Settings:
    kw.setdefault("config_path", str(tmp_path / "config.json"))
    kw.setdefault("log_level", "INFO")
    kw.setdefault("warmup_on_startup", True)
    return settings(**kw)


def _acc(email: str, token: str = "tok") -> AccountConfig:
    return AccountConfig(email=email, token=token)


# --------------------------------------------------------------------------- #
# T1: the stop-key watcher must die with the serve call                           #
# --------------------------------------------------------------------------- #


def _record_threads(monkeypatch) -> list:
    """Real threads that remember themselves (so the test can join them).

    The target is captured at construction time on purpose: CPython 3.13's
    ``Thread.run`` does ``del self._target`` when the thread finishes, so
    reading it afterwards only works while the thread is still alive.
    """
    made: list = []

    class _Recording(threading.Thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, k.get("target", a[0] if a else None)))

    monkeypatch.setattr(threading, "Thread", _Recording)
    return made


def _watchers(made) -> list:
    return [t for t, target in made if target is server_runner._watch_stop_keys]


def _serve_with_broken_bind(monkeypatch, keys: bytes = b""):
    """Install the fakes for a _run_server whose serve call comes back early.

    A uvicorn Server whose run() is patched to a plain return (or a SystemExit,
    which is what uvicorn's startup() raises when the bind fails) reproduces
    the "no should_exit, control is back in the menu" shape. A fake msvcrt is
    installed too, so the watcher takes its real Windows branch and every key
    it takes is recorded.

    Returns (recorded threads, fake console).
    """
    import uvicorn

    made = _record_threads(monkeypatch)
    msvcrt = _FakeMsvcrt(keys)
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr("deepseaport.server.create_app", lambda settings=None: object())
    monkeypatch.setattr("deepseaport.server.apply_log_level", lambda *a, **k: None)
    return made, msvcrt


@pytest.mark.parametrize("mode", ["return", "sys_exit"])
def test_stop_key_watcher_dies_when_the_serve_call_ends(monkeypatch, tmp_path, mode):
    """T1. Pre-change: after a failed bind the daemon watcher was still looping
    and draining CONIN$, so every main-menu keystroke that was not Esc/q/Q was
    discarded - the menu looked frozen. ``return`` and ``sys_exit`` are the two
    ways the serve call can come back without ever setting should_exit.
    """
    def _run(self):
        if mode == "sys_exit":
            raise SystemExit(3)  # uvicorn's startup() on a failed bind

    made, msvcrt = _serve_with_broken_bind(monkeypatch)
    monkeypatch.setattr(FakeUvicornServer, "run", _run)

    settings = _settings(tmp_path, port=5001, log_level="INFO")
    args = argparse.Namespace(host="127.0.0.1", port=0, workers=1)
    if mode == "sys_exit":
        with pytest.raises(SystemExit):
            server_runner._run_server(settings, args, ensure_free=lambda s, h, p: p)
    else:
        assert server_runner._run_server(
            settings, args, ensure_free=lambda s, h, p: p) == 0

    watchers = _watchers(made)
    assert len(watchers) == 1, "the stop-key watcher was never started"
    watcher = watchers[0]
    # Dead the moment the call returns, not "within one 100ms poll": a
    # keystroke arriving in that window is a menu keystroke.
    assert not watcher.is_alive(), (
        "stop-key watcher outlived the serve call: it keeps eating "
        "main-menu keystrokes after control is back in the menu")
    watcher.join(timeout=5)
    # And now the user types in the menu: nothing may be taken. Deterministic
    # because the thread is joined, so no further getch can happen at all.
    msvcrt.feed(b"251")
    assert msvcrt.consumed == []


def test_stopped_watcher_consumes_no_keys(monkeypatch):
    """T1, the keystroke half. With the stop already signalled the watcher
    returns without reading a single byte - the 100 ms poll used to be the
    thing that swallowed the next menu keypress."""
    msvcrt = _FakeMsvcrt(b"251")
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    stop = threading.Event()
    stop.set()

    server_runner._watch_stop_keys(types.SimpleNamespace(should_exit=False), stop)

    assert msvcrt.consumed == []
    assert msvcrt.polls == 0  # not even a probe: it never entered the loop


def test_dry_run_watcher_dies_with_the_serve_call(monkeypatch, tmp_path):
    """T1, second spawn site: run_dry_run started the same watcher with no way
    to stop it, so a dry run that failed to bind left the menu keyless."""
    import uvicorn

    made = _record_threads(monkeypatch)
    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda s, h, p: p)
    settings = _settings(tmp_path, port=5001, keys=[])

    assert dry_run_mod.run_dry_run(
        settings, argparse.Namespace(host="127.0.0.1", port=0)) == 0

    watchers = _watchers(made)
    assert len(watchers) == 1
    assert not watchers[0].is_alive()
    watchers[0].join(timeout=5)


def test_watch_stop_keys_survives_a_raising_should_exit(monkeypatch):
    """T1, guard on the refactor: a server object whose attribute access raises
    must still return quietly (tests/test_cli_gaps.py pins the same contract)."""
    monkeypatch.setitem(sys.modules, "msvcrt", _FakeMsvcrt(b""))

    class _Boom:
        @property
        def should_exit(self):
            raise RuntimeError("boom")

    server_runner._watch_stop_keys(_Boom())  # must not raise


# --------------------------------------------------------------------------- #
# T2: login must never open a browser with a missing credential                   #
# --------------------------------------------------------------------------- #


def _browser_spy(monkeypatch) -> list:
    """Record every call that would reach a real browser or login driver.

    Patched at the *implementation* modules, not at
    ``cli_mod._obscura_login_token``, so the shipped driver stays canonical and
    cmd_login's guard is the thing under test.
    """
    calls: list = []

    def _impl(email, password, settings):
        calls.append(("driver", email, password))
        return (None, False, "")

    def _real(email, password, settings):
        calls.append(("REAL_BROWSER", email, password))
        return (None, False, "")

    monkeypatch.setattr("deepseaport.providers.deepseek.auth.obscura_login_token", _impl)
    monkeypatch.setattr(
        "deepseaport.providers.deepseek.real_browser.login_with_real_browser", _real)
    return calls


def test_login_without_a_password_never_opens_the_browser(monkeypatch, tmp_path,
                                                           capsys):
    """T2. Pre-change: Enter at the password prompt (no "Enter=skip" hint
    there) opened a real Helium/Chrome window, filled the email with an empty
    password and clicked Log in - a failed sign-in from the user's own IP,
    which is what trips risk control / captcha. The TUI already refused this."""
    calls = _browser_spy(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _PipedStdin(""))  # non-tty, immediate EOF
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="", provider=None))

    assert rc != 0
    assert calls == [], f"a login driver was invoked with {calls!r}"
    out = capsys.readouterr().out
    assert "password" in out
    assert "password.txt" in out  # names the piped-secret contract
    assert on_disk(cfg)["accounts"] == []  # nothing persisted


def test_login_without_an_email_never_eats_the_piped_password(monkeypatch, tmp_path,
                                                              capsys):
    """T2, piped half. Pre-change: ``login < password.txt`` with no --email let
    input() read the PASSWORD as the email, then logged in as
    (email=<the password>, password="")."""
    calls = _browser_spy(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _PipedStdin("s3cret-from-the-pipe\n"))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="", password="", provider=None))

    assert rc != 0
    assert calls == [], f"a login driver was invoked with {calls!r}"
    out = capsys.readouterr().out
    assert "--email" in out
    assert "s3cret-from-the-pipe" not in out  # the secret is never echoed back
    assert on_disk(cfg)["accounts"] == []


def test_login_with_enter_at_the_password_prompt_is_cancelled(monkeypatch, tmp_path,
                                                              capsys):
    """T2, the exact live shape: a console, the user presses Enter at the
    no-echo password prompt (which carries no "Enter=skip" hint), and before
    the fix a real browser window opened and tried to sign in with an empty
    password."""
    calls = _browser_spy(monkeypatch)
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "a@x.com")
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "")  # Enter
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="", password="", provider=None))

    assert rc != 0
    assert calls == [], f"a login driver was invoked with {calls!r}"
    out = capsys.readouterr().out
    assert "no password" in out
    assert on_disk(cfg)["accounts"] == []


def test_login_with_both_credentials_still_reaches_the_driver(monkeypatch, tmp_path,
                                                             capsys):
    """T2, control: the guard must not break the normal interactive or piped
    login (the piped secret still reaches the driver, byte for byte)."""
    seen: dict = {}
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: seen.update(email=e, password=p)
                        or ("tok-from-login", False, ""))
    monkeypatch.setattr(sys, "stdin", _PipedStdin("s3cret\n"))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="", provider=None)) == 0
    assert seen == {"email": "a@x.com", "password": "s3cret"}
    assert on_disk(cfg)["accounts"][0]["password"] == "s3cret"


def test_injected_driver_still_sees_an_empty_piped_password(monkeypatch, tmp_path):
    """T2, the seam the guard must not close.

    tests/test_cli_config_hardening.py pins that a caller who injects its own
    driver reaches it with the empty piped credential (that is how a caller
    drives its own empty-credential handling, and the shipped driver is the
    only one that reaches a real browser). Re-asserted here so the design is
    pinned on this side too.
    """
    seen: dict = {}
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: seen.update(password=p) or ("", False, ""))
    monkeypatch.setattr(sys, "stdin", _PipedStdin(""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="", provider=None)) == 3
    assert seen == {"password": ""}
    assert on_disk(cfg)["accounts"] == []


# --------------------------------------------------------------------------- #
# T3: every serve path warns on a non-loopback bind, with auth disabled           #
# --------------------------------------------------------------------------- #


def _run_plain_serve(monkeypatch, settings) -> int:
    import uvicorn

    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr("deepseaport.server.create_app", lambda s=None: object())
    monkeypatch.setattr("deepseaport.server.apply_log_level", lambda *a, **k: None)
    return server_runner._run_server(
        settings, argparse.Namespace(host="", port=0, workers=1),
        ensure_free=lambda s, h, p: p)


def _run_chat(monkeypatch, settings, on_input=None) -> int:
    import uvicorn

    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr(threading, "Thread", AliveThread)
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda s, h, p: p)
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(chat_ui, "_post_completion", lambda *a, **k: "hi")
    monkeypatch.setattr("builtins.input",
                        on_input or (lambda *a, **k: "/quit"))
    return chat_ui.run_chat_server(settings)


def _run_multi_chat(monkeypatch, settings, count=2) -> int:
    import uvicorn

    usable = [_acc("a@x.com", "t1"), _acc("b@x.com", "t2")]
    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr(threading, "Thread", AliveThread)
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr(chat_ui, "_select_usable_accounts_live",
                        lambda st: (usable, 0, 0))
    base = int(settings.port)
    monkeypatch.setattr(chat_ui, "_allocate_sequential_ports",
                        lambda h, b, n, is_free=None: [base + i for i in range(n)])
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(chat_ui, "_post_completion", lambda *a, **k: "hi")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "/quit")
    return chat_ui.run_multi_chat_servers(settings, count=count)


def _run_dry(monkeypatch, settings) -> int:
    import uvicorn

    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda s, h, p: p)
    monkeypatch.setattr(server_runner, "_watch_stop_keys", lambda *a, **k: None)
    return dry_run_mod.run_dry_run(
        settings, argparse.Namespace(host="", port=0))


def _lan_settings(tmp_path) -> Settings:
    return _settings(tmp_path, keys=[], listen=True, port=5100,
                     accounts=[_acc("a@x.com", "t1")])


@pytest.mark.parametrize("path", ["serve", "chat", "multi", "dry"])
def test_every_serve_path_warns_on_a_lan_bind_without_keys(monkeypatch, tmp_path,
                                                            capsys, path):
    """T3. Pre-change only plain serve printed the notice; dry run and both chat
    modes stayed quiet - and dry run is the mode that echoes the ENTIRE raw
    request body and formatted prompt back to the caller."""
    settings = _lan_settings(tmp_path)
    runner = {"serve": lambda: _run_plain_serve(monkeypatch, settings),
              "chat": lambda: _run_chat(monkeypatch, settings),
              "multi": lambda: _run_multi_chat(monkeypatch, settings),
              "dry": lambda: _run_dry(monkeypatch, settings)}[path]
    assert runner() == 0
    out = capsys.readouterr().out
    assert "WARNING: serving on 0.0.0.0:5100" in out, f"{path} did not warn"
    assert "AUTH DISABLED" in out, f"{path} did not say auth is disabled"


@pytest.mark.parametrize("path", ["serve", "chat", "multi", "dry"])
def test_no_lan_warning_on_a_loopback_bind(monkeypatch, tmp_path, capsys, path):
    """T3, control: the notice is for non-loopback binds only, so the warning
    cannot rot into noise that every local user learns to skip."""
    settings = _lan_settings(tmp_path)
    settings.listen = False
    runner = {"serve": lambda: _run_plain_serve(monkeypatch, settings),
              "chat": lambda: _run_chat(monkeypatch, settings),
              "multi": lambda: _run_multi_chat(monkeypatch, settings),
              "dry": lambda: _run_dry(monkeypatch, settings)}[path]
    assert runner() == 0
    out = capsys.readouterr().out
    assert "WARNING: serving on" not in out
    assert "non-loopback" not in out


def test_chat_still_silences_request_logs_while_printing_the_notice(monkeypatch,
                                                                   tmp_path, capsys):
    """T3, the constraint. The chat modes must keep the chat window clean
    (ARCHITECTURE.md "Chat log silence"), so the notice is printed and the
    log-level raise is untouched: no per-request record, not even at the
    server's own ERROR level."""
    root_level = logging.getLogger().level
    access = logging.getLogger("uvicorn.access")
    access_level = access.level
    seen: dict = {}

    def _input(*a, **k):
        # Measured while the REPL is up and the server is serving.
        seen["access"] = access.level
        seen["info_enabled"] = access.isEnabledFor(logging.INFO)
        seen["root"] = logging.getLogger().level
        return "/quit"

    try:
        assert _run_chat(monkeypatch, _lan_settings(tmp_path), on_input=_input) == 0
        assert seen["access"] == logging.ERROR
        assert seen["info_enabled"] is False
        assert seen["root"] == logging.ERROR
    finally:
        access.setLevel(access_level)
        logging.getLogger().setLevel(root_level)

    out = capsys.readouterr().out
    assert "WARNING: serving on" in out
    # And the levels are still restored on the way out (test_chatui_gaps pins
    # the same; the notice must not add an exit path that skips _cleanup).
    assert access.level == access_level


def test_multi_chat_warns_for_the_instance_port_range(monkeypatch, tmp_path, capsys):
    """T3, multi-chat: N instances are N adjacent ports on one host. The
    warning has to name the range (not just the first port) and be printed
    once, so an 8-instance run does not scroll the REPL help away with eight
    copies of a 400-character line."""
    settings = _lan_settings(tmp_path)
    assert _run_multi_chat(monkeypatch, settings, count=2) == 0
    out = capsys.readouterr().out
    assert "[2 instances: 0.0.0.0:5100-5101]" in out
    assert "WARNING: serving on" in out
    assert "AUTH DISABLED" in out
    assert out.count("WARNING: serving on") == 1


def test_single_multi_chat_instance_names_its_own_port(monkeypatch, tmp_path, capsys):
    """T3, multi-chat, single instance: the one-instance wording keeps the
    port visible without a range nobody needs."""
    settings = _lan_settings(tmp_path)
    assert _run_multi_chat(monkeypatch, settings, count=1) == 0
    out = capsys.readouterr().out
    assert "[1/1 instance] WARNING: serving on 0.0.0.0:5100" in out


# --------------------------------------------------------------------------- #
# T4: the REPL must notice a dead server, and only a dead one                      #
# --------------------------------------------------------------------------- #


def _repl(monkeypatch, seq, post, alive, stop_note):
    """Drive run_repl with a scripted keyboard; `alive` is a dict or a bool.

    A dict lets a test kill the server mid-session (the real sequence); a bool
    is a fixed liveness. Raises out of the REPL if it asks for a keystroke the
    script does not have, which is how "kept prompting" is caught.
    """
    prompts: list[str] = []
    answers = iter(seq)
    up = (lambda: alive["alive"]) if isinstance(alive, dict) else (lambda: alive)

    def _input(prompt=""):
        prompts.append(prompt)
        try:
            return next(answers)
        except StopIteration:
            raise AssertionError(
                f"the REPL asked for another keystroke after {list(seq)!r}")

    monkeypatch.setattr("builtins.input", _input)
    chat_repl.run_repl(
        model="m", models=["m"], api_key="", history=[], settings=Settings(),
        post=post, can_continue=up, print_help=lambda: None,
        next_target=lambda: ("http://127.0.0.1:1", ""),
        stop_note=stop_note,
    )
    return prompts


def test_repl_ends_with_a_clear_note_when_the_server_thread_is_gone(monkeypatch,
                                                                   capsys):
    """T4. Pre-change: a uvicorn thread that died never set should_exit, so the
    REPL kept prompting and every message came back
    "request failed: <urlopen error [WinError 10061] Connection refused>" -
    a client-side error for a server-side death, forever.

    The server is alive for the first keystroke (so the REPL is genuinely in
    the loop) and gone by the time the request comes back, which is the real
    sequence: the thread dies while the user is typing."""
    state = {"alive": True}

    def _post(*a, **k):
        state["alive"] = False  # the uvicorn thread died
        raise RuntimeError("request failed: <urlopen error [WinError 10061] "
                           "Connection refused>")

    prompts = _repl(monkeypatch, ["hello"], _post, alive=state,
                    stop_note=lambda: "chat server is no longer running")
    out = capsys.readouterr().out

    assert len(prompts) == 1, "the REPL kept prompting a dead server"
    assert "no longer running" in out
    assert "Connection refused" in out  # the error is still shown, not swallowed


def test_repl_does_not_blame_the_server_for_one_transient_failure(monkeypatch,
                                                                  capsys):
    """T4, the other half. A single failed request on a live server is an
    ordinary request error; calling it a dead server would train the user to
    ignore the one message that matters."""
    calls = {"n": 0}

    def _post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("request failed: <urlopen error [WinError 10061] "
                               "Connection refused>")
        return "reply"

    prompts = _repl(monkeypatch, ["hi", "again", "/quit"], _post, alive=True,
                    stop_note=lambda: "chat server is no longer running")
    out = capsys.readouterr().out

    assert calls["n"] == 2
    assert len(prompts) == 3
    assert "reply" in out
    assert "no longer running" not in out


def test_repl_quit_is_not_reported_as_a_dead_server(monkeypatch, capsys):
    """T4, /quit still means /quit: a user who asked to leave is not told the
    server died."""
    prompts = _repl(monkeypatch, ["/quit"],
                    lambda *a, **k: pytest.fail("no request on /quit"),
                    alive=True,
                    stop_note=lambda: "chat server is no longer running")
    out = capsys.readouterr().out
    assert prompts == ["You: "]
    assert "no longer running" not in out


def test_chat_server_reports_a_dead_background_server(monkeypatch, tmp_path, capsys):
    """T4, end to end: the single-server mode's liveness seam is wired to the
    background thread, so a dead thread ends the REPL with the reason."""
    import uvicorn

    monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    monkeypatch.setattr(threading, "Thread", DeadThreadNoStart)
    monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
    monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda s, h, p: p)
    monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)

    def _input(*a, **k):
        raise AssertionError("a dead server must not invite another message")

    monkeypatch.setattr("builtins.input", _input)
    settings = _settings(tmp_path, keys=[], port=5001)

    assert chat_ui.run_chat_server(settings) == 0
    out = capsys.readouterr().out
    assert "no longer running" in out
    assert "/quit" in out


# --------------------------------------------------------------------------- #
# T5: `accounts unblock <typo>` must not report success                           #
# --------------------------------------------------------------------------- #


def _banned_cfg(tmp_path) -> str:
    # banned_until must be in the future: config.AccountConfig clears a ban
    # whose expiry has already passed at load time (config.py:127).
    until = time.time() + 3600
    return write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "a@x.com", "token": "t1", "banned": True, "banned_until": until},
        {"email": "b@x.com", "token": "t2", "banned": False, "banned_until": 0.0},
    ]})


def test_unblock_unknown_identifier_exits_non_zero(tmp_path, capsys):
    """T5. Pre-change: exit 0 and "cleared for 0 account(s)", so a typo read
    as "done", nothing was restarted and the account stayed banned."""
    cfg = _banned_cfg(tmp_path)

    rc = cli_mod.cmd_accounts(argparse.Namespace(
        config=cfg, accounts_action="unblock", identifier="typo@x.com"))

    assert rc == 1
    out = capsys.readouterr().out
    assert "not found: typo@x.com" in out
    assert "cleared" not in out
    # Nothing was touched: the still-banned account is still banned.
    assert on_disk(cfg)["accounts"][0]["banned"] is True


def test_unblock_without_an_identifier_still_clears_everything(tmp_path, capsys):
    """T5, control: bare `unblock` means "all accounts", which legitimately
    reports 0 and must keep working (never a false negative)."""
    cfg = _banned_cfg(tmp_path)

    rc = cli_mod.cmd_accounts(argparse.Namespace(
        config=cfg, accounts_action="unblock", identifier=""))

    assert rc == 0
    assert "not found" not in capsys.readouterr().out
    assert all(a["banned"] is False for a in on_disk(cfg)["accounts"])


def test_unblock_a_banned_account_still_clears_it(tmp_path, capsys):
    """T5, control: the real thing still works, and reports the count."""
    cfg = _banned_cfg(tmp_path)

    rc = cli_mod.cmd_accounts(argparse.Namespace(
        config=cfg, accounts_action="unblock", identifier="a@x.com"))

    assert rc == 0
    assert "cleared for 1 account(s)" in capsys.readouterr().out
    accs = {a["email"]: a for a in on_disk(cfg)["accounts"]}
    assert accs["a@x.com"]["banned"] is False
    assert accs["a@x.com"]["banned_until"] == 0.0


# --------------------------------------------------------------------------- #
# T7: the chat shutdown must stay single-shot after the notice/liveness changes #
# --------------------------------------------------------------------------- #


class _SrvExitCounted(FakeUvicornServer):
    """Fake uvicorn server that records each should_exit=True assignment."""

    exits: list = []

    def __setattr__(self, name, val):
        if name == "should_exit" and val:
            type(self).exits.append(val)
        super().__setattr__(name, val)


def test_chat_cleanup_is_single_shot_when_the_repl_is_interrupted(monkeypatch,
                                                                   tmp_path):
    """T7. Ctrl+C inside the REPL is a BaseException, the case the once-guard
    exists for; the Task 3 notice and the Task 4 stop_note must not add an exit
    path that skips it. Each server is signalled and joined exactly once and
    every log level is restored, or the TUI session stays at ERROR."""
    import uvicorn

    threads: list = []

    def _remember(*a, **k):
        t = JoinCountingThread(*a, **k)
        threads.append(t)
        return t

    _SrvExitCounted.exits = []
    root = logging.getLogger()
    old_root, old_access = root.level, logging.getLogger("uvicorn.access").level
    try:
        monkeypatch.setattr(uvicorn, "Config", FakeUvicornConfig)
        monkeypatch.setattr(uvicorn, "Server", _SrvExitCounted)
        monkeypatch.setattr(threading, "Thread", _remember)
        monkeypatch.setattr("deepseaport.server.create_app", lambda st: object())
        monkeypatch.setattr("deepseaport.cli._ensure_free_port", lambda s, h, p: p)
        monkeypatch.setattr(chat_ui, "_wait_for_server", lambda *a, **k: True)

        def _boom(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(chat_repl, "run_repl", _boom)
        settings = _settings(tmp_path, keys=[], port=5001, log_level="INFO")

        with pytest.raises(KeyboardInterrupt):
            chat_ui.run_chat_server(settings)

        assert threads[0].joined == 1, "the server thread was joined more than once"
        assert _SrvExitCounted.exits == [True]
        assert root.level == old_root
        assert logging.getLogger("uvicorn.access").level == old_access
        assert settings.log_level == "INFO"
    finally:
        root.setLevel(old_root)
        logging.getLogger("uvicorn.access").setLevel(old_access)


# --------------------------------------------------------------------------- #
# T6: --help gaps                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", ["serve", "waf", "models", "accounts", "login",
                                  "chat"])
def test_every_subcommand_has_help(capsys, path):
    """T6. Pre-change four of the six rendered as a bare {serve,waf,models,...}
    name list, so `deepseaport --help` said nothing about what they do."""
    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # One line per subcommand: "name  description".
    line = next((ln for ln in out.splitlines() if ln.strip().startswith(path + " ")),
                None)
    assert line is not None, f"{path} has no help line in:\n{out}"
    assert line.split(None, 1)[1].strip(), f"{path} still has an empty help"


def test_set_token_requirement_is_discoverable(capsys):
    """T6: nothing in --help said set-token needs --token, although omitting it
    is a hard argparse error (and a tokenless account cannot serve)."""
    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["accounts", "set-token", "--help"])
    assert exc.value.code == 0
    help_out = capsys.readouterr().out
    assert "--token" in help_out
    assert "required" in help_out.lower()

    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["accounts", "set-token", "a@x.com"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--token" in err


# --------------------------------------------------------------------------- #
# T7: the pre-flight port check must not be optimistic about a live listener      #
# --------------------------------------------------------------------------- #


def test_is_port_free_sees_a_reuseaddr_listener(tmp_path):
    """T7. SO_REUSEADDR on Windows lets a bind steal an address that another
    socket already holds - but only when BOTH set it, which node,
    ``python -m http.server`` and waitress all do. Setting it in the pre-flight
    check reported "free" for a port uvicorn then failed to bind; asyncio
    (uvicorn's transport) does not set it on Windows, so not setting it here
    matches the bind that will actually happen."""
    import socket

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    holder.listen(1)
    try:
        assert server_runner._is_port_free("127.0.0.1", port) is False
    finally:
        holder.close()
    assert server_runner._is_port_free("127.0.0.1", port) is True


def test_is_port_free_survives_an_impossible_port():
    """T7, guard: an out-of-range port reaches socket.bind as OverflowError,
    which is not an OSError and would otherwise escape as a traceback out of
    the pre-flight check."""
    assert server_runner._is_port_free("127.0.0.1", 999999) is False
    assert server_runner._is_port_free("127.0.0.1", "not-a-port") is False
