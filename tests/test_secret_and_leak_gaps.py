"""Secret + identifier leak gaps: cancelled sessions, unbounded waits, ACLs.

Seven fixes, each pinned here by a test that fails on the pre-change code:

* K1 -- a client disconnect during ``create_session`` cancelled a future that
  had already started, so the session id it produced was never deleted.
* K2 -- ``slot_releaser``'s reacquire loop had no deadline and no cancel
  token was required, so a ``cancel_event=None`` caller could park forever.
* K3 -- the ban-failover CURRENT switch saved the SETTINGS FALLBACK, i.e. a
  Settings bound to the operator's real ``config.json``.
* K4 -- the managed-serve reaper signalled a pid read from a plain JSON file
  whose only gate was a probe against that same file.
* K5 -- ``config.json`` claims 0600 protection that is a no-op on Windows.
* K6 -- upstream error bodies were reflected verbatim into client 4xx/5xx
  bodies and into the server log.
* K7 -- the main serve path logged every request target, and an account
  identifier may be a full bearer token.
* K8 -- four paths the K6 fix missed: the non-200 completion log, the
  recovery-failure log (message AND traceback), the three producer tails that
  emit ``str(exc)`` without passing the terminal ladder, and the SSE error
  frame, which clamped nothing at all.

No network, no credentials: every upstream call is a monkeypatched stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time

import pytest
from fastapi import HTTPException

from support import collect_sse, settings, sse_text
from deepseaport.accounts import AccountPool, PooledAccount
from deepseaport.config import AccountConfig, Settings
from deepseaport.core import engine as engine_mod
from deepseaport.core import transport as transport_mod
from deepseaport.providers.deepseek import completion_service as CS
from deepseaport.providers.opencode import completion_service as OCS
from deepseaport.providers.opencode import managed_serve as MS
from deepseaport.providers.qwen import completion_service as QCS
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare


# A realistic account token: 64 opaque chars, exactly what the scrubber is
# supposed to catch and what accounts.py permits as an account IDENTIFIER.
LIVE_TOKEN = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"

# The OTHER credential shape, and the one that needed a second regex: Qwen's
# credential is a JWT whose dot-framed segments are each short enough to slip
# under a run-length redactor (see providers/qwen/AGENTS.md).
LIVE_JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiJ1LTEiLCJuYW1lIjoiYUB4LmNvbSJ9"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")

# Bounded waits for the Event handshakes in K1. NOT assertion bounds -- they
# exist so a regression FAILS instead of hanging the suite, and so a fake that
# never gets released still unwinds. They sit far above what is actually being
# waited for (the 50ms cancel poll in `server_concurrency._CANCEL_POLL_SECONDS`
# and an in-process delete), so a loaded `windows-latest` runner cannot turn a
# slow scheduler into a red build.
_PARK_TIMEOUT = 10.0
# Bounded window in which a second, unwanted delete_session would have to
# arrive. There is no positive signal for a call that must not happen, so this
# is a quiescence window rather than an Event handshake; same length as the
# `sleep(0.3)` it replaced, so the strength is unchanged -- it just fails with
# a message instead of sampling and hoping.
_QUIESCENCE_SECONDS = 0.3


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.


def _settings(**kw) -> Settings:
    kw.setdefault("accounts", [AccountConfig(email="a@x.com", token="tok")])
    return settings(**kw)


# --- K1: cancelled create_session must not leak the session -------------------


def test_cancel_during_create_session_deletes_the_session_exactly_once(monkeypatch):
    """Pre-change: ``fut.cancel()`` returns False for a running callable, so
    ``_session_and_challenge`` raised ``_StreamCancelled`` with no session id
    in hand, the ``finally`` in ``_run_completion_core`` had nothing to delete
    and the session created a moment later stayed alive upstream forever."""
    from deepseaport.server_concurrency import _StreamCancelled
    from deepseaport.providers.deepseek.completion_service import _session_and_challenge

    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    second_delete = threading.Event()
    deleted: list[str] = []
    deleted_once = threading.Event()

    def _slow_session(headers):
        # Park until the test releases us, instead of sleeping a fixed 0.3s.
        # "Long enough that the cancel lands while this is still running" was
        # a scheduling bet: the cancel only lands if this is still parked when
        # the 50ms poll in _await_cancelable next looks, and a 0.3s window on a
        # loaded runner is not a guarantee. Parking on an Event makes the
        # cancel and the unwind two things the test orders explicitly, and it
        # is also what makes the post-cancel `deleted_once` wait below
        # reachable at all -- the id only exists once this returns.
        started.set()
        try:
            release.wait(_PARK_TIMEOUT)
        finally:
            finished.set()
        return "sess-inflight"

    def _record_delete(headers, sid):
        deleted.append(sid)
        if len(deleted) == 1:
            deleted_once.set()
        else:
            second_delete.set()

    monkeypatch.setattr("deepseaport.providers.deepseek.client.create_session", _slow_session)
    monkeypatch.setattr("deepseaport.providers.deepseek.client.fetch_pow", lambda h: {"a": 1})
    monkeypatch.setattr("deepseaport.providers.deepseek.client.delete_session", _record_delete)

    cancel = threading.Event()

    def _flip():
        assert started.wait(_PARK_TIMEOUT), "create_session never started"
        cancel.set()

    threading.Thread(target=_flip, daemon=True).start()

    with pytest.raises(_StreamCancelled):
        _session_and_challenge({"h": "1"}, parallel=True, cancel_event=cancel)

    assert not second_delete.is_set(), "cancelled setup deleted before it ran"
    # Only now: `sess-inflight` does not exist until the parked callable
    # returns, and the cleanup callback fires off that return.
    release.set()
    assert finished.wait(_PARK_TIMEOUT), "create_session never returned"
    assert deleted_once.wait(_PARK_TIMEOUT), (
        "the session created by the in-flight create_session was never "
        "deleted: it lingers upstream with no id anyone kept")
    assert deleted == ["sess-inflight"]
    # Give any second cleanup a chance to run: a double delete is as wrong as
    # none (the id is reused by the next session upstream).
    #
    # This window is NOT redundant with deleted_once above -- that proves the
    # first delete happened, which says nothing about a second one -- so it
    # stays. What changes is its shape: there is no positive signal for a call
    # that must NOT happen, so this is an explicit bounded quiescence wait
    # that FAILS the instant a second delete lands, instead of a bare sleep
    # that samples at 0.3s and hopes. Same length as the sleep it replaces, so
    # the strength is identical.
    assert not second_delete.wait(_QUIESCENCE_SECONDS), (
        f"session deleted more than once: {deleted}")


def test_cancel_before_create_session_starts_deletes_nothing(monkeypatch):
    """The other half of the same fix: a pre-set cancel event still aborts
    immediately, and a create_session that never ran has no session to leak."""
    from deepseaport.server_concurrency import _StreamCancelled
    from deepseaport.providers.deepseek.completion_service import _session_and_challenge

    monkeypatch.setattr("deepseaport.providers.deepseek.client.create_session", lambda h: pytest.fail(
        "a pre-set cancel must not create a session"))
    monkeypatch.setattr("deepseaport.providers.deepseek.client.fetch_pow", lambda h: pytest.fail(
        "a pre-set cancel must not fetch a challenge"))
    monkeypatch.setattr("deepseaport.providers.deepseek.client.delete_session",
                        lambda h, sid: pytest.fail("nothing to delete"))

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(_StreamCancelled):
        _session_and_challenge({"h": "1"}, parallel=True, cancel_event=cancel)
    # No clock here. The three patched seams above are `pytest.fail`, so any
    # session create / challenge fetch / delete raises inside this call and
    # fails the test outright -- a strictly stronger statement than the
    # `monotonic() - t0 < 1.0` budget this used to carry, which a slow runner
    # could turn red while proving nothing. The bounded-wait style is kept for
    # the neighbouring tests, where the fakes return normally and a wait is
    # the only way to observe ordering.


# --- K2: slot_releaser must terminate without a cancel token -----------------


class _AlwaysContendedLock:
    """Account lock held by "another request" for the whole test."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lock.acquire()  # slot_releaser's entry release() must succeed
        self.acquire_calls = 0

    def release(self) -> None:
        self._lock.release()

    def acquire(self, timeout=None) -> bool:
        self.acquire_calls += 1
        return False

    def locked(self) -> bool:
        return False


def test_slot_releaser_gives_up_on_contended_lock_without_cancel_token(monkeypatch):
    """Pre-change: ``while True: lock.acquire(timeout=0.5)`` with no deadline
    and no cancel token. ``cancel_event=None`` is legal (``Provider.complete``'s
    documented default, and the direct-call path), so a caller parked there
    forever while a WAF refresh or browser login held the account."""
    # raising=False: the point of the test is the loop, not the knob's
    # existence, so on pre-change code it must fail on "never returned"
    # rather than on a missing attribute.
    monkeypatch.setattr(engine_mod, "SLOT_REACQUIRE_TIMEOUT", 0.2, raising=False)
    item = type("Item", (), {})()
    item.lock = _AlwaysContendedLock()
    finished = threading.Event()
    error: list[BaseException] = []

    def _run():
        try:
            with engine_mod.slot_releaser(item, None):  # documented default
                pass
        except BaseException as exc:  # noqa: BLE001 - reported below
            error.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # Thread + wait (not a bare call): a regression here must FAIL, not hang
    # the suite forever.
    assert finished.wait(8), (
        "slot_releaser never returned: the reacquire loop has no deadline "
        "when no cancel token is supplied")
    assert not error, f"give-up path raised: {error!r}"
    assert item.lock.acquire_calls >= 1, "the lock was never even polled"


def test_slot_releaser_still_reacquires_an_uncontended_lock():
    """The give-up must not become the normal path: a free account comes back."""
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="t"))
    item.lock.acquire()
    with engine_mod.slot_releaser(item) as released:
        assert released is True
        assert not item.lock.locked(), "slot was not released for the global wait"
    assert item.lock.locked(), "slot was not reacquired after the global wait"
    item.lock.release()


def test_slot_releaser_still_aborts_on_cancel(monkeypatch):
    """Unchanged: a set cancel event still wins over the reacquire wait."""
    from deepseaport.server_concurrency import _StreamCancelled

    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="t"))
    item.lock.acquire()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(_StreamCancelled):
        with engine_mod.slot_releaser(item, cancel):
            pass
    assert not item.lock.locked(), "cancel must leave the slot released"


# --- K3: the ban-failover CURRENT switch must not write the operator file ----


def _ban_403(identifier: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail=f"Account {identifier} banned. Due to violation of user "
               f"policies, your account has been suspended.")


def _two_account_settings(config_path: str) -> Settings:
    return Settings(
        keys=[],
        accounts=[AccountConfig(email="a@x.com", token="tok-a-1234567890"),
                  AccountConfig(email="b@x.com", token="tok-b-1234567890")],
        active_account="a@x.com",
        config_path=config_path,
        warmup_on_startup=False,
        max_retries=0,
        log_level="WARNING",
    )


def _install_ban_failover(monkeypatch, settings: Settings) -> AccountPool:
    """Failover from the banned a@x.com onto b@x.com; returns the pool."""
    pool = AccountPool(settings.accounts, current=settings.active_account)
    used: list[str] = []

    def _core(app_, item_, prep_, **kw):
        used.append(item_.cfg.email)
        if item_.cfg.email == "a@x.com":
            AccountPool.mark_bad(item_, seconds=86400)
            raise _ban_403("a@x.com")
        return {"content": "hi from b", "thinking": "", "usage_total": 5}

    monkeypatch.setattr(CS, "_run_completion_core", _core)
    return pool


def _failover_sync(app, pool, monkeypatch) -> None:
    from deepseaport.providers.deepseek.completion_service import _complete_with_failover_sync
    from deepseaport.server_routes import _prepare

    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]},
                    getattr(app.state, "settings", None))
    first = pool.acquire(timeout=1)
    assert first.cfg.email == "a@x.com"
    result = _complete_with_failover_sync(app, pool, prep, first)
    assert result["content"] == "hi from b"


def test_ban_failover_current_switch_does_not_write_the_operator_config(
        tmp_path, monkeypatch):
    """Pre-change: ``switch_current_after_ban_failover`` saved whatever
    ``get_settings`` returned. On the default ``_app_settings`` that is
    ``load_settings()`` -- a Settings bound to the operator's real
    config.json -- so a settings-less app rewrote their file with a CURRENT
    switch they never made (the sibling ``ban_403_for_item`` was already
    guarded this session)."""
    operator_cfg = tmp_path / "operator.json"
    operator_cfg.write_text(json.dumps({
        "keys": ["sk-real"], "accounts": [{"email": "a@x.com", "token": "tok"}],
        "active_account": "a@x.com", "provider": "deepseek",
    }), encoding="utf-8")
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(operator_cfg))
    before = operator_cfg.read_text(encoding="utf-8")

    # An app that owns no settings: engine.app_settings() falls back to
    # load_settings(), i.e. the operator file above.
    embedded = create_app(_two_account_settings(""))
    del embedded.state.settings
    pool = _install_ban_failover(monkeypatch, _two_account_settings(""))

    _failover_sync(embedded, pool, monkeypatch)

    after = json.loads(operator_cfg.read_text(encoding="utf-8"))
    assert after["active_account"] == "a@x.com", (
        "the operator's config.json was rewritten by an app that owns no "
        f"settings (active_account={after['active_account']!r})")
    assert operator_cfg.read_text(encoding="utf-8") == before


def test_ban_failover_current_switch_persists_when_the_app_owns_settings(
        tmp_path, monkeypatch, caplog):
    """The guard must not become \"never persist\": the real serve app owns
    ``app.state.settings`` and the switched CURRENT still has to reach disk."""
    cfg = tmp_path / "config.json"
    settings = _two_account_settings(str(cfg))
    settings.save()
    app = create_app(settings)
    pool = _install_ban_failover(monkeypatch, settings)

    with caplog.at_level(logging.WARNING, logger="deepseaport.core.engine"):
        _failover_sync(app, pool, monkeypatch)

    assert pool.current == "b@x.com"
    assert settings.active_account == "b@x.com"
    persisted = json.loads(cfg.read_text(encoding="utf-8"))
    assert persisted["active_account"] == "b@x.com", \
        "the app-owned CURRENT switch stopped being persisted"
    assert "NOT persisted" not in caplog.text


# --- K4: the managed-serve reaper must identify its target -------------------


def _no_signals(monkeypatch) -> list:
    """Record every channel the reaper could use, as ``(channel, pid)``.

    Windows kills with ``taskkill /T`` and POSIX with ``os.kill``, so both are
    stubbed: an unstubbed ``_kill_tree`` would put a REAL taskkill against the
    fake pid 4242 on the Windows runner.
    """
    from deepseaport.providers.opencode import managed_serve as ms

    signalled: list[tuple] = []

    def _kill(pid, sig):
        signalled.append(("os.kill", pid))
        raise OSError("no such process")

    def _tree(pid):
        signalled.append(("taskkill", pid))
        return True

    monkeypatch.setattr(ms.os, "kill", _kill)
    monkeypatch.setattr(ms, "_kill_tree", _tree, raising=False)
    return signalled


def _claim_cmdline(monkeypatch, cmdline) -> None:
    """Pin the command line the reaper should see.

    ``raising=False`` on purpose: on pre-change code (no reader at all) the
    fake is simply never consulted, the pid gets signalled anyway, and the
    test fails on the signal assertion -- which is the regression.
    """
    monkeypatch.setattr(MS, "_process_command_line", lambda pid: cmdline,
                        raising=False)


def test_reaper_refuses_a_pid_that_is_not_opencode_serve(monkeypatch):
    """Pre-change: the only gate on the SIGTERM was a probe against a base URL
    stored in the SAME serve.json the pid came from, so any writer of that
    file could aim the signal at an arbitrary same-user process."""
    signalled = _no_signals(monkeypatch)
    _claim_cmdline(monkeypatch, r"C:\Windows\notepad.exe somefile.txt")

    MS._kill_pid(4242)

    assert signalled == [], \
        "a process that is not 'opencode serve' was signalled"


def test_reaper_signals_a_matching_opencode_serve(monkeypatch):
    """The other half: the real target must still be reaped."""
    signalled = _no_signals(monkeypatch)
    _claim_cmdline(
        monkeypatch,
        '"/c/Users/op/.bun/bin/opencode" serve --port 45999 '
        "--hostname 127.0.0.1")

    MS._kill_pid(4242)

    assert [pid for _channel, pid in signalled] == [4242], \
        "the stale managed serve was not signalled"


def test_reaper_skips_unreadable_command_line_and_never_raises(monkeypatch):
    signalled = _no_signals(monkeypatch)
    _claim_cmdline(monkeypatch, "")
    MS._kill_pid(4242)  # no /proc, no wmic, no PowerShell: must not guess
    assert signalled == [], "an unreadable command line must skip the signal"

    _claim_cmdline(monkeypatch, "")
    monkeypatch.setattr(MS, "_process_command_line",
                        lambda pid: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)
    MS._kill_pid(4242)  # must swallow, not surface into ensure()
    assert signalled == [], "a raising probe must not turn into a signal"


def test_reaper_never_signals_its_own_process_group(monkeypatch):
    """The self-protection guard, as a real assertion.

    ``_kill_pid`` opens with ``if not pid or pid == os.getpid(): return``, so
    our own pid and pid 0 (the POSIX "whole process group" signal) must be
    refused *before* the command-line probe even runs. The signal spy stays
    installed for the whole test, and a probe spy records whether the probe was
    consulted at all -- that is what makes the early return a guard rather
    than a coincidence of the cmdline check.

    This is the assertion the pre-change version of this test tried to make by
    calling ``monkeypatch.undo()`` and then firing at the REAL ``os.kill`` --
    which asserted nothing (``_kill_pid`` returns None) while aiming a live
    SIGTERM at the test runner. The guard is a property of ``_kill_pid``, so it
    is tested through the seam, never against the real process.
    """
    signalled = _no_signals(monkeypatch)
    seen: list[int] = []
    matching = ('"/c/Users/op/.bun/bin/opencode" serve --port 45999 '
                "--hostname 127.0.0.1")
    monkeypatch.setattr(
        MS, "_process_command_line",
        lambda pid: (seen.append(pid) or matching), raising=False)

    MS._kill_pid(os.getpid())
    MS._kill_pid(0)

    assert seen == [], f"a self/group pid reached the cmdline probe: {seen}"
    assert signalled == [], "a self/group pid reached os.kill"

    # Control: a pid that is neither self nor falsy still gets probed, so the
    # two asserts above are the guard working and not a dead probe. The command
    # line matches, so this one IS reaped -- the guard must not be
    # "never signal anything".
    MS._kill_pid(4242)
    assert seen == [4242], seen
    assert [pid for _channel, pid in signalled] == [4242], \
        "a real opencode serve pid must still be reaped, not swallowed"


@pytest.mark.parametrize("pid", [-1, 0])
def test_non_positive_pids_are_refused_by_the_command_line_probe(monkeypatch, pid):
    """The OTHER pid guard, which is not the early return.

    ``-1`` is truthy and is not our pid, so it passes ``_kill_pid``'s
    ``if not pid or pid == os.getpid()`` line and is stopped one layer down:
    ``_process_command_line`` returns "" for any ``pid <= 0`` before it ever
    touches the OS, and an empty command line is not identifiable, so nothing
    is signalled. Pinned separately so the two guards are not confused for one
    another -- and so nobody "simplifies" one of them away.
    """
    signalled = _no_signals(monkeypatch)

    assert MS._process_command_line(pid) == "", \
        "a non-positive pid must not be probed against the OS"
    MS._kill_pid(pid)

    assert signalled == [], f"pid {pid} was signalled"


def test_windows_command_line_reader_strips_the_wmic_header(monkeypatch):
    """The Windows reader must be wired to something real (CIM, wmic
    fallback), not a placeholder that always fails open.

    Platform-honest: this exercises ``_windows_command_line`` DIRECTLY, so it
    runs and is verified on a Linux runner too. The pre-change version faked
    ``ms.os.name = "nt"`` to steer the ``_process_command_line`` dispatch --
    ``ms.os`` is the ``os`` module itself, so that mutated ``os.name`` for the
    whole process, and ``pathlib.Path`` branches on ``os.name`` when choosing
    a flavour. On Linux that makes ``Path()`` return a ``WindowsPath``, which
    raises ``NotImplementedError`` on instantiation, so any code reached
    during the test that touched the filesystem would explode for a reason
    that has nothing to do with what was being tested. The dispatch is
    platform-honest below instead, and the reader itself is tested here.
    """
    from deepseaport.providers.opencode import managed_serve as ms

    ran: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = "CommandLine\r\nC:\\oc\\opencode.exe serve --port 45999\r\n\r\n"

    monkeypatch.setattr(ms.shutil, "which", lambda name: f"C:/fake/{name}")
    monkeypatch.setattr(ms.subprocess, "run",
                        lambda argv, **kw: (ran.append(list(argv)), _Done())[1])

    assert ms._windows_command_line(7) == "C:\\oc\\opencode.exe serve --port 45999"
    assert len(ran) == 1, "the CIM attempt succeeded; wmic must not run"
    assert "Get-CimInstance" in ran[0][-1]
    assert ms._looks_like_managed_serve(
        ms._windows_command_line(7)) is True

    # The wmic fallback is only reached when CIM is unavailable.
    ran.clear()
    monkeypatch.setattr(ms.shutil, "which",
                        lambda name: None if name == "powershell" else f"C:/fake/{name}")
    assert ms._windows_command_line(7) == "C:\\oc\\opencode.exe serve --port 45999"
    assert len(ran) == 1, "wmic must be the fallback when CIM is missing"
    assert "wmic" in ran[0][0]

    # Neither reader on PATH -> "" -> the reaper refuses to signal. Fails
    # closed, which is the security-relevant direction.
    ran.clear()
    monkeypatch.setattr(ms.shutil, "which", lambda name: None)
    assert ms._windows_command_line(7) == ""
    assert ms._looks_like_managed_serve("") is False
    assert ran == []


def test_process_command_line_dispatch_follows_the_real_platform(monkeypatch):
    """The ``_process_command_line`` reader selection, tested on the real
    platform instead of by patching ``os.name`` process-wide.

    Both halves are faked, so this is deterministic on Linux and Windows and
    asserts only that the dispatcher calls the reader its platform names.
    """
    from deepseaport.providers.opencode import managed_serve as ms

    picked: list[str] = []
    monkeypatch.setattr(ms, "_windows_command_line",
                        lambda pid: (picked.append("windows"), "C:\\oc\\opencode.exe serve --port 45999")[1])
    monkeypatch.setattr(ms, "_posix_command_line",
                        lambda pid: (picked.append("posix"), "/usr/bin/opencode serve --port 45999")[1])

    out = ms._process_command_line(7)
    expected = "windows" if os.name == "nt" else "posix"

    assert picked == [expected], f"dispatcher chose {picked}, expected {expected}"
    assert "opencode" in out
    # Garbage never reaches either reader.
    for bad in (None, "nope", 0, -1):
        picked.clear()
        assert ms._process_command_line(bad) == ""


@pytest.mark.skipif(os.name == "nt", reason="/proc/<pid>/cmdline is POSIX only")
def test_posix_command_line_reads_a_real_process():
    from deepseaport.providers.opencode import managed_serve as ms

    out = ms._posix_command_line(os.getpid())
    assert "python" in out.lower()
    assert "\0" not in out  # NUL-separated argv was joined, not leaked
    assert ms._process_command_line(os.getpid()) == out
    assert ms._posix_command_line(0) == ""


# --- K5: config.json has no 0600 on Windows ----------------------------------


def test_acl_warning_is_emitted_once_on_windows(tmp_path, monkeypatch, caplog):
    """``os.chmod(tmp, 0o600)`` only toggles the read-only attribute on
    win32 and never touches the DACL, so config.json (account emails,
    passwords, 64-char bearer tokens) keeps the directory's inherited ACL --
    on a default install ``BUILTIN\\Users`` has read access. The code claimed
    owner-only and the operator was never told otherwise."""
    from deepseaport import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_ACL_WARNING_EMITTED", False, raising=False)
    target = tmp_path / "config.json"

    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        cfg_mod._warn_inherited_acl_once(target, platform="nt")
        first = [r for r in caplog.records if "inherits the ACL" in r.getMessage()]
        assert len(first) == 1, "no ACL warning on a Windows save"
        # One per process: every account add/select/ban persists, and a
        # warning per save would bury the log it lives in.
        caplog.clear()
        cfg_mod._warn_inherited_acl_once(target, platform="nt")
        assert [r for r in caplog.records if "inherits the ACL" in r.getMessage()] == []

    message = first[0].getMessage()
    assert str(target) in message
    assert "0600" in message, "the warning must say the chmod does not apply"
    assert "bearer token" in message, "the warning must name what is at risk"
    # The concrete mitigation, not a vague "be careful".
    assert "DEEPSEAPORT_CONFIG" in message


def test_acl_warning_is_silent_on_posix(tmp_path, monkeypatch, caplog):
    """POSIX keeps the real 0600, so the warning would be noise there."""
    from deepseaport import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_ACL_WARNING_EMITTED", False, raising=False)

    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        cfg_mod._warn_inherited_acl_once(tmp_path / "config.json", platform="posix")

    assert "inherits the ACL" not in caplog.text
    assert getattr(cfg_mod, "_ACL_WARNING_EMITTED", False) is False, \
        "the POSIX save consumed the once-only flag"


def test_config_save_runs_the_acl_check_on_every_write(tmp_path, monkeypatch):
    """Wiring: a successful save must hand the written file to the checker.
    (Both branches themselves are covered above; this is the call site.)"""
    from deepseaport import config as cfg_mod

    checked: list = []
    monkeypatch.setattr(cfg_mod, "_warn_inherited_acl_once",
                        lambda target, **kw: checked.append(target), raising=False)
    path = tmp_path / "config.json"
    settings = Settings(accounts=[AccountConfig(email="a@x.com", token=LIVE_TOKEN)],
                        config_path=str(path))

    settings.save()

    assert checked == [path], "save() never reported the written file"
    assert json.loads(path.read_text(encoding="utf-8"))["accounts"][0]["token"] \
        == LIVE_TOKEN, "the save itself must still have worked"


@pytest.mark.skipif(os.name != "nt",
                    reason="the end-to-end case only exists on Windows")
def test_config_save_on_windows_really_warns(tmp_path, monkeypatch, caplog):
    """End to end on the platform where the no-op chmod actually bites."""
    from deepseaport import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_ACL_WARNING_EMITTED", False, raising=False)
    path = tmp_path / "config.json"
    settings = Settings(accounts=[AccountConfig(email="a@x.com", token=LIVE_TOKEN)],
                        config_path=str(path))
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        settings.save()
    assert "inherits the ACL" in caplog.text, \
        "on Windows the save claimed 0600 protection it does not have"
    assert json.loads(path.read_text(encoding="utf-8"))["accounts"]


# --- K6: upstream bodies must not travel verbatim ---------------------------


def test_scrubber_redacts_tokens_and_keeps_the_diagnostics():
    body = ("HTTP_500: {'code': 1002, 'token': '" + LIVE_TOKEN +
            "', 'msg': 'generation_timeout: Server busy'}")

    out = engine_mod.scrub_upstream_text(body, 800)

    assert LIVE_TOKEN not in out, "the 64-char token survived scrubbing"
    assert "[redacted]" in out
    # Everything an operator diagnoses with stays, verbatim.
    for word in ("HTTP_500", "1002", "generation_timeout", "Server busy", "code"):
        assert word in out, f"diagnostic text {word!r} was scrubbed away"
    assert len(out) < len(body)


def test_scrubber_clamps_the_length():
    out = engine_mod.scrub_upstream_text("HTTP_500: boom " * 500, 100)
    assert out.startswith("HTTP_500: boom"), "the head of the body must survive"
    assert "[+7400 chars]" in out, "the drop must be accounted for, not silent"
    assert len(out) < 200
    assert engine_mod.scrub_upstream_text("HTTP_500: boom", 0) == "HTTP_500: boom"
    assert engine_mod.scrub_upstream_text(None) == ""


def test_upstream_502_detail_and_log_redact_the_token(caplog):
    body = "HTTP_500: internal error, trace=" + LIVE_TOKEN + " (biz_code 1002)"

    with caplog.at_level(logging.WARNING, logger="deepseaport.core.engine"):
        with pytest.raises(HTTPException) as exc_info:
            engine_mod.raise_upstream("DeepSeek", body)

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 502
    assert LIVE_TOKEN not in detail, "the token reached the client"
    assert LIVE_TOKEN not in caplog.text, "the token reached the server log"
    assert "upstream error" in detail and "1002" in detail
    assert "[redacted]" in detail


def test_upstream_422_detail_redacts_the_token():
    body = 'HTTP_422: {"detail":[{"loc":["body"],"token":"' + LIVE_TOKEN + '"}]}'

    with pytest.raises(HTTPException) as exc_info:
        engine_mod.raise_upstream_invalid("DeepSeek", {"prompt": "hi", "model": "m"},
                                          body)

    detail = str(exc_info.value.detail)
    assert LIVE_TOKEN not in detail
    assert "HTTP_422" in detail and "rejected the request as invalid" in detail


def test_ban_403_detail_redacts_the_login_page_token(monkeypatch):
    """The ban path quotes up to 300 chars of the upstream error text, and the
    Obscura path feeds it scraped login-page DOM (which carries the very
    token the refresh failed on)."""
    from deepseaport.providers.deepseek.completion_service import _ban_403_for_item

    settings = _settings(accounts=[AccountConfig(email="b@x.com", token="tok")])
    app = create_app(settings)
    item = PooledAccount(cfg=settings.accounts[0])
    login_dom = ("Obscura login page shows suspended: <html>userToken=" + LIVE_TOKEN +
                 "</html>")

    exc = _ban_403_for_item(app, item, None, login_dom)

    assert exc.status_code == 403
    assert LIVE_TOKEN not in str(exc.detail)
    assert "suspended" in str(exc.detail), "the ban evidence was scrubbed away"
    assert "[redacted]" in str(exc.detail)


def test_invalid_token_401_detail_redacts_the_login_note():
    note = '{"value":"' + LIVE_TOKEN + '","__version":"0"}'

    detail = CS._invalid_token_401_detail("a@x.com", "40003: Authorization Failed",
                                          note, refreshed=False)

    assert LIVE_TOKEN not in detail, "the login note leaked the userToken"
    assert "40003" in detail and "token invalid" in detail.lower()
    assert "BANNED" in detail


# --- K7: the access log must not record bearer tokens -----------------------


def test_serve_config_keeps_the_access_log_off(monkeypatch, tmp_path):
    """Pre-change: uvicorn's access log stayed on for the main serve path, and
    ``accounts._matches`` accepts a FULL token as an account identifier, so
    ``DELETE /v1/accounts/<token>`` wrote a live bearer token into the log."""
    import argparse
    import types

    import deepseaport.server as srv_mod
    import uvicorn

    from deepseaport import server_runner

    seen: dict = {}

    monkeypatch.setattr(uvicorn, "Config",
                        lambda app, **kw: (seen.update(kw),
                                           types.SimpleNamespace())[1])
    monkeypatch.setattr(uvicorn, "Server",
                        lambda config: types.SimpleNamespace(run=lambda: None))
    monkeypatch.setattr(srv_mod, "create_app", lambda settings=None: object())
    monkeypatch.setattr(srv_mod, "apply_log_level", lambda *a, **k: None)
    monkeypatch.setattr(server_runner, "_watch_stop_keys", lambda *a, **k: None)

    settings = _settings(port=5001, config_path=str(tmp_path / "c.json"))
    assert server_runner._run_server(
        settings, argparse.Namespace(host="127.0.0.1", port=0, workers=1),
        ensure_free=lambda s, h, p: p) == 0

    assert seen.get("access_log", "<missing>") is False, \
        "uvicorn would log every request target, and an account identifier " \
        "may be a full bearer token"
    # Nothing else regresses: uvicorn's error log is a different logger and is
    # still configured, so startup/uncaught-exception output is unaffected.
    assert seen.get("log_level") == "warning"
    assert seen.get("limit_concurrency") == server_runner.UVICORN_LIMIT_CONCURRENCY
    assert seen.get("timeout_keep_alive") == server_runner.UVICORN_TIMEOUT_KEEP_ALIVE


def test_accounts_route_rejects_a_full_token_identifier(monkeypatch):
    """A follow-up closed the leak at the route, not only at the log.

    ``accounts._matches`` still matches a full token, because the CLI
    (``deepseaport accounts delete <token>``) and the TUI need it. Over HTTP a
    URL is the wrong place for a credential: paths land in access logs, proxy
    logs and ``Referer`` headers, so the edge refuses a token-shaped segment
    with a 400 that never echoes the value back.
    """
    from fastapi.testclient import TestClient

    import deepseaport.server as srv_mod

    settings = _settings(accounts=[AccountConfig(token=LIVE_TOKEN, email="")],
                         keys=["sk-test"])
    app = create_app(settings)
    monkeypatch.setattr(srv_mod, "_pool",
                        lambda app_: AccountPool(settings.accounts, current=""))

    with TestClient(app) as client:
        listed = client.get("/v1/accounts",
                            headers={"Authorization": "Bearer sk-test"})
        assert listed.status_code == 200
        rejected = client.delete(f"/v1/accounts/{LIVE_TOKEN}",
                                 headers={"Authorization": "Bearer sk-test"})

    assert rejected.status_code == 400, rejected.text
    assert LIVE_TOKEN not in rejected.text
    assert len(settings.accounts) == 1  # nothing was removed


# --- K8: the four paths K6 missed -------------------------------------------

# What a provider client actually raises. providers/deepseek/client.py does
# this twice (`create_session: no id (HTTP 403): <resp.text[:200]>` and
# `pow challenge: bad shape: ...`), and that body is the one most likely to
# echo back the Authorization header the same request was just given.
RAW_UPSTREAM_BODY = (
    "{'biz_code': 40003, 'Authorization': 'Bearer " + LIVE_TOKEN + "', "
    "'token': '" + LIVE_JWT + "', 'msg': 'user is muted'}")

# The three producer tails, byte-identical sibling copies. Listed with the seam
# each `_produce` calls so the real producer thread, queue and consumer run and
# only the upstream work is replaced.
_PRODUCER_TAILS = [
    pytest.param(CS, "_complete_with_failover_sync", "deepseek",
                 "deepseek-flash", id="deepseek"),
    pytest.param(QCS, "_complete_with_failover_sync", "qwen",
                 "qwen3.7-plus", id="qwen"),
    pytest.param(OCS, "_run_completion_core", "opencode",
                 "opencode/muse-spark-1.3-contributor-free", id="opencode"),
]


def _crashing_stream(monkeypatch, mod, seam, provider, model) -> str:
    """One provider run whose producer raises the way the clients raise.

    ``item`` is None for all three: the producer entry is the only thing that
    ever touched it, and that is what was replaced.
    """
    def _boom(*_a, **_k):
        raise RuntimeError(f"create_session: no id (HTTP 403): {RAW_UPSTREAM_BODY}")

    monkeypatch.setattr(mod, seam, _boom)
    st = settings(provider=provider,
                  accounts=[AccountConfig(email="a@x.com", token="tok")])
    app = create_app(st)
    pool = AccountPool(st.accounts, current=st.active_account)
    prep = _prepare({"model": model, "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})
    return sse_text(asyncio.run(collect_sse(mod, app, pool, None, prep)))


@pytest.mark.parametrize("mod,seam,provider,model", _PRODUCER_TAILS)
def test_producer_tail_scrubs_the_exception_it_emits(monkeypatch, mod, seam,
                                                     provider, model):
    """K8a. ``HTTPException`` goes to ``_raise_for_terminal_error`` and is
    scrubbed there; EVERY other exception fell into a bare ``except Exception``
    and ``str(exc)`` went out raw. With the SSE headers already sent the client
    gets HTTP 200 carrying the upstream body inside ``{"error": {"message"}}``.

    The frame is stubbed to a PASSTHROUGH so this asserts the tail alone.
    Without that, the frame's own clamp+scrub would mask a regression here and
    the two fixes would only ever be tested as one.
    """
    monkeypatch.setattr(engine_mod, "sse_error_frame",
                        lambda message, status_code=None: (
                            b"data: " + json.dumps(
                                {"error": {"message": message}}).encode() + b"\n\n"))

    body = _crashing_stream(monkeypatch, mod, seam, provider, model)

    assert LIVE_TOKEN not in body, "the SSE stream carried a live bearer token"
    assert LIVE_JWT not in body, "the SSE stream carried a live JWT"
    assert "[redacted]" in body, "the tail did not scrub at all"
    # Scrubbing is not censorship: the status and the reason an operator
    # diagnoses with are still there.
    assert "HTTP 403" in body and "user is muted" in body, body


def test_sse_error_frame_clamps_and_scrubs_its_message():
    """K8b. Pre-change: ``str(message)`` with no clamp, while every other
    upstream-text path in the module clamped. The response has no
    Content-Length yet, and the watchdog / generic-consumer paths raise
    locally, so a locally-raised frame carried whatever ``str(exc)`` held.
    """
    raw = ("consumer failed: " + RAW_UPSTREAM_BODY
           + " " + ("padding " * 900))

    frame = json.loads(engine_mod.sse_error_frame(raw, 502)
                       .decode().removeprefix("data: "))
    msg = frame["error"]["message"]

    assert LIVE_TOKEN not in msg, "the SSE error frame carried a live bearer token"
    assert LIVE_JWT not in msg, "the SSE error frame carried a live JWT"
    # Delegated to the scrubber, not sliced independently: one place decides
    # the bound, so a caller cannot be clamped shorter than the frame allows.
    assert msg == engine_mod.scrub_upstream_text(raw, 800)
    assert "[+" in msg, "an unbounded message must be clamped, not silently cut"
    assert "consumer failed:" in msg, "the head of the message must survive"
    # Only the message was bounded; the OpenAI shape is unchanged.
    assert frame["error"]["type"] == "upstream_error"
    assert frame["error"]["code"] == 502
    assert json.loads(engine_mod.sse_error_frame(None).decode()
                      .removeprefix("data: "))["error"]["message"] == ""


def test_non_200_completion_log_does_not_carry_the_token_it_was_given(caplog):
    """K8c. ``build_http_error_event`` logged 500 chars of the upstream body on
    EVERY non-200 from both web providers, before the terminal paths ever ran.
    The request that provoked the error is precisely the one carrying the
    account's bearer token. The ``%.500s`` was a length guard, not a secret
    guard; the bound is the scrubber's now.
    """
    class _Resp:
        status_code = 403
        headers = {"content-type": "application/json"}
        text = "HTTP_403: " + RAW_UPSTREAM_BODY

    with caplog.at_level(logging.WARNING, logger="deepseaport.client"):
        event = transport_mod.build_http_error_event(
            _Resp(), {"prompt": "hi", "model": "deepseek-flash"})

    assert LIVE_TOKEN not in caplog.text, "the completion log carried a live bearer token"
    assert LIVE_JWT not in caplog.text, "the completion log carried a live JWT"
    assert "[redacted]" in caplog.text
    assert "HTTP_403" in caplog.text and "user is muted" in caplog.text, \
        "scrubbing must not take the diagnosis out of the log"
    # The returned event is left alone on purpose: it flows to the terminal
    # paths, which scrub it there.
    assert event.kind == "error" and event.code == "HTTP_403_WAF"


def test_recovery_failure_log_is_scrubbed_and_traceback_free(caplog):
    """K8d. Pre-change: ``_run_recovery`` interpolated ``exc`` AND passed
    ``exc_info=True``, which prints ``str(exc)`` plus the traceback. A recovery
    calls real work -- ``_recover_pow`` -> ``fetch_pow`` raises with
    ``resp.text[:200]`` in the message -- so a failed recovery wrote the same
    raw body to the log twice, from the one place that never reaches the
    terminal ladder.
    """
    boom = RuntimeError(f"pow challenge: bad shape: {RAW_UPSTREAM_BODY}")

    with caplog.at_level(logging.WARNING, logger="deepseaport.core.engine"):
        ok = engine_mod._run_recovery(
            "pow", lambda: (_ for _ in ()).throw(boom))

    assert ok is False, "a failed recovery must stay contained (E1 still holds)"
    assert LIVE_TOKEN not in caplog.text, \
        "the recovery-failure log carried a live bearer token"
    assert LIVE_JWT not in caplog.text, \
        "the recovery-failure log carried a live JWT"
    assert "Traceback" not in caplog.text, (
        "exc_info=True re-prints str(exc) at the bottom of the traceback, "
        "which is the same unscrubbed body a second time")
    # Still diagnosable: which recovery, what kind of fault, and why.
    assert "'pow'" in caplog.text and "RuntimeError" in caplog.text, caplog.text
    assert "user is muted" in caplog.text, "scrubbing must not eat the diagnosis"


# K9: qwen shipped the bridge's DeepSeek jar to chat.qwen.ai ------------------
#
# The app bridge holds ONE jar, warmed against CHAT_HOME (chat.deepseek.com).
# qwen's `_headers` read `bridge.cookie_header()` and put the whole jar on
# every qwen request, i.e. handed chat.qwen.ai a live aws-waf-token its
# operator never gave it. Qwen's own cookies are its client-side acw_tc cache
# (prime/fetch against the qwen host); the bridge now contributes only the UA.


def test_qwen_headers_never_ship_the_bridge_jar(monkeypatch):
    from deepseaport.providers.qwen import client as QC

    QC._clear_waf_cache()
    try:
        class _State:
            user_agent = "DS-UA"

        class _Bridge:
            state = _State()

            def cookie_header(self, names=None):
                return "aws-waf-token=SECRETWAF; intercom-device-id=ID"

        class _St:
            bridge = _Bridge()

        class _App:
            state = _St()

        monkeypatch.setattr("deepseaport.core.obscura_bridge.get_default_bridge",
                            lambda: _Bridge())

        h = QCS._headers(_App(), "jwtLEAK")
        assert "aws-waf-token" not in str(h)
        assert "SECRETWAF" not in str(h)
        assert h["Cookie"] == "token=jwtLEAK"
        assert h["User-Agent"] == "DS-UA"

        # The no-app-bridge fallback must not source the DeepSeek jar either.
        h2 = QCS._headers(None, "jwtLEAK2")
        assert "aws-waf-token" not in str(h2)
        assert "SECRETWAF" not in str(h2)
        assert h2["Cookie"] == "token=jwtLEAK2"
        assert h2["User-Agent"] == "DS-UA"
    finally:
        QC._clear_waf_cache()

