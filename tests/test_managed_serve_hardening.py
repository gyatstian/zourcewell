"""Managed opencode serve hardening: reap identity, secret dir, child env.

Six confirmed findings, each pinned here by a test that fails on the
pre-change code:

* P2-1 -- the stale-serve reaper's identity check was a SUBSTRING search
  ("opencode" and "serve" appear anywhere in the line), so
  ``notepad.exe C:\\opencode\\serve.txt``,
  ``powershell -c Write-Host opencode serve`` and ``calc.exe serve opencode``
  all passed it. The pid it reads is an ordinary JSON file in the data dir,
  so the control bought nothing.
* P2-2 -- ``data/opencode-managed/serve.json`` holds the live
  ``OPENCODE_SERVER_PASSWORD`` in cleartext and the dir was created with
  mkdir()'s umask, while the sibling obscura cookie-jar dir had just been
  hardened to 0700 + a one-shot Windows ACL warning. ``os.chmod(f, 0o600)``
  is a documented no-op on win32.
* P2-3 -- the upstream-text scrubber redacted ``[A-Za-z0-9_-]{32,}`` only, so
  a dotted (JWT-shaped) secret passed through whole: short segments never trip
  a 32-char run rule.
* P2-4 -- the child env strip was prefix-shaped
  (DEEPSEAPORT_/DEEPSEEK_/OPENCODE_), so ``ANTHROPIC_API_KEY``,
  ``OPENAI_API_KEY``, ``GITHUB_TOKEN`` and the AWS pair rode into the agentic
  child.
* P2-5 -- ``_invalid_token_401_detail`` kept a dead, UNSCRUBBED second return
  (the raw ``error[:200]``) below an unconditional one: a live leak the day
  someone reorders the ``if refreshed:`` block above it.
* P2-6 -- the whole reap was DEAD on Windows. CPython hands a ``.cmd``
  launcher to the OS shell, so the pid recorded in serve.json is
  ``cmd.exe /c .../opencode.CMD serve ...``: argv[0] is ``cmd``, the identity
  gate refused it on every single run ("does not look like 'opencode serve'"),
  and even a signalled wrapper left the real ``opencode.exe`` child holding its
  port. Every deepseaport restart leaked one.

No network, no opencode binary, no credentials: every process identity is a
monkeypatched command line and every "os.kill" is a fake.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import stat
import subprocess
import textwrap

import pytest

from deepseaport.core import engine as engine_mod
from deepseaport.core import obscura_bridge as OB
from deepseaport.providers.deepseek import completion_service as CS
from deepseaport.providers.opencode import managed_serve as MS


# 64 opaque chars, exactly what scrub_upstream_text is meant to catch.
LIVE_TOKEN = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"

# A JWT: 20 / 30 / 9 chars, so no segment reaches the 32-char run threshold.
JWT = ("eyJhbGciOiJIUzI1NiJ9"
       ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
       ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")


# --------------------------------------------------------------------------
# P2-1: the reaper must identify its target, not match a word in a sentence
# --------------------------------------------------------------------------

# Command lines that contain BOTH words and are not `opencode serve`. Every
# one of these was signalled before the fix.
SPOOFED_CMDLINES = [
    r"C:\Windows\notepad.exe C:\opencode\serve.txt",
    r"C:\opencode\serve.exe --port 45999",
    "powershell -NoProfile -NonInteractive -Command Write-Host opencode serve",
    "calc.exe serve opencode",
    "cmd.exe /c start opencode serve",
]

# The shapes the child is actually spawned as (subprocess.Popen(binary, ...)).
REAL_CMDLINES = [
    "opencode serve --port 45999 --hostname 127.0.0.1",
    r"C:\oc\opencode.exe serve --port 45999 --hostname 127.0.0.1",
    r'"/c/Users/op/.bun/bin/opencode" serve --port 45999 --hostname 127.0.0.1',
    r'"C:\Program Files\opencode\opencode.exe" serve --port=45999',
    "OPENCODE serve --port 45999",  # argv[0] case is irrelevant
    # Windows: CPython hands a .cmd/.bat launcher to the OS shell, so the pid
    # recorded in serve.json is the cmd.exe wrapper and the serve is its child.
    r"C:\WINDOWS\system32\cmd.exe /c C:\Users\op\AppData\Roaming\npm\opencode.CMD "
    r"serve --port 45999 --hostname 127.0.0.1",
    r'cmd.exe /c "C:\Users\op\AppData\Roaming\npm\opencode.cmd" serve --port 45999',
    r"C:\Windows\System32\cmd.exe /c C:\oc\opencode.exe serve --port=45999",
]

# The wrapper branch is the one that can loosen the gate, so it gets its own
# spoofs: a cmd.exe that merely mentions opencode LATER in the line is not the
# launcher we spawned, and neither is one whose /c target is another program.
WRAPPER_SPOOFED_CMDLINES = [
    "cmd.exe /c start opencode serve --port 45999",
    "cmd.exe /c notepad.exe serve --port 45999",
    r"cmd.exe /c echo opencode serve --port 45999",
    # right launcher, wrong subcommand
    r"cmd.exe /c C:\npm\opencode.CMD run --port 45999",
    # right launcher, but the port belongs to another instance
    r"cmd.exe /c C:\npm\opencode.CMD serve --port 1234",
    # not a /c invocation at all
    r"cmd.exe C:\npm\opencode.CMD serve --port 45999",
]


@pytest.mark.parametrize("cmdline", SPOOFED_CMDLINES)
def test_reaper_gate_rejects_command_lines_that_only_mention_the_words(cmdline):
    """Pre-change: ``all(token in low for token in ("opencode", "serve"))``.

    Every shape here satisfies that test, which is the whole problem: the
    gate existed to keep a writable serve.json from aiming a SIGTERM at an
    arbitrary same-user process, and it did not.
    """
    assert MS._looks_like_managed_serve(cmdline) is False, (
        "a command line that merely contains the words was accepted as the "
        f"managed serve: {cmdline!r}")


@pytest.mark.parametrize("cmdline", REAL_CMDLINES)
def test_reaper_gate_accepts_the_real_managed_serve_shape(cmdline):
    """The other half: an over-tight gate would silently stop the reap, and a
    stale serve holds its port forever."""
    assert MS._looks_like_managed_serve(cmdline) is True, (
        f"the real managed serve was rejected: {cmdline!r}")


@pytest.mark.parametrize("cmdline", WRAPPER_SPOOFED_CMDLINES)
def test_reaper_gate_rejects_a_cmd_wrapper_that_is_not_our_launcher(cmdline):
    """The Windows wrapper branch, pinned from the other side.

    Accepting ``cmd.exe`` lines is only safe because exactly three tokens are
    allowed in front of the subcommand (``cmd`` ``/c`` ``<opencode>``). Every
    case here is a cmd.exe line that mentions opencode or a matching --port
    somewhere and must still be refused, or a writable serve.json could aim the
    reap at a stranger.
    """
    assert MS._looks_like_managed_serve(cmdline, "http://127.0.0.1:45999") is False, (
        f"a cmd wrapper that is not the managed launcher was accepted: {cmdline!r}")


def test_reaper_gate_accepts_the_wrapper_only_when_the_recorded_port_agrees():
    """The wrapper does not buy an exemption from the port cross-check."""
    line = (r"C:\WINDOWS\system32\cmd.exe /c C:\npm\opencode.CMD serve "
            r"--port 45999 --hostname 127.0.0.1")
    assert MS._looks_like_managed_serve(line, "http://127.0.0.1:45999") is True
    assert MS._looks_like_managed_serve(line, "http://127.0.0.1:1234") is False


def test_reaper_gate_requires_serve_as_an_argument_not_a_word():
    """``serve`` may be an argument, and the executable must BE opencode."""
    assert MS._looks_like_managed_serve("opencode serve") is True
    # the word is there, but not as an argument
    assert MS._looks_like_managed_serve(
        "opencode run --prompt 'restart opencode serve'") is False
    # right subcommand family, wrong program
    assert MS._looks_like_managed_serve("opencode.exe --help") is False


@pytest.mark.parametrize("cmdline", [
    "",                     # no /proc, no PowerShell, no wmic
    "   ",                  # reader returned whitespace only
    '"unbalanced opencode serve',   # unparseable: shlex raises on this
    "serve",                # no executable at all
    "opencode",             # executable only, no subcommand
])
def test_reaper_gate_skips_an_unreadable_or_unparseable_command_line(cmdline):
    """Conservative by contract: no signal, and no raise either."""
    try:
        assert MS._looks_like_managed_serve(cmdline) is False
    except Exception as exc:  # pragma: no cover - the contract is "never raises"
        pytest.fail(f"the identity gate raised on {cmdline!r}: {exc!r}")


def test_reaper_gate_skips_a_port_that_disagrees_with_the_port_file():
    """A serve on another port is not the instance whose pid was recorded."""
    line = "opencode serve --port 45999 --hostname 127.0.0.1"
    assert MS._looks_like_managed_serve(line, "http://127.0.0.1:45999") is True
    assert MS._looks_like_managed_serve(line, "http://127.0.0.1:1234") is False
    # --port=N parses, and a base/command line that carry no port simply
    # skip the cross-check rather than blocking a legitimate reap.
    assert MS._looks_like_managed_serve("opencode serve --port=45999",
                                       "http://127.0.0.1:45999") is True
    assert MS._looks_like_managed_serve(line, "http://127.0.0.1") is True
    assert MS._looks_like_managed_serve(line, "") is True


def _no_signals(monkeypatch) -> list:
    """Record every channel the reaper could use, as ``(channel, pid)``.

    Windows kills with ``taskkill /T`` and POSIX with ``os.kill``, so both are
    stubbed: a test that only watched ``os.kill`` would pass vacuously there.
    """
    signalled: list[tuple] = []

    def _kill(pid, sig):
        signalled.append(("os.kill", pid))
        raise OSError("no such process")

    def _tree(pid):
        signalled.append(("taskkill", pid))
        return True

    monkeypatch.setattr(MS.os, "kill", _kill)
    monkeypatch.setattr(MS, "_kill_tree", _tree, raising=False)
    return signalled


def test_reaper_does_not_signal_a_spoofed_opencode_serve(monkeypatch, tmp_path):
    """End to end: a serve.json written by something other than us must not be
    able to move the SIGTERM onto another program."""
    signalled = _no_signals(monkeypatch)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    (tmp_path / "m").mkdir()
    monkeypatch.setattr(MS, "_process_command_line",
                        lambda pid: r"C:\Windows\notepad.exe C:\opencode\serve.txt",
                        raising=False)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:45999",
                        "pw", 4242, "hash")

    MS._kill_pid(4242)

    assert signalled == [], "a notepad was signalled as if it were opencode serve"


def test_reaper_still_signals_the_real_serve_with_matching_port(monkeypatch, tmp_path):
    """The port file is the source of the cross-check, so it must be read from
    there and must not break the reap when it agrees."""
    signalled = _no_signals(monkeypatch)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    (tmp_path / "m").mkdir()
    monkeypatch.setattr(MS, "_process_command_line",
                        lambda pid: "opencode serve --port 45999 --hostname 127.0.0.1",
                        raising=False)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:45999",
                        "pw", 4242, "hash")

    MS._kill_pid(4242)

    assert [pid for _channel, pid in signalled] == [4242]


def test_reaper_signals_the_windows_wrapper_that_spawned_the_serve(monkeypatch,
                                                                   tmp_path):
    """The recorded pid on Windows is ``cmd.exe``, not the serve.

    Pre-fix the gate read argv[0] as ``cmd``, refused the pid, and every
    restart logged "does not look like 'opencode serve'" and left the old
    serve running on its port.
    """
    signalled = _no_signals(monkeypatch)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    (tmp_path / "m").mkdir()
    monkeypatch.setattr(MS, "_process_command_line",
                        lambda pid: (
                            r"C:\WINDOWS\system32\cmd.exe /c "
                            r"C:\Users\op\AppData\Roaming\npm\opencode.CMD serve "
                            r"--port 45999 --hostname 127.0.0.1"),
                        raising=False)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:45999",
                        "pw", 4242, "hash")

    MS._kill_pid(4242)

    assert [pid for _channel, pid in signalled] == [4242]


# --------------------------------------------------------------------------
# P2-6: terminating the Windows wrapper left the serve itself running
# --------------------------------------------------------------------------


def test_kill_tree_is_the_windows_kill_and_a_noop_elsewhere(monkeypatch):
    """``taskkill /T`` is the only way to reach the serve behind the shim.

    ``Popen.terminate()`` on Windows kills ``cmd.exe`` and orphans
    ``opencode.exe``, which keeps holding its port; the tree flag is what makes
    a shutdown or a restart actually give the port back.
    """
    calls: list[list[str]] = []

    def _run(argv, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(MS.subprocess, "run", _run)
    monkeypatch.setattr(MS.os, "name", "nt")

    assert MS._kill_tree(4242) is True
    assert calls and calls[0][:2] == ["taskkill", "/PID"]
    assert "4242" in calls[0]
    assert "/T" in calls[0] and "/F" in calls[0], \
        "without /T the serve behind the cmd wrapper survives"

    calls.clear()
    monkeypatch.setattr(MS.os, "name", "posix")
    assert MS._kill_tree(4242) is False
    assert calls == [], "POSIX must not shell out to taskkill"


def test_kill_tree_never_raises(monkeypatch):
    """A missing taskkill, a dead pid or a timeout is a stale serve, not a
    failed boot."""
    monkeypatch.setattr(MS.os, "name", "nt")

    def _boom(*a, **k):
        raise OSError("taskkill is not available")

    monkeypatch.setattr(MS.subprocess, "run", _boom)
    assert MS._kill_tree(4242) is False


@pytest.mark.skipif(os.name != "nt",
                    reason="the wrapper child only exists on Windows")
def test_terminate_proc_kills_the_tree_not_just_the_handle(monkeypatch):
    """The tracked child is the wrapper, so terminate() alone would leak."""
    monkeypatch.setattr(MS.os, "name", "nt")
    killed: list[int] = []
    monkeypatch.setattr(MS, "_kill_tree", lambda pid: killed.append(pid) or True)

    class _Proc:
        pid = 4242

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("escalation must not run after the tree kill")

    proc = _Proc()
    MS._terminate_proc(proc)

    assert killed == [4242], "the serve behind the wrapper was not killed"
    assert proc.terminated is False, \
        "terminate() kills only the wrapper and would orphan the serve"


@pytest.mark.skipif(os.name == "nt",
                    reason="POSIX keeps the SIGTERM path")
def test_terminate_proc_on_posix_still_terminates_the_child(monkeypatch):
    """The tree kill is Windows-only; POSIX keeps terminate() then kill()."""
    monkeypatch.setattr(MS.os, "name", "posix")
    monkeypatch.setattr(MS, "_kill_tree",
                        lambda pid: pytest.fail("taskkill on POSIX"))
    events: list[str] = []

    class _Proc:
        pid = 4242

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            events.append("kill")

    MS._terminate_proc(_Proc())

    assert events == ["terminate"]


def test_reaper_skips_when_the_recorded_port_disagrees(monkeypatch, tmp_path):
    signalled = _no_signals(monkeypatch)
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    (tmp_path / "m").mkdir()
    monkeypatch.setattr(MS, "_process_command_line",
                        lambda pid: "opencode serve --port 45999", raising=False)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:1234",
                        "pw", 4242, "hash")

    MS._kill_pid(4242)

    assert signalled == [], "a serve on a different port was signalled"


# --------------------------------------------------------------------------
# P2-2: the managed dir holds the live password; protect it like the profile
# --------------------------------------------------------------------------


def test_managed_dirs_are_owner_only_on_posix(tmp_path):
    """Pre-change: mkdir()'s umask (0755 on a default POSIX umask) left
    serve.json -- the per-boot basic-auth password in cleartext --
    world-readable."""
    base = tmp_path / "opencode-managed"
    paths = MS.write_isolated_files(base)

    assert base.is_dir()
    # POSIX modes are a no-op on Windows; the ACL warning below is what covers
    # the platform.
    if os.name != "nt":
        for target in (base, paths["work"], paths["cfg"], paths["home"],
                       paths["cfg"] / "plugins"):
            mode = stat.S_IMODE(target.stat().st_mode)
            assert mode == 0o700, f"{target} is {oct(mode)}"
    # Idempotent: a second boot on an existing dir must not raise or widen it.
    MS.write_isolated_files(base)
    assert base.is_dir()


def test_secure_dir_keeps_the_obscura_profile_contract(tmp_path):
    """The shared helper is what the profile dir uses."""
    profile = tmp_path / "prof"
    OB.secure_dir(str(profile), "obscura profile")
    assert profile.is_dir()
    if os.name != "nt":
        assert stat.S_IMODE(profile.stat().st_mode) == 0o700
    OB.secure_dir(str(profile), "obscura profile")               # idempotent
    OB.secure_dir(str(tmp_path / "missing" / "x"), "obscura profile")  # parents


def test_managed_acl_warning_names_the_password_and_fires_once(tmp_path, caplog,
                                                              monkeypatch):
    """Same mechanism and wording as config.py's warning, pointed at this file.

    ``os.chmod(serve.json, 0o600)`` only toggles the read-only attribute on
    win32 and never touches the DACL, so the password stays readable by every
    other local user the data dir grants access to.
    """
    monkeypatch.setattr(OB, "_ACL_WARNED_LABELS", set(), raising=False)
    target = tmp_path / "serve.json"

    with caplog.at_level(logging.WARNING, logger="deepseaport.obscura"):
        OB.warn_inherited_acl_once(
            target, secret="the per-boot opencode basic-auth password",
            label="managed opencode port file", platform="nt")
        first = [r for r in caplog.records if "inherits the ACL" in r.getMessage()]
        assert len(first) == 1, "no ACL warning for the managed port file"
        caplog.clear()
        # Once per store, not once per call: every deepseaport worker that
        # boots a serve re-enters here.
        OB.warn_inherited_acl_once(
            target, secret="the per-boot opencode basic-auth password",
            label="managed opencode port file", platform="nt")
        assert [r for r in caplog.records if "inherits the ACL" in r.getMessage()] == []

    message = first[0].getMessage()
    assert str(target) in message
    assert "0600" in message, "the warning must say the chmod does not apply"
    assert "basic-auth password" in message, "the warning must name the secret"
    assert "DEEPSEAPORT_DATA" in message, "the warning must give the mitigation"


def test_managed_acl_warning_is_silent_on_posix(tmp_path, monkeypatch, caplog):
    """POSIX keeps the real 0700/0600, so the warning would be noise there --
    and it must not consume the once-only flag."""
    monkeypatch.setattr(OB, "_ACL_WARNED_LABELS", set(), raising=False)

    with caplog.at_level(logging.WARNING, logger="deepseaport.obscura"):
        OB.warn_inherited_acl_once(tmp_path / "serve.json", secret="pw",
                                   label="managed opencode port file",
                                   platform="posix")

    assert "inherits the ACL" not in caplog.text
    assert OB._ACL_WARNED_LABELS == set()


def test_write_isolated_files_reports_the_serve_json_acl(tmp_path, monkeypatch):
    """Wiring: the dir creator is where the warning has to be emitted."""
    seen: list = []
    monkeypatch.setattr(
        OB, "warn_inherited_acl_once",
        lambda target, **kw: seen.append((str(target), kw)), raising=False)

    paths = MS.write_isolated_files(tmp_path / "m")

    assert [t for t, _kw in seen] == [str(paths["port_file"])], \
        "write_isolated_files never reported the port file"
    assert "password" in seen[0][1].get("secret", "")


@pytest.mark.skipif(os.name != "nt",
                    reason="the no-op chmod only bites on Windows")
def test_managed_dir_really_warns_on_windows(tmp_path, monkeypatch, caplog):
    """End to end on the platform where the no-op chmod actually applies."""
    monkeypatch.setattr(OB, "_ACL_WARNED_LABELS", set(), raising=False)

    with caplog.at_level(logging.WARNING, logger="deepseaport.obscura"):
        paths = MS.write_isolated_files(tmp_path / "m")

    assert "inherits the ACL" in caplog.text
    assert str(paths["port_file"]) in caplog.text
    MS._write_port_file(paths["port_file"], "http://127.0.0.1:45999", "pw", 1, "h")
    written = json.loads(paths["port_file"].read_text(encoding="utf-8"))
    assert written["password"] == "pw", "the port file itself must still work"


# --------------------------------------------------------------------------
# P2-3: a dotted (JWT-shaped) secret must not survive the scrubber
# --------------------------------------------------------------------------


def test_scrubber_redacts_a_jwt_shaped_secret_whole():
    """Pre-change: no segment of a JWT is 32 chars long, so the run rule
    matched nothing and the header, the payload and the signature all
    travelled verbatim into a client-visible detail and the server log."""
    out = engine_mod.scrub_upstream_text(f"401: {JWT}", 800)

    for fragment in ("eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkw",
                     "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV"):
        assert fragment not in out, f"JWT segment {fragment!r} survived scrubbing"
    assert "[redacted]" in out
    assert "401" in out, "the diagnostic code was scrubbed away"


def test_scrubber_redacts_a_whole_dot_separated_run():
    """The confirming probe: a 54-char dot-separated string, entirely
    unredacted before (no 32-char run anywhere)."""
    blob = "AbCdEfGhIjKlMnOpQrStUvWxYz.AbCdEfGhIjKlMnOpQrStUvWxYz"

    out = engine_mod.scrub_upstream_text(blob, 800)

    assert out == "[redacted]"
    assert "AbCdEfGh" not in out


def test_scrubber_leaves_dotted_prose_and_diagnostics_alone():
    """A sentence with spaces around its dots, a dotted hostname, a version
    string and an error code are all things an operator reads; redacting them
    would make the detail useless. Spaces and short segments are what keep
    them out of the framed pattern."""
    prose = ("HTTP_500: generation_timeout. Server busy. Try again in 30.5s. "
             "See docs.example.com for the opencode 1.18.32 known issues. "
             "biz_code 1002, attempt 3 of 5. Rate limit resets at 12:00.")

    out = engine_mod.scrub_upstream_text(prose, 800)

    assert out == prose
    for word in ("HTTP_500", "generation_timeout", "Server busy", "1002",
                 "docs.example.com", "1.18.32"):
        assert word in out, f"diagnostic {word!r} was redacted"


def test_scrubber_still_redacts_long_runs_and_provider_keys():
    """Both patterns exist for a reason: the run rule is still what catches
    the single-segment secrets, including DeepSeek's 64-char userToken."""
    sk_key = "sk-" + "A1b2C3d4E5f6G7h8" * 3

    for secret in (LIVE_TOKEN, sk_key):
        out = engine_mod.scrub_upstream_text(f"trace={secret} (code 1002)", 800)
        assert secret not in out
        assert "[redacted]" in out
        assert "1002" in out


# --------------------------------------------------------------------------
# P2-4: the child env strip must cover third-party credentials
# --------------------------------------------------------------------------

THIRD_PARTY = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
)


def test_child_env_drops_wellknown_third_party_credentials():
    """Pre-change: only the DEEPSEAPORT_/DEEPSEEK_/OPENCODE_ prefixes were
    stripped, so every one of these rode into the agentic child."""
    parent = {"PATH": "/usr/bin", "HOME": "/home/op", "KEEP_ME": "harmless"}
    parent.update({name: f"value-of-{name}" for name in THIRD_PARTY})
    # Windows env names are case-insensitive, so the deny check must be too.
    parent["anthropic_api_key"] = "lower-case-variant"

    env = MS._child_env(parent)

    for name in THIRD_PARTY:
        assert name not in env, f"{name} survived into the managed child"
    assert "anthropic_api_key" not in env
    assert "ANTHROPIC_API_KEY" not in json.dumps(env)
    # Unrelated process env is inherited untouched, and the source mapping is
    # never mutated.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/op"
    assert env["KEEP_ME"] == "harmless"
    assert parent["ANTHROPIC_API_KEY"] == "value-of-ANTHROPIC_API_KEY"


def test_child_env_deny_list_never_covers_a_key_the_child_needs():
    """Everything the child legitimately needs is set explicitly in ensure()
    AFTER the strip. A name on this list would be deleted a moment later and
    the child would silently misbehave, so the two sets must stay disjoint."""
    isolation_keys = {
        "XDG_CONFIG_HOME", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE",
        "OPENCODE_SERVER_PASSWORD", "DEEPSEAPORT_OPENCODE_BASE",
        "DEEPSEAPORT_OPENCODE_PASSWORD", "DEEPSEAPORT_OPENCODE_MANAGED_BASE",
    }

    assert isolation_keys & set(MS._CHILD_ENV_DENY_NAMES) == set(), \
        "a key the managed child needs is on the strip list"
    assert isolation_keys & set(MS._CHILD_ENV_STRIP_PREFIXES) == set()

    # ...and the values survive when they are set explicitly after the strip,
    # while a parent credential does not.
    env = MS._child_env({"ANTHROPIC_API_KEY": "sk-ant-parent", "PATH": "/usr/bin"})
    env["OPENCODE_SERVER_PASSWORD"] = "per-boot-password"
    env["OPENCODE_CONFIG_DIR"] = "/tmp/cfg"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["OPENCODE_SERVER_PASSWORD"] == "per-boot-password"
    assert env["OPENCODE_CONFIG_DIR"] == "/tmp/cfg"


# --------------------------------------------------------------------------
# P2-5: no unscrubbed variant of the 401 detail may exist, dead or not
# --------------------------------------------------------------------------

@pytest.mark.parametrize("refreshed", [True, False])
def test_invalid_token_401_detail_scrubs_both_branches(refreshed):
    """Both wordings are client-visible, so both must stay scrubbed."""
    detail = CS._invalid_token_401_detail(
        "a@x.com", f"40003: Authorization Failed: {LIVE_TOKEN}",
        f"Obscura login page shows userToken={LIVE_TOKEN}", refreshed)

    assert LIVE_TOKEN not in detail, f"refreshed={refreshed} leaked the token"
    assert "[redacted]" in detail
    assert "40003" in detail
    assert "token invalid" in detail.lower()
    assert "BANNED" in detail
    assert ("after browser refresh" in detail) is refreshed
