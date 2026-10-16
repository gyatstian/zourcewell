"""Managed isolated ``opencode serve`` for deepseaport-only minimal system prompt.

First-time users get zero-config startup: on the first completion the
provider boots its own dedicated ``opencode serve`` subprocess tuned for raw
API passthrough (minimal agent + deny-all permissions + system-strip
plugin). The user's own opencode setup is never touched:

- child-only env (``XDG_CONFIG_HOME`` -> empty dir, ``OPENCODE_CONFIG_DIR``
  -> generated dir, ``OPENCODE_CONFIG_CONTENT`` -> generated JSON,
  ``OPENCODE_DISABLE_PROJECT_CONFIG``/``OPENCODE_DISABLE_CLAUDE_CODE``);
  the parent process env, ``~/.config/opencode``, and any project files are
  left intact, so plain ``opencode`` runs exactly as before;
- the parent's ``OPENCODE_CONFIG`` file path is dropped from the child env
  (config merges additively, so it would otherwise leak user instructions
  into the managed instance), as is every well-known third-party provider key
  (see ``_CHILD_ENV_DENY_NAMES``);
- an ephemeral basic-auth password is generated per boot and only shared
  between the child and this process (the port file and every managed dir are
  created owner-only 0700 on POSIX, with a one-shot Windows ACL warning, so
  extra deepseaport workers can reuse the same instance without publishing
  the password to other local users);
- an explicitly user-configured ``DEEPSEAPORT_OPENCODE_BASE`` is respected:
  managed boot is skipped entirely and the user's serve is used as-is.

Never raises: every failure returns ``None`` and the caller falls back to
the historical serve-down 502 with a hint.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("deepseaport.server")

# Agent the managed instance defines via OPENCODE_CONFIG_CONTENT. Replaces
# the stock provider prompt (agent.prompt wins over providerDefault); the
# strip plugin below is the backstop for the remaining base layers.
MANAGED_AGENT = "deepseaport-passthrough"

# The managed agent's system prompt, and the ONLY system text the managed
# instance should carry: the strip plugin splices exactly this into
# `event.system` (see STRIP_PLUGIN_JS) and the config below sets it as the
# agent prompt, so the two layers cannot drift apart. It carries the DSML
# tool-call contract the harness (pi and friends) depends on.
PASSTHROUGH_SYSTEM = (
    "You are a plain chat model behind an OpenAI-compatible API. Answer the "
    "user message directly in text and never execute native tools yourself "
    "(no bash/read/edit/glob/grep/task calls, even when they share names with "
    "tools described in the user message). When the user message describes "
    "tools with a DSML tool_calls format, your ONLY way to act is to output the "
    "tool call in that exact DSML (or JSON tool_calls) format as text so the "
    "outer harness executes it. If no tool is needed, answer normally.")

_MANAGED_FLAG = "DEEPSEAPORT_OPENCODE_MANAGED"
_BASE_ENV = "DEEPSEAPORT_OPENCODE_BASE"
_PASSWORD_ENV = "DEEPSEAPORT_OPENCODE_PASSWORD"
_MANAGED_BASE_SENTINEL = "DEEPSEAPORT_OPENCODE_MANAGED_BASE"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_managed_base: str = ""
_managed_password: str = ""
_atexit_registered = False


def managed_enabled() -> bool:
    """False when the user opts out via DEEPSEAPORT_OPENCODE_MANAGED=0."""
    return os.environ.get(_MANAGED_FLAG, "1").strip().lower() not in (
        "0", "false", "no", "off")


def _explicit_user_base() -> str:
    """Non-empty BASE env that is not a base this module set itself."""
    raw = (os.environ.get(_BASE_ENV, "") or "").strip().rstrip("/")
    if raw and raw != _managed_base:
        return raw
    return ""


def managed_base_dir() -> Path:
    """Root dir for all managed-serve files (work + config + logs)."""
    from ...config import DATA_DIR

    return Path(DATA_DIR) / "opencode-managed"


STRIP_PLUGIN_JS = """// deepseaport-managed: replace the opencode system prompt with the managed
// agent's own passthrough instruction. Loaded only by the deepseaport-owned
// `opencode serve` child process (OPENCODE_CONFIG_DIR/plugins); the user's own
// opencode never sees this file.
//
// v2 plugin shape: a default-exported {id, setup(ctx)} module registering a
// session hook. The v1 `experimental.chat.system.transform` hook object no
// longer exists in opencode 2.x; `context` is the replacement (it runs for the
// agent loop just before dispatch). In-place mutation is required (reassignment
// is a silent no-op) and one entry must remain (empty restores the original
// upstream).
//
// It injects PASSTHROUGH_SYSTEM, the SAME text the managed config sets as the
// agent prompt: on v2 `event.system` carries the stock opencode prompt AND the
// agent's own prompt in one array, so stripping it to a generic one-liner would
// also erase the DSML tool-call contract the harness depends on.
export default {
  id: "deepseaport-strip-system",
  setup(ctx) {
    ctx.session.hook("context", (event) => {
      if (Array.isArray(event.system)) {
        event.system.splice(0, event.system.length, {
          type: "text",
          text: %s,
        });
      }
    });
  },
};
""" % json.dumps(PASSTHROUGH_SYSTEM, ensure_ascii=False)


def managed_config_content() -> str:
    """Inline opencode config (highest precedence) for the managed instance.

    opencode v2 accepts these keys and normalizes them (``agent`` to
    ``agents``, ``prompt`` to ``system``, ``bash`` to ``shell``); the
    permission map below is the ONLY thing that bounds what the model may
    call on v2, since v2 has no per-message tool map left to override it.
    bash is DENIED for the reason that map exists at all: a prompt-injection
    chain must not be able to run commands in a process that holds the
    operator's env (see _child_env) and cwd. The agent prompt forces
    text-only output so the harness (e.g. pi) owns execution; sessions are
    pinned to the caller's workdir via the create body's
    ``location.directory`` (see protocol.session_directory).
    """
    perms = {"read": "allow", "edit": "deny", "bash": "deny",
             "webfetch": "deny", "external_directory": "deny"}
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "instructions": [],
        "permission": dict(perms),
        "agent": {
            MANAGED_AGENT: {
                "mode": "primary",
                "description": "Raw passthrough for API use (managed by deepseaport)",
                "prompt": PASSTHROUGH_SYSTEM,
                "permission": dict(perms),
            }
        },
    }
    return json.dumps(cfg)


def _config_hash() -> str:
    """Short hash of everything baked into the serve at boot (never raises)."""
    import hashlib

    try:
        return hashlib.sha256(
            (managed_config_content() + "\n" + STRIP_PLUGIN_JS).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return ""


def write_isolated_files(base: Path) -> dict[str, Path]:
    """Create the managed dirs + strip plugin; return key paths.

    ``base`` holds: ``work/`` (serve cwd, kept empty so no AGENTS.md /
    project config is discovered), ``cfg/plugins/strip-system.js`` (loaded
    via OPENCODE_CONFIG_DIR), ``home/`` (XDG_CONFIG_HOME target, stays
    empty), ``serve.log`` and ``serve.json`` (port/password reuse file).
    Pure file work; never touches the real HOME. Never raises.

    ``base`` and every subdir are created 0700 on POSIX: ``serve.json`` holds
    the live basic-auth password in cleartext and mkdir()'s umask default is
    0755, which left it world-readable. Same helper the obscura cookie-jar
    dir uses (``core.obscura_bridge.secure_dir``), plus the one-shot Windows
    ACL warning, because chmod is a no-op there.
    """
    from ...core.obscura_bridge import secure_dir, warn_inherited_acl_once

    paths = {
        "work": base / "work",
        "cfg": base / "cfg",
        "home": base / "home",
        "log": base / "serve.log",
        "port_file": base / "serve.json",
    }
    try:
        for target in (base, paths["work"], paths["cfg"], paths["home"],
                       paths["cfg"] / "plugins"):
            secure_dir(target, "managed opencode dir")
    except Exception:
        return paths
    try:
        warn_inherited_acl_once(
            paths["port_file"],
            secret="the per-boot opencode basic-auth password it stores in "
                   "cleartext (and the base URL any local process can then use)",
            label="managed opencode port file")
    except Exception:
        pass
    try:
        plugin = paths["cfg"] / "plugins" / "strip-system.js"
        if (not plugin.exists()
                or plugin.read_text(encoding="utf-8") != STRIP_PLUGIN_JS):
            plugin.write_text(STRIP_PLUGIN_JS, encoding="utf-8")
    except Exception:
        pass
    # Best-effort git init: bare non-git dirs make some serve versions fall
    # back to `/` as the worktree; a repo keeps the instance rooted here.
    try:
        if shutil.which("git") and not (paths["work"] / ".git").exists():
            subprocess.run(["git", "init", "-q", str(paths["work"])],
                           capture_output=True, timeout=30)
    except Exception:
        pass
    return paths


def _probe(base: str, password: str, timeout: float = 2.0) -> bool:
    """True when ``GET /api/info`` answers 200 with a JSON body (never raises).

    The body check is load-bearing, not decoration: an unknown GET path on an
    opencode v2 serve falls back to the web-app index and answers 200 with
    HTML, so a status-only probe would call a v1 instance (or any other local
    server) healthy.
    """
    try:
        req = urllib.request.Request(base.rstrip("/") + "/api/info",
                                     method="GET",
                                     headers={"Accept": "application/json"})
        if password:
            token = base64.b64encode(
                f"opencode:{password}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if int(getattr(resp, "status", 0) or 0) != 200:
                return False
            raw = resp.read(4096).decode("utf-8", "replace")
        return isinstance(json.loads(raw), dict)
    except Exception:
        return False


def _read_port_file(port_file: Path) -> tuple[str, str, int, str]:
    """(base, password, serve_pid, config_hash); blanks when unreadable."""
    try:
        raw = port_file.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if isinstance(data, dict):
            try:
                pid = int(data.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            return (str(data.get("base") or "").strip().rstrip("/"),
                    str(data.get("password") or ""), pid,
                    str(data.get("config_hash") or ""))
    except Exception:
        pass
    return "", "", 0, ""


def _write_port_file(port_file: Path, base: str, password: str,
                     pid: int, config_hash: str) -> None:
    # 0600 is real on POSIX only; on Windows os.chmod merely toggles the
    # read-only attribute and the file keeps its directory's DACL, which is
    # why write_isolated_files also emits the one-shot ACL warning for this
    # file. The parent dir is 0700 on POSIX, so the mode here is belt-and-braces.
    try:
        port_file.write_text(json.dumps(
            {"base": base, "password": password, "pid": pid,
             "config_hash": config_hash}), encoding="utf-8")
        try:
            os.chmod(port_file, 0o600)
        except OSError:
            pass
    except Exception:
        pass


# Subprocess probe timeout for the command-line read in _process_command_line.
# The Windows path spawns a PowerShell (or wmic) child, so it is not free; it
# runs only on the stale-serve reap path (a config-hash change with an
# unhealthy recorded base), never per request.
_CMDLINE_PROBE_TIMEOUT = 5.0
# What the managed child's command line must LOOK like for the reaper to signal
# it. These are the executable and the subcommand, and they are matched as
# argv[0]'s basename and as a standalone argv element -- never as substrings
# of the rendered line. The substring form was no gate at all:
# `notepad.exe C:\opencode\serve.txt`, `powershell -c Write-Host opencode
# serve` and `calc.exe serve opencode` all satisfied it, so anything that
# could write serve.json could still aim the SIGTERM at a stranger.
_SERVE_BIN_NAME = "opencode"
_SERVE_SUBCOMMAND = "serve"
# The shell CPython hands a .cmd/.bat launcher to on Windows. The recorded pid
# is THIS process, not the serve, which is its child.
_CMD_WRAPPER_NAME = "cmd"


def _posix_command_line(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError, TypeError):
        return ""
    # NUL-separated argv; join so a substring check sees one line.
    return " ".join(p.decode("utf-8", "replace")
                    for p in raw.split(b"\0") if p)


def _windows_command_line(pid: int) -> str:
    """CIM query first (present on every current Windows), wmic as fallback.

    Neither is a dependency: a stripped-down image without PowerShell or
    wmic, a locked-down host, or a timeout all return "" and the caller
    refuses to signal the pid. Losing the reap is the acceptable outcome;
    signalling a stranger is not.
    """
    attempts = (
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine"],
        ["wmic", "process", "where", f"processid={int(pid)}", "get", "commandline"],
    )
    for argv in attempts:
        exe = shutil.which(argv[0])
        if not exe:
            continue
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            creationflags = 0
        try:
            done = subprocess.run(
                [exe, *argv[1:]], capture_output=True, text=True, errors="replace",
                timeout=_CMDLINE_PROBE_TIMEOUT, creationflags=creationflags)
        except Exception:
            continue
        if getattr(done, "returncode", 1) != 0:
            continue
        out = str(getattr(done, "stdout", "") or "")
        if not out.strip():
            continue
        if out.lstrip().lower().startswith("commandline"):
            # wmic prints a header row above the value.
            out = out.split("\n", 1)[1] if "\n" in out else ""
        return " ".join(out.split())
    return ""


def _process_command_line(pid: int) -> str:
    """Command line of ``pid`` as one string; "" when it cannot be read.

    Never raises: the only caller is the reaper, which must not be able to
    fail a request (or guess).
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return ""
    if pid <= 0:
        return ""
    try:
        return (_windows_command_line(pid) if os.name == "nt"
                else _posix_command_line(pid))
    except Exception:
        return ""


def _split_argv(cmdline: str) -> list[str]:
    """Command line -> argv list; ``[]`` when it cannot be parsed (never raises).

    Two quoting dialects, and picking the wrong one is a security bug, not a
    cosmetic one: POSIX ``shlex`` mode eats backslashes, so
    ``C:\\oc\\opencode.exe`` would come back as ``C:ocopencode.exe``. On
    Windows the non-POSIX splitter runs instead and the quotes it leaves
    behind are stripped (``shlex`` has no CommandLineToArgvW backslash rule --
    irrelevant here because only argv[0]'s basename, the ``serve`` element and
    ``--port`` are read). A line with an unbalanced quote raises inside
    ``shlex`` and yields ``[]``, which the caller reads as "not identifiable".

    The reader above already collapsed ``/proc/<pid>/cmdline`` into one
    string, so this re-splits it. The only lossy case is a POSIX argv element
    that itself contains a space and no quotes (``a b serve``); the identity
    check does not depend on it and a quoted path round-trips exactly.
    """
    try:
        raw = " ".join(str(cmdline or "").split())  # fold the reader's newlines
        if not raw:
            return []
        if os.name == "nt":
            tokens = shlex.split(raw, posix=False)
            return [t for t in (str(tok).strip().strip('"').strip("'")
                               for tok in tokens) if t]
        return shlex.split(raw)
    except Exception:
        return []


def _exe_basename(command: str) -> str:
    """``opencode.exe`` -> ``opencode``; quotes, separators and case folded."""
    try:
        raw = str(command or "").strip().strip('"').strip("'")
        name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if name.lower().endswith(".exe"):
            name = name[:-4]
        return name.strip().lower()
    except Exception:
        return ""


def _serve_binary_names() -> set[str]:
    """Acceptable argv[0] basenames: the literal name plus what PATH resolves.

    A recorded binary that is not the one on PATH is still the same program
    (two opencode installs, a moved shim), so a name mismatch is never
    disqualifying on its own -- only a non-``opencode`` name is.
    """
    names = {_SERVE_BIN_NAME}
    try:
        resolved = shutil.which(_SERVE_BIN_NAME)
    except Exception:
        resolved = ""
    if resolved:
        base = _exe_basename(resolved)
        if base:
            names.add(base)
    return names


# Suffixes a launcher may carry: the real binary is ``.exe``, and an npm /
# bun install puts a ``.cmd`` / ``.bat`` shim next to it. The identity check
# compares STEMS so one rule covers every shape.
_LAUNCHER_SUFFIXES = (".exe", ".cmd", ".bat")


def _exe_stem(command: str) -> str:
    """``opencode.exe`` / ``opencode.CMD`` -> ``opencode``; case folded."""
    try:
        name = _exe_basename(command)
        for suffix in _LAUNCHER_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)].strip()
        return name
    except Exception:
        return ""


def _serve_binary_stems() -> set[str]:
    """Launchable stems for the serve: the literal name plus what PATH resolves."""
    return {_SERVE_BIN_NAME} | {
        stem for stem in (_exe_stem(name)
                          for name in _serve_binary_names()) if stem}


def _argv_port(argv: list[str]) -> int:
    """``--port N`` / ``--port=N`` in argv; 0 when absent or unreadable."""
    try:
        for idx, tok in enumerate(argv):
            if tok == "--port" and idx + 1 < len(argv):
                return int(str(argv[idx + 1]).strip())
            if tok.startswith("--port="):
                return int(tok.split("=", 1)[1].strip())
    except Exception:
        return 0
    return 0


def _base_port(base: str) -> int:
    """Port of a base URL; 0 when it carries none."""
    try:
        return int(urllib.parse.urlsplit(str(base or "").strip()).port or 0)
    except Exception:
        return 0


def _looks_like_managed_serve(cmdline: str, expected_base: str = "") -> bool:
    """True only when a command line IS the managed ``opencode serve``.

    An identity check, not a resemblance check, because the pid this gates
    comes from an ordinary JSON file in the data dir:

    * The launcher must BE opencode, case-folded and suffix-insensitive
      (Windows: ``opencode.exe``, ``opencode.CMD`` and a bare ``opencode`` are
      the same stem). ``shutil.which`` is consulted when it resolves, so an
      install whose binary carries a different name still matches. Anything
      else is a stranger: notepad, powershell, calc, a file named
      ``serve.exe``.
    * OR the Windows shell wrapper that launches it: CPython hands a batch
      file to the OS shell, so ``Popen([opencode.CMD, ...])`` records
      ``cmd.exe /c <path>\\opencode.CMD serve ...`` and the pid in the port
      file is the WRAPPER's, with the serve as its child. Only that exact
      three-token prefix is accepted (``cmd`` ``/c`` ``<opencode>``), so
      ``cmd.exe /c start opencode serve`` still fails on the ``start`` in
      argv[2] -- a wrapper that merely mentions opencode later in the line is
      not the launcher we spawned.
    * ``serve`` must be a real argv element of the launcher, not a word
      somewhere in the line. The line is split into argv first (POSIX quoting
      on POSIX, CommandLineToArgvW-style quoting on Windows) so a quoted path
      and a path-less invocation both parse.
    * ``--port N``, when present, must agree with the base URL recorded
      alongside the pid. A serve on another port is not the instance whose
      pid was recorded.

    Returns False on ANY failure to read or parse: an unreadable command
    line, an unbalanced quote, an empty argv. The caller then skips the
    signal (see :func:`_kill_pid`), so the cost is a stale process and a busy
    port -- ``ensure()`` boots a fresh instance on a new free port anyway --
    while signalling a stranger is not recoverable. Never raises.
    """
    try:
        argv = _split_argv(cmdline)
        if len(argv) < 2:
            return False
        stems = _serve_binary_stems()
        if _exe_stem(argv[0]) in stems:
            rest = argv[1:]
        elif (_exe_stem(argv[0]) == _CMD_WRAPPER_NAME and len(argv) >= 4
              and argv[1].strip().strip('"').lower() == "/c"
              and _exe_stem(argv[2]) in stems):
            rest = argv[3:]
        else:
            return False
        if not any(tok.lower() == _SERVE_SUBCOMMAND for tok in rest):
            return False
        port = _argv_port(argv)
        if port and expected_base:
            want = _base_port(expected_base)
            if want and want != port:
                logger.warning(
                    "managed serve reaper: recorded base %s is port %d but the "
                    "pid's command line says --port %d; not signalling it.",
                    expected_base, want, port)
                return False
        return True
    except Exception:  # pragma: no cover - defensive, contract is "never raises"
        return False


def _kill_pid(pid: int) -> None:
    """Best-effort SIGTERM of a stale managed serve (never raises).

    The pid comes out of ``data/opencode-managed/serve.json`` -- an ordinary
    file in the data dir -- and the only gate used to be a probe against a
    base URL stored in that SAME file, so anything that could write that
    file (another same-user process, a sync conflict, a stale copy of a
    checkout) could aim the signal at an arbitrary process of this user on
    the next deepseaport start.

    So the target is identified before it is signalled: its command line
    must be the ``opencode serve`` we boot -- the opencode binary as argv[0]
    with ``serve`` as a real argument (see
    :func:`_looks_like_managed_serve`), and its ``--port`` must agree with the
    base URL recorded next to the pid in the SAME serve.json. Anything else --
    a different program, a recycled pid, an unreadable or unparseable command
    line -- is skipped, and the reap simply does not happen: ``ensure()`` boots
    a fresh instance on a new free port anyway, so the cost is a stale process
    and a busy port, never a stranger's process.
    """
    if not pid or pid == os.getpid():
        return
    try:
        cmdline = _process_command_line(pid)
    except Exception:
        cmdline = ""
    # The base recorded alongside the pid: an extra "this is the instance we
    # recorded" check. Read here rather than passed in so the pid and the base
    # can never come from different snapshots, and so an unreadable/missing
    # port file simply skips the cross-check instead of blocking a legitimate
    # reap.
    try:
        expected_base = _read_port_file(managed_base_dir() / "serve.json")[0]
    except Exception:
        expected_base = ""
    if not _looks_like_managed_serve(cmdline, expected_base):
        logger.warning(
            "managed serve reaper: pid %s does not look like 'opencode "
            "serve' (command line %s); not signalling it. The stale serve is "
            "left running and a fresh one boots on a new port -- delete %s "
            "if that persists.", pid, "unreadable" if not cmdline else "different",
            managed_base_dir() / "serve.json")
        return
    if os.name == "nt":
        # The recorded pid is the cmd.exe wrapper, and the serve is its child:
        # signalling the wrapper alone would leave the real serve running on
        # the port forever. /T reaches it (see :func:`_kill_tree`).
        _kill_tree(pid)
        return
    try:
        import signal

        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.2)
    except Exception:
        pass


def _kill_tree(pid: int) -> bool:
    """Windows: kill ``pid`` AND every process it started. True if it ran.

    ``taskkill /T`` is the only way to reach the grandchildren from stdlib:
    the managed child is spawned through the npm ``.cmd`` shim, so Windows
    records a ``cmd.exe /c opencode.CMD serve ...`` wrapper whose child is the
    actual serve. Terminating the wrapper (what ``Popen.terminate`` does)
    leaves that child alive and still holding its port, which is how every
    deepseaport restart leaked a serve before this existed.

    ``/F`` because Windows has no graceful signal for a console child, and
    ``proc.terminate()`` was already an immediate TerminateProcess there. Off
    Windows this is a no-op: SIGTERM to the process group is what POSIX wants.
    Never raises; a failure here only costs a stale serve on a port that
    ``ensure()`` probes around anyway.
    """
    if os.name != "nt":
        return False
    try:
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            creationflags = 0
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True, text=True, timeout=15,
            creationflags=creationflags)
        return True
    except Exception:
        return False


# Grace between terminate() and the kill() escalation. shutdown() already
# waited this long for a SIGTERM to land; the reap helpers now share it.
_TERMINATE_WAIT = 5.0


def _terminate_proc(proc, timeout: float = _TERMINATE_WAIT) -> None:
    """terminate() a managed child, then wait() for it; escalate to kill().

    ``wait()`` is what actually reaps, so every managed-child exit goes
    through here. A bare ``terminate()`` leaves the child unreaped -- a POSIX
    zombie until the garbage collector gets round to the ``Popen`` (where its
    table entry is still busy) and, on Windows, an unowned live process
    holding its port. On ensure()'s not-ready path nothing else would ever
    wait on that child at all: ``_proc`` is not assigned before it returns.

    On Windows the handle we hold is the ``cmd.exe`` wrapper, so terminate()
    kills the wrapper and leaves the serve itself running on its port. The
    tree kill goes first there instead (:func:`_kill_tree`); the ``wait()`` that
    follows is still what reaps the handle.

    A child that ignores SIGTERM is killed rather than left running: an
    orphan ``opencode serve`` is a hole in the isolation the managed instance
    exists to provide. Never raises.
    """
    try:
        alive = proc.poll() is None
    except Exception:
        alive = True
    if alive and os.name == "nt":
        _kill_tree(int(getattr(proc, "pid", 0) or 0))
    elif alive:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        return
    try:
        proc.wait(timeout=timeout)
    except Exception:
        pass


def _reap_previous_child(recorded_pid: int = 0) -> None:
    """Retire the managed child a fresh boot is about to replace.

    ``_proc`` is the only handle ``shutdown()`` and the ``atexit`` hook know,
    so the ``_proc = proc`` assignment at the end of :func:`ensure` makes the
    previous child unreachable the moment a new instance boots. The scenario
    is not exotic: the reuse probe is a 2 s ``GET /api/info``, so an
    *unresponsive* -- not dead -- serve (a loaded box, a wedged agent loop)
    misses it, and ensure() boots a fresh instance on a new free port. The
    orphan then keeps its port, its ``work/`` cwd and its per-boot basic-auth
    password for the life of the bridge, serving an instance (with the
    managed agent's ``bash: "deny"`` map) that nothing can reach or shut down
    again -- and ``ensure()`` runs from every serve-down error path
    (completion_service.py:202, 288), so they accumulated.

    The tracked ``_proc`` is reaped by handle: we spawned it, so it cannot be
    a stranger and the identity gate is not needed. Only when the handle is
    gone (module reloaded, the port file written by another process) does the
    recorded pid go through the gated :func:`_kill_pid`, which refuses to
    signal anything that does not look like the managed serve. Never raises.
    """
    global _proc
    proc, _proc = _proc, None
    if proc is not None:
        _terminate_proc(proc)
        return
    if recorded_pid:
        _kill_pid(recorded_pid)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# Env prefixes stripped from the managed child. The child is an agentic CLI
# with native read/bash rooted at the operator's cwd, and .env.example
# documents DEEPSEAPORT_KEYS / DEEPSEEK_EMAIL / DEEPSEEK_PASSWORD /
# DEEPSEEK_TOKEN — a sourced .env therefore lands verbatim in a process the
# model can be talked into reading (or echoing) via a prompt-injection chain
# in a user message. The agent prompt is a prompt-level control only; the env
# is a real boundary. Everything the child legitimately needs is set
# explicitly in ensure() right after this strip.
_CHILD_ENV_STRIP_PREFIXES = ("DEEPSEAPORT_", "DEEPSEEK_", "OPENCODE_")

# ...and the well-known THIRD-PARTY credentials the prefixes do not cover. The
# operator's own provider keys (Anthropic/OpenAI/GitHub/AWS/Azure/Google) are
# worth the same protection as deepseaport's: today the exploit path is closed
# (bash is denied, external_directory is denied, the agent prompt forces text
# output), but this strip exists for exactly the day a native tool is granted
# again, and a leaked ANTHROPIC_API_KEY is a bill someone else pays.
#
# Deny-shaped on purpose. A prefix rule fails open for every provider that
# ships after this list was written, and nothing in the codebase would notice;
# an explicit name has to be consciously added by whoever adds the key. The
# trade-off is a false negative, which costs nothing here, over a false
# positive, which would silently break a child that needs the variable. Nothing
# the child legitimately needs may be listed: the isolation keys and the
# per-boot password are set explicitly in ensure() AFTER this strip, and
# ``_CHILD_ENV_STRIP_PREFIXES`` already covers the OPENCODE_* family.
_CHILD_ENV_DENY_NAMES = frozenset({
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
})


def _child_env(source: dict | None = None) -> dict:
    """Copy the parent env with every secret-bearing deepseaport/opencode key
    removed. Names only are ever logged; values are never touched elsewhere."""
    env = dict(os.environ if source is None else source)
    dropped = 0
    for key in [k for k in env
                if str(k).upper().startswith(_CHILD_ENV_STRIP_PREFIXES)
                or str(k).upper() in _CHILD_ENV_DENY_NAMES]:
        env.pop(key, None)
        dropped += 1
    if dropped:
        try:
            logger.debug("managed serve child env: dropped %d "
                         "DEEPSEAPORT_/DEEPSEEK_/OPENCODE_* or known "
                         "third-party credential key(s)", dropped)
        except Exception:
            pass
    return env


def ensure(timeout: float = 30.0) -> str | None:
    """Boot (or reuse) the managed serve; return its base URL or None.

    No-op when managed mode is disabled or the user configured their own
    ``DEEPSEAPORT_OPENCODE_BASE``. On success sets ``DEEPSEAPORT_OPENCODE_BASE``
    (+ ``_PASSWORD`` + sentinel) in this process so ``protocol.base_url()``
    picks it up. Thread-safe; never raises.
    """
    global _proc, _managed_base, _managed_password, _atexit_registered
    try:
        if not managed_enabled():
            return None
        if _explicit_user_base():
            return None
        with _lock:
            base_dir = managed_base_dir()
            paths = write_isolated_files(base_dir)
            wanted_hash = _config_hash()
            if (_managed_base and _managed_password
                    and _probe(_managed_base, _managed_password)):
                return _managed_base
            old_base, old_password, old_pid, old_hash = _read_port_file(
                paths["port_file"])
            # Whether the recorded pid is already retired below, so the
            # pre-boot reap does not signal the same pid a second time (which
            # would fail the identity gate on a now-dead process and log a
            # misleading "not signalling it" warning).
            old_pid_reaped = False
            if old_base and old_password and _probe(old_base, old_password):
                if not wanted_hash or old_hash == wanted_hash:
                    _managed_base, _managed_password = old_base, old_password
                    os.environ[_BASE_ENV] = old_base
                    os.environ[_PASSWORD_ENV] = old_password
                    os.environ[_MANAGED_BASE_SENTINEL] = old_base
                    return old_base
                logger.info("managed serve config changed, restarting it")
                _kill_pid(old_pid)
                old_pid_reaped = True
            binary = shutil.which("opencode")
            if not binary:
                logger.warning("opencode binary not found on PATH; "
                               "install opencode for the accountless provider")
                return None
            cli_version = ""
            try:
                proc_ver = subprocess.run(
                    [binary, "--version"], capture_output=True, text=True,
                    timeout=10)
                cli_version = (((proc_ver.stdout or "") + " " +
                                (proc_ver.stderr or "")).strip()[:80])
            except Exception:
                pass
            logger.info("managed serve binary: %s (log %s)",
                        cli_version or "version unknown", paths["log"])
            # Past this point a new child is definitely starting, so retire
            # whatever is running first: replacing `_proc` orphans the old
            # serve for the life of the process (see _reap_previous_child).
            # Deliberately after the binary check -- with no binary there is
            # no replacement to install and the previous child stays.
            _reap_previous_child(0 if old_pid_reaped else old_pid)
            port = _find_free_port()
            password = secrets.token_urlsafe(32)
            base = f"http://127.0.0.1:{port}"
            env = _child_env()
            # Explicit child-only env, set AFTER the strip above so these
            # survive it (XDG/OPENCODE isolation keys + the per-boot password).
            env["XDG_CONFIG_HOME"] = str(paths["home"])
            env["OPENCODE_CONFIG_DIR"] = str(paths["cfg"])
            env["OPENCODE_CONFIG_CONTENT"] = managed_config_content()
            # The parent's OPENCODE_CONFIG path was already dropped by the
            # strip above (config merges additively, so it would otherwise leak
            # user instructions into the managed instance).
            env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
            env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
            env["OPENCODE_SERVER_PASSWORD"] = password
            try:
                log_fh = open(paths["log"], "a", encoding="utf-8")
            except Exception:
                log_fh = None  # type: ignore[assignment]
            creationflags = 0
            try:
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            except Exception:
                creationflags = 0
            try:
                proc = subprocess.Popen(
                    [binary, "serve", "--port", str(port),
                     "--hostname", "127.0.0.1"],
                    cwd=str(paths["work"]),
                    env=env,
                    stdout=log_fh or subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags)
            except Exception as exc:
                logger.warning("managed opencode serve failed to start: %s", exc)
                try:
                    if log_fh is not None:
                        log_fh.close()
                except Exception:
                    pass
                return None
            deadline = time.monotonic() + max(5.0, timeout)
            ready = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    logger.warning("managed opencode serve exited early (code %s)",
                                   proc.returncode)
                    break
                if _probe(base, password):
                    ready = True
                    break
                time.sleep(0.25)
            try:
                if log_fh is not None:
                    log_fh.close()
            except Exception:
                pass
            if not ready:
                # Never adopted (`_proc` is not assigned on this path), so
                # terminate + wait here or the child stays unreaped forever.
                _terminate_proc(proc)
                return None
            _proc = proc
            _managed_base, _managed_password = base, password
            os.environ[_BASE_ENV] = base
            os.environ[_PASSWORD_ENV] = password
            os.environ[_MANAGED_BASE_SENTINEL] = base
            _write_port_file(paths["port_file"], base, password,
                             int(getattr(proc, "pid", 0) or 0), wanted_hash)
            if not _atexit_registered:
                try:
                    atexit.register(shutdown)
                    _atexit_registered = True
                except Exception:
                    pass
            logger.info("managed opencode serve ready at %s (opencode %s)",
                        base, cli_version or "unknown version")
            return base
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed opencode serve ignored: %s", exc)
        return None


def shutdown() -> None:
    """Terminate the managed serve child (best-effort, never raises)."""
    global _proc
    try:
        proc, _proc = _proc, None
        if proc is not None:
            _terminate_proc(proc)
    except Exception:
        pass
