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
  into the managed instance);
- an ephemeral basic-auth password is generated per boot and only shared
  between the child and this process (plus a ``0o600`` port file so extra
  deepseaport workers can reuse the same instance);
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
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("deepseaport.server")

# Agent the managed instance defines via OPENCODE_CONFIG_CONTENT. Replaces
# the stock provider prompt (agent.prompt wins over providerDefault); the
# strip plugin below is the backstop for the remaining base layers.
MANAGED_AGENT = "deepseaport-passthrough"

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


STRIP_PLUGIN_JS = """// deepseaport-managed: collapse the opencode system prompt to one line.
// Loaded only by the deepseaport-owned `opencode serve` child process
// (OPENCODE_CONFIG_DIR); the user's own opencode never sees this file.
export const DeepSeaportStripSystem = async () => ({
  "experimental.chat.system.transform": async (_input, output) => {
    // In-place mutation is required (reassignment is a silent no-op) and
    // one entry must remain (empty restores the original upstream).
    output.system.splice(
      0,
      output.system.length,
      "You are a concise assistant. Follow the user message directly.",
    );
  },
});
"""


def managed_config_content() -> str:
    """Inline opencode config (highest precedence) for the managed instance."""
    # read+bash stay allowed (declared, not used): the free-tier gatekeeper
    # needs them effective upstream (an explicit all-false tool list 403s
    # with "only be used from within OpenCode"; omitting the per-message
    # key passes via these agent defaults on opencode 1.18.32). The agent
    # prompt forces text-only output so the harness (e.g. pi) owns
    # execution; sessions are pinned to the caller's workdir via
    # `?directory=` (see protocol.session_directory) so any accidental
    # native tool still runs where the harness expects.
    perms = {"read": "allow", "edit": "deny", "bash": "allow",
             "webfetch": "deny", "external_directory": "deny"}
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "instructions": [],
        "permission": dict(perms),
        "agent": {
            MANAGED_AGENT: {
                "mode": "primary",
                "description": "Raw passthrough for API use (managed by deepseaport)",
                "prompt": ("You are a plain chat model behind an OpenAI-compatible "
                           "API. Answer the user message directly in text and "
                           "never execute native tools yourself (no bash/read/"
                           "edit/glob/grep/task calls, even when they share "
                           "names with tools described in the user message). "
                           "When the user message describes tools with a DSML "
                           "tool_calls format, your ONLY way to act is to output "
                           "the tool call in that exact DSML (or JSON tool_calls) "
                           "format as text so the outer harness executes it. "
                           "If no tool is needed, answer normally."),
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
    """
    paths = {
        "work": base / "work",
        "cfg": base / "cfg",
        "home": base / "home",
        "log": base / "serve.log",
        "port_file": base / "serve.json",
    }
    try:
        paths["work"].mkdir(parents=True, exist_ok=True)
        (paths["cfg"] / "plugins").mkdir(parents=True, exist_ok=True)
        paths["home"].mkdir(parents=True, exist_ok=True)
    except Exception:
        return paths
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
    """True when /global/health answers 200 (never raises)."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/global/health",
                                     method="GET",
                                     headers={"Accept": "application/json"})
        if password:
            token = base64.b64encode(
                f"opencode:{password}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 0) or 0) == 200
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


def _kill_pid(pid: int) -> None:
    """Best-effort SIGTERM of a stale managed serve (never raises)."""
    if not pid or pid == os.getpid():
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
            if old_base and old_password and _probe(old_base, old_password):
                if not wanted_hash or old_hash == wanted_hash:
                    _managed_base, _managed_password = old_base, old_password
                    os.environ[_BASE_ENV] = old_base
                    os.environ[_PASSWORD_ENV] = old_password
                    os.environ[_MANAGED_BASE_SENTINEL] = old_base
                    return old_base
                logger.info("managed serve config changed, restarting it")
                _kill_pid(old_pid)
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
            port = _find_free_port()
            password = secrets.token_urlsafe(32)
            base = f"http://127.0.0.1:{port}"
            env = dict(os.environ)
            env["XDG_CONFIG_HOME"] = str(paths["home"])
            env["OPENCODE_CONFIG_DIR"] = str(paths["cfg"])
            env["OPENCODE_CONFIG_CONTENT"] = managed_config_content()
            env.pop("OPENCODE_CONFIG", None)
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
                try:
                    proc.terminate()
                except Exception:
                    pass
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
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
