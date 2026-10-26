"""Obscura bridge: WAF-cleared cookies + browser UA for the hot path.

chat.deepseek.com sits behind an AWS WAF JavaScript challenge. Obscura
(h4ckf0r0day/obscura) executes it in its V8 engine and persists cookies in a
profile dir. This module warms the profile (challenge then real-app fetch),
dumps the jar (`--dump cookies`, HttpOnly included), captures the engine's
User-Agent for header consistency, and exposes the jar as a `Cookie` header
for direct `curl_cffi` calls. Direct HTTP (not DOM driving) stays the hot
path; Obscura is re-invoked only to refresh the WAF token (startup + on 403).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Skip a global WAF refresh when the last successful warmup is this fresh.
# WAF tokens live minutes-hours; re-solving on every 403 in a thundering herd
# just serializes all in-flight requests on the profile lock.
WARMUP_SKIP_SECONDS = 60.0

logger = logging.getLogger("zourcewell.obscura")

CHAT_HOME = os.environ.get("ZOURCEWELL_CHAT_HOME", "https://chat.deepseek.com/").strip() or "https://chat.deepseek.com/"


def _repo_root() -> Path:
    # Same base as config.json: repo root in a checkout, per-user dir for an
    # installed wheel, exe location when frozen -- see config._app_dir(), which
    # this used to duplicate as Path(__file__).parents[3] (the interpreter's
    # Lib/ for a wheel). Imported inside the function because this module is
    # imported from config's neighbourhood: config imports only core.tokens
    # (stdlib-only) and core/__init__.py is empty, so there is no cycle.
    from ..config import _app_dir

    return _app_dir()


REPO_ROOT = _repo_root()
# No absolute/host-specific paths here: the repo is published to GitHub and
# everyone checks out to a different folder. Probes repo-local dirs for a
# dropped-in binary (e.g. extracted release folder
# `obscura-x86_64-windows-no-render/obscura.exe` next to config.json).
# Glob patterns support versioned release dirs.
OBSCURA_LOCAL_PATTERNS = (
    "obscura.exe",
    "obscura",
    "bin/obscura.exe",
    "bin/obscura",
    "tools/obscura.exe",
    "tools/obscura",
    "vendor/obscura.exe",
    "vendor/obscura",
    "obscura-*/obscura.exe",
    "obscura-*/obscura",
)

log = logger


@dataclass
class WafState:
    cookies: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    warmed: bool = False


# One lock per resolved --storage-dir, shared across every ObscuraBridge in
# this process. Distinct bridge instances (app bridge, CLI/TUI login, token
# refresh) may point at the same profile dir; a per-instance lock would let
# their obscura subprocesses race on the cookie jar and corrupt it. Keyed by
# resolved (case-normalized on Windows) path so aliases map to one lock.
_PROFILE_LOCKS: dict[str, threading.RLock] = {}
_PROFILE_LOCKS_GUARD = threading.Lock()


def _profile_key(profile: str) -> str:
    try:
        path = Path(profile).expanduser().resolve()
    except Exception:
        path = Path(profile)
    raw = str(path)
    return os.path.normcase(raw) if os.name == "nt" else raw


# Owner-only mode for any directory that holds credentials: the Obscura
# profile dir (the aws-waf-token cookie jar, a live solved WAF session) and
# the managed opencode dir (serve.json holds the per-boot basic-auth
# password). mkdir() used the process umask, i.e. 0755 on a default POSIX
# umask, leaving those files world-readable. Windows has no POSIX modes (the
# attribute is a no-op), so the chmod is POSIX-only -- see
# ``warn_inherited_acl_once`` for the Windows half of the same problem.
_DIR_MODE = 0o700


def secure_dir(path, label: str = "") -> None:
    """Create ``path`` (with parents) and tighten its mode to 0700.

    Shared by every credential-bearing dir in the app so the POSIX half of the
    "don't leave secrets world-readable" rule is stated once, and so a
    Windows caller has one place to attach the ACL warning
    (:func:`warn_inherited_acl_once`).

    Never raises: a locked/undeletable dir must not stop the caller (the
    bridge's WinError5 handling downstream relies on that).
    """
    name = str(label or path)
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("%s mkdir failed for %s: %s", name, path, exc)
        return
    if os.name == "nt":
        return
    try:
        os.chmod(target, _DIR_MODE)
    except Exception as exc:
        # Read-only/foreign filesystem: the dir still works, only the mode
        # stays wider than intended. Never fatal.
        logger.debug("%s chmod 0700 skipped for %s: %s", name, path, exc)


# Windows half of ``secure_dir``: os.chmod on win32 only toggles the read-only
# attribute and never touches the DACL, so a "protected" file keeps whatever
# the containing directory grants (on a default install that includes read
# access for ``BUILTIN\Users``). Warning, not a fix: rewriting the ACL behind
# the operator's back (``icacls``, a deny ACE) can lock them out of their own
# credential store, and where the file lives is their call. Mirrors
# ``config._warn_inherited_acl_once`` (config.py owns that one and imports
# nothing from here, so this copy keeps the modules cycle-free). The once flag
# is keyed by ``label``, not a single module boolean, so each secret store
# gets its own warning and neither consumes the other's.
_ACL_WARNED_LABELS: set[str] = set()


def warn_inherited_acl_once(target, secret: str, label: str = "credential file",
                            platform: str | None = None) -> None:
    """On Windows, warn ONCE per ``label`` that ``target`` inherits its dir ACL.

    Same wording contract as the config.json warning (a log grep for
    "inherits the ACL" finds every one of them): it names the file, the secret
    at risk, the fact that the POSIX mode does not apply here, and the
    mitigation. No-op on POSIX, where the 0700/0600 modes are real.

    ``platform`` exists only so the branch is testable on both platforms:
    faking ``os.name`` process-wide breaks ``pathlib`` (it picks PosixPath,
    which cannot be instantiated on Windows). It defaults to the real one.
    """
    if (os.name if platform is None else platform) != "nt":
        return
    key = str(label or "credential file")
    if key in _ACL_WARNED_LABELS:
        return
    _ACL_WARNED_LABELS.add(key)
    try:
        logger.warning(
            "%s %s inherits the ACL of its directory: %s is readable by every "
            "other local user that directory grants access to. POSIX 0600/0700 "
            "does NOT apply on Windows -- os.chmod only toggles the read-only "
            "attribute and never touches the DACL. Mitigation: keep it in a "
            "directory only your own user can read, or point ZOURCEWELL_DATA "
            "at one (a per-user data dir, not a shared repo checkout).",
            key, target, secret)
    except Exception:  # pragma: no cover - defensive
        pass


def profile_lock(profile: str) -> threading.RLock:
    """Return the shared reentrant lock serializing obscura on `profile`.

    Callers that spawn obscura/MCP subprocesses against a profile dir must
    hold this for the subprocess lifetime: two obscura processes on one
    --storage-dir race on the cookie jar and corrupt it.
    """
    key = _profile_key(profile)
    with _PROFILE_LOCKS_GUARD:
        lock = _PROFILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROFILE_LOCKS[key] = lock
        return lock


# Worst case for a single warmup leader: it holds _proc_lock across pass 1
# (``_run(timeout=timeout)``) and then, only if pass 1 produced no cookies, over
# ``dump_cookies(timeout=30)`` and ``fetch_ua(timeout=30)``. A follower that gave
# up before that returned the PRE-refresh jar, sent it, and 403'd again --
# burning another full recovery pass, which reads to the user as "the WAF fix
# does not work". So the follower waits for the leader's own bound (plus slack
# for process startup) and re-checks the leader's completion flag before it
# hands back cookies: an expired wait is still a stale jar, but a rare one we
# log instead of one that fires on every slow refresh.
WARMUP_SUBSTEP_TIMEOUT = 30.0  # dump_cookies / fetch_ua defaults
WARMUP_FOLLOWER_SLACK = 15.0  # obscura startup + parse headroom


def _follower_wait_budget(timeout: int) -> float:
    """Seconds a singleflight follower waits for the leader's refresh."""
    return max(1.0, float(timeout) + 2 * WARMUP_SUBSTEP_TIMEOUT + WARMUP_FOLLOWER_SLACK)


class ObscuraBridge:
    def __init__(self, binary: str = "", profile: str = "") -> None:
        self.binary = self._resolve_binary(binary)
        default_profile = str(Path.home() / ".zourcewell" / "obscura-profile")
        self.profile = profile or default_profile
        secure_dir(self.profile, "obscura profile")
        self.state = WafState()
        # All obscura subprocesses share one --storage-dir. Parallel warmups
        # (multi-account 403 refreshes) would race on it and corrupt the
        # profile/cookie jar, so cold refreshes serialize. The hot path
        # (cookie_header/has_waf_token) never takes this lock. RLock because
        # warmup() calls dump_cookies()/fetch_ua() which also lock. Shared
        # per-profile across instances (see profile_lock) so token-refresh /
        # CLI login subprocesses cannot race the app bridge on the same dir.
        self._proc_lock = profile_lock(self.profile)
        # Singleflight for global refreshes: one leader runs the subprocess
        # while concurrent callers wait on the same event instead of queueing
        # N serial warmups. _last_warmup_ok gates the <60s skip.
        self._warm_lock = threading.Lock()
        self._warm_inflight: threading.Event | None = None
        self._last_warmup_ok: float = 0.0

    @staticmethod
    def _resolve_binary(binary: str = "") -> str:
        """Resolve an explicit binary path, tolerating a directory.

        A pasted release folder (e.g. ``obscura-x86_64-windows-no-render``)
        used to reach ``subprocess`` verbatim and fail as ``[WinError 5]
        Odmowa dostępu`` with 0 cookies and an empty UA, which then starved
        every later request of its WAF token. A directory now resolves to
        the ``obscura.exe``/``obscura`` inside it.
        """
        raw = (binary or "").strip()
        if not raw:
            return ObscuraBridge._discover()
        try:
            cand = Path(raw).expanduser()
        except Exception:
            return raw
        try:
            if cand.is_dir():
                for name in ("obscura.exe", "obscura"):
                    hit = cand / name
                    try:
                        if hit.is_file():
                            return str(hit)
                    except Exception:
                        continue
        except Exception:
            pass
        return raw

    @staticmethod
    def _search_roots() -> list[Path]:
        """Dirs to probe for a repo-local binary: cwd first, then repo root."""
        roots: list[Path] = []
        for raw in (Path.cwd(), REPO_ROOT):
            try:
                resolved = raw.resolve()
            except Exception:
                continue
            if resolved not in roots:
                roots.append(resolved)
        return roots

    @staticmethod
    def _discover() -> str:
        found = shutil.which("obscura")
        if found:
            return found
        for base in ObscuraBridge._search_roots():
            for pat in OBSCURA_LOCAL_PATTERNS:
                try:
                    hits = sorted(base.glob(pat))
                except Exception:
                    continue
                for hit in hits:
                    if hit.is_file():
                        return str(hit)
        return "obscura"  # hope for PATH at call time

    def _run(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        cmd = [self.binary, "--storage-dir", self.profile, *args]
        log.debug("obscura: %s", " ".join(cmd[1:]))
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def version(self) -> str:
        try:
            proc = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=15)
            return (proc.stdout or proc.stderr).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    def _parse_cookie_dump(self, payload: str) -> dict[str, str]:
        """Extract {name: value} from obscura --dump cookies stdout."""
        payload = payload or ""
        start = payload.find("[")
        try:
            items = json.loads(payload[start:] if start >= 0 else payload or "[]")
            return {c["name"]: c["value"] for c in items if c.get("name") and c.get("value")}
        except Exception:
            return {}

    def warmup(self, timeout: int = 45) -> dict[str, str]:
        """Solve/refresh the WAF challenge; return cookie dict (may be empty).

        Singleflight: concurrent callers share one execution; a warmup that
        succeeded <WARMUP_SKIP_SECONDS ago is skipped (cached cookies
        returned). Subprocess work still serializes on _proc_lock.
        """
        # Fast skip + singleflight join without holding _proc_lock.
        with self._warm_lock:
            try:
                fresh = (time.time() - self._last_warmup_ok) < WARMUP_SKIP_SECONDS
            except Exception:
                fresh = False
            if fresh and self.state.warmed and self.has_waf_token():
                return dict(self.state.cookies)
            if self._warm_inflight is not None:
                waiter = self._warm_inflight
                leader = False
            else:
                waiter = threading.Event()
                self._warm_inflight = waiter
                leader = True
        if not leader:
            # Follower: wait for the leader to publish its refresh (the flag is
            # set in the leader's finally), then share its result. Waiting is
            # strictly better than giving up: the jar held here is the one the
            # leader is in the middle of replacing, so returning early sends
            # cookies that are already known to be stale.
            budget = _follower_wait_budget(timeout)
            try:
                done = waiter.wait(timeout=budget)
            except Exception:
                done = waiter.is_set()
            if not done:
                log.warning("obscura warmup follower gave up after %.0fs without "
                            "the leader's refresh; sending the current jar "
                            "(%d cookies)", budget, len(self.state.cookies))
            return dict(self.state.cookies)
        try:
            with self._proc_lock:
                # Pass 1: execute challenge.js, persist cookies to profile. Its stdout
                # already carries the cookie dump, so parsing it avoids a second full
                # browser spawn (pass1 -> dump_cookies would each launch the engine).
                cookies: dict[str, str] = {}
                try:
                    proc = self._run("fetch", CHAT_HOME, "--dump", "cookies", "--timeout", "30",
                                     "--wait", "6", "--quiet", timeout=timeout)
                    cookies = self._parse_cookie_dump(proc.stdout)
                except Exception as exc:
                    log.warning("obscura warmup pass1 failed: %s", exc)
                # Pass 2 (fallback): read back the jar (proves persistence) + capture UA.
                # UA rarely changes: reuse cached value to save one browser spawn.
                if not cookies:
                    cookies = self.dump_cookies()
                if not cookies and self.state.cookies:
                    log.warning("obscura dump empty, keeping %d cached cookies", len(self.state.cookies))
                    cookies = dict(self.state.cookies)
                ua = self.state.user_agent or self.fetch_ua()
                self.state = WafState(cookies=cookies, user_agent=ua, warmed=bool(cookies))
                if self.state.warmed and self.has_waf_token():
                    try:
                        self._last_warmup_ok = time.time()
                    except Exception:
                        pass
                log.info("obscura warmup: %d cookies, ua=%s", len(cookies), ua[:60])
                return cookies
        finally:
            with self._warm_lock:
                self._warm_inflight = None
                try:
                    waiter.set()
                except Exception:
                    pass

    def dump_cookies(self, timeout: int = 30) -> dict[str, str]:
        with self._proc_lock:
            try:
                proc = self._run("fetch", CHAT_HOME, "--dump", "cookies",
                                 "--timeout", "20", "--quiet", timeout=timeout)
                return self._parse_cookie_dump(proc.stdout)
            except Exception as exc:
                log.warning("obscura dump_cookies failed: %s", exc)
                return {}

    def fetch_ua(self, timeout: int = 30) -> str:
        with self._proc_lock:
            try:
                proc = self._run("fetch", "https://example.com", "--eval", "navigator.userAgent",
                                 "--timeout", "20", "--quiet", timeout=timeout)
                return proc.stdout.strip().strip('"')[:300]
            except Exception:
                return ""

    def cookie_header(self, names: "Iterable[str] | None" = None) -> str:
        """Join the jar as a Cookie header, optionally filtered to ``names``.

        ``None`` ships every cookie (the historical behavior). A name
        allowlist exists because the jar is ONE upstream's WAF session: a
        caller talking to a different host must not hand it cookies it never
        issued. Jar order is preserved.
        """
        if names is None:
            return "; ".join(f"{k}={v}" for k, v in self.state.cookies.items())
        wanted = set(names)
        return "; ".join(f"{k}={v}" for k, v in self.state.cookies.items()
                         if k in wanted)

    def has_waf_token(self) -> bool:
        return "aws-waf-token" in self.state.cookies

    def status(self) -> dict:
        return {
            "binary": self.binary,
            "version": self.version(),
            "profile": self.profile,
            "warmed": self.state.warmed,
            "has_waf_token": self.has_waf_token(),
            "cookie_names": sorted(self.state.cookies),
            "user_agent": self.state.user_agent,
        }


# Process-wide default bridge for out-of-server callers (ban probes). The
# server owns its own bridge on app.state; this registry lets header builders
# reuse a warm WAF/UA bridge without threading app.state through every call.
_DEFAULT_BRIDGE: ObscuraBridge | None = None
_DEFAULT_BRIDGE_LOCK = threading.Lock()


def register_default_bridge(bridge: ObscuraBridge | None) -> None:
    """Publish the app's bridge so module-level header helpers can reuse it."""
    global _DEFAULT_BRIDGE
    with _DEFAULT_BRIDGE_LOCK:
        _DEFAULT_BRIDGE = bridge


def get_default_bridge() -> ObscuraBridge | None:
    """Return the registered bridge (may be unwarmed), else None."""
    with _DEFAULT_BRIDGE_LOCK:
        return _DEFAULT_BRIDGE
