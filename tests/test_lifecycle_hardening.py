"""Lifecycle/resource hardening for the bridge's own seams.

Four defects, each of which needed a fake at the boundary instead of a real
obscura/browsers process:

1. ``core/mcp_client.py`` -- a handshake that never completed orphaned the
   child process AND both reader threads (they hold bound methods -> ``self``,
   so the object is rooted in ``threading._active`` and ``__del__`` never
   runs). The caller releases the profile lock anyway, so the next login spawns
   a second obscura on the same ``--storage-dir``.
2. ``providers/deepseek/real_browser.py`` -- one ban-phrase frame latched a
   permanent ban, which the caller turns into ``banned: true`` in
   ``config.json`` while throwing the working token away.
3. ``providers/deepseek/pow.py`` -- one shared ``<name>.tmp`` for every caller.
4. ``core/obscura_bridge.py`` -- the singleflight follower's wait was shorter
   than the leader's worst case, so it returned the pre-refresh jar.

No real subprocess, browser or network is involved: every fake below is
deterministic and drives the loop with events/barriers, never sleeps.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time

import pytest

# --- Task 1: McpClient construction/teardown -------------------------------


class _Pipe:
    """Writable stdin / readable stdout stand-in.

    Iteration blocks on :attr:`eof` so the reader threads are parked exactly
    like a child that never answers, and unblocks (EOF) when the child is
    terminated -- what a real pipe does when the write end closes.
    """

    def __init__(self):
        self.eof = threading.Event()
        self.closed = False
        self.written: list[str] = []

    # -- stdin --
    def write(self, data):
        if self.closed:
            raise ValueError("I/O operation on closed pipe")
        self.written.append(data)

    def flush(self):
        pass

    # -- stdout/stderr (iterable) --
    def __iter__(self):
        self.eof.wait(10.0)
        return iter(())

    def readline(self):
        return ""

    def close(self):
        self.closed = True
        self.eof.set()


class _FakeChild:
    """Popen stand-in: alive until terminate()/kill(), silent until then."""

    def __init__(self, alive: bool = True):
        self.stdin = _Pipe()
        self.stdout = _Pipe()
        self.stderr = _Pipe()
        self._alive = alive
        self.returncode = None if alive else 1
        self.terminated = False
        self.killed = False
        self.waits = 0

    def poll(self):
        return None if self._alive else (self.returncode or 1)

    def _die(self):
        self._alive = False
        self.returncode = -15
        # Child death closes the pipes: both reader loops hit EOF and exit.
        self.stdout.eof.set()
        self.stderr.eof.set()

    def terminate(self):
        self.terminated = True
        self._die()

    def kill(self):
        self.killed = True
        self._die()

    def wait(self, timeout=None):
        self.waits += 1
        return self.returncode


def _reader_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if "_read_loop" in t.name or "_stderr_loop" in t.name]


def _patch_popen(monkeypatch, child: _FakeChild | None):
    """Point McpClient's Popen at `child`; None means "Popen raises"."""
    from deepseaport.core import mcp_client as mc

    if child is None:
        def _boom(*a, **k):
            raise FileNotFoundError("obscura not found")
        monkeypatch.setattr(mc.subprocess, "Popen", _boom)
        return
    monkeypatch.setattr(mc.subprocess, "Popen", lambda *a, **k: child)


def test_mcp_failed_handshake_terminates_child_and_joins_readers(monkeypatch):
    """A live child that never answers `initialize` must not be orphaned."""
    from deepseaport.core import mcp_client as mc

    before = len(_reader_threads())
    child = _FakeChild(alive=True)
    _patch_popen(monkeypatch, child)

    with pytest.raises(TimeoutError, match="initialize timed out"):
        mc.McpClient("fake-obscura", "profile", timeout=0.05)

    assert child.terminated is True, "hung child survived a failed handshake"
    assert child.poll() is not None
    assert len(_reader_threads()) == before, "reader threads outlived the client"
    assert child.stdin.closed is True


def test_mcp_child_death_during_handshake_cleans_up(monkeypatch):
    """Child exits on its own (EOF on both pipes) -> readers exit by EOF."""
    from deepseaport.core import mcp_client as mc

    before = len(_reader_threads())
    child = _FakeChild(alive=True)

    def _die_immediately(*a, **k):
        child._die()
        return child

    monkeypatch.setattr(mc.subprocess, "Popen", _die_immediately)

    with pytest.raises(RuntimeError, match="exited|failed"):
        mc.McpClient("fake-obscura", "profile", timeout=1.0)

    assert child.terminated is False, "an already-dead child needs no terminate"
    assert len(_reader_threads()) == before


def test_mcp_close_is_idempotent_and_partial_object_safe():
    from deepseaport.core import mcp_client as mc

    child = _FakeChild(alive=True)
    c = mc.McpClient.__new__(mc.McpClient)
    c.proc = child
    c._closed = False
    c.close()
    assert c._closed is True and child.terminated is True
    calls = child.waits
    c.close()
    assert child.waits == calls, "second close did work again"

    # Partially constructed: no proc, no reader threads (the Popen-raised path).
    half = mc.McpClient.__new__(mc.McpClient)
    half.proc = None
    half._closed = False
    half.close()
    assert half._closed is True
    half.close()


def test_mcp_popen_failure_propagates_without_secondary_error(monkeypatch):
    """A failure BEFORE Popen must surface as itself, not as a cleanup crash."""
    from deepseaport.core import mcp_client as mc

    _patch_popen(monkeypatch, None)
    with pytest.raises(FileNotFoundError):
        mc.McpClient("missing-binary", "profile", timeout=0.05)
    assert _reader_threads() == []


# --- Task 2: the CDP driver's ban verdict needs a streak -------------------


def _login_browser(monkeypatch, snaps, timeout=5.0):
    """A RealBrowser whose login loop is driven by a snapshot script."""
    from deepseaport.providers.deepseek import real_browser as rb

    browser = rb.RealBrowser.__new__(rb.RealBrowser)
    browser.timeout = timeout
    monkeypatch.setattr(browser, "navigate", lambda url: None)
    monkeypatch.setattr(browser, "dismiss_cookie_banner", lambda: None)
    monkeypatch.setattr(browser, "fill_credentials", lambda e, p: None)
    monkeypatch.setattr(browser, "click_login", lambda: None)
    monkeypatch.setattr(browser, "evaluate", lambda *a, **k: "cleared")

    script = list(snaps)
    seen = {"n": 0}

    def _snap():
        i = seen["n"]
        seen["n"] += 1
        return script[min(i, len(script) - 1)]

    monkeypatch.setattr(browser, "snapshot", _snap)
    monkeypatch.setattr(rb.time, "sleep", lambda *a, **k: None)
    return browser, seen


def _snap(body, token=None, captcha=False):
    return {"url": "https://chat.deepseek.com/sign_in", "token": token,
            "body": body, "captcha_visible": captcha}


def test_login_single_ban_frame_then_token_is_not_a_ban(monkeypatch):
    """One ban-word frame must not veto a working credential.

    ``LOGIN_BAN_PHRASES`` matches bare tokens ("muted", "suspended",
    "violation"), so a single frame of ordinary page text used to ban the
    account for good.
    """
    browser, seen = _login_browser(monkeypatch, [
        _snap("your account has been muted pending review"),
        _snap("welcome back", token="tok-live"),
    ])
    token, banned, detail = browser.login("a@x.com", "pw")
    assert token == "tok-live"
    assert banned is False, detail
    assert seen["n"] == 2, "one poll too many or too few"


def test_login_two_consecutive_ban_frames_are_banned(monkeypatch):
    """Mirror of the MCP driver: ban_streak >= 2 (auth.py:267-274)."""
    browser, _ = _login_browser(monkeypatch, [
        _snap("account has been suspended"),
        _snap("account has been suspended"),
    ])
    token, banned, detail = browser.login("a@x.com", "pw")
    assert banned is True
    assert "suspension" in detail


def test_login_ban_streak_resets_on_a_healthy_frame(monkeypatch):
    """Two sightings separated by a healthy page are not consecutive."""
    browser, _ = _login_browser(monkeypatch, [
        _snap("account has been suspended"),
        _snap("sign in to continue"),
        _snap("account has been suspended"),
        _snap("welcome back", token="tok-live"),
    ])
    token, banned, detail = browser.login("a@x.com", "pw")
    assert token == "tok-live"
    assert banned is False, detail


def test_login_captcha_mid_streak_still_surfaces_the_hint(monkeypatch):
    """Mirror of tests/test_auth_browser_gaps.py:758-783: a ban hint seen on
    an earlier frame is reported on a captcha exit, which cannot state a
    verdict of its own -- the hint goes to the DETAIL, never the banned slot
    (auth.py only stops on a token or a confirmed ban, so the Obscura fallback
    still runs). Re-pinned to the deepseek login-policy change."""
    browser, _ = _login_browser(monkeypatch, [
        _snap("account muted"),
        _snap("just a moment"),
    ])
    token, banned, detail = browser.login("a@x.com", "pw")
    assert token is None
    assert banned is False
    assert "captcha" in detail
    assert "[ban wording seen in an earlier frame]" in detail


# --- Task 3: ensure_wasm's temp file is per-call ---------------------------


class _Resp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_ensure_wasm_concurrent_callers_leave_no_tmp(monkeypatch, tmp_path):
    """Two callers that both pass the exists() check must not share a tmp."""
    from deepseaport.providers.deepseek import pow as pw

    blob = b"\x00asm\x01\x00\x00\x00" + b"x" * 4000
    assert len(blob) > 1000
    monkeypatch.setattr(pw, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pw, "_WASM_PATH", None)

    downloads = {"n": 0}
    start = threading.Barrier(2)
    downloading = threading.Barrier(2)

    def _get(url, **kwargs):
        downloads["n"] += 1
        # Both callers are past the exists() check when this releases.
        downloading.wait(5.0)
        return _Resp(blob)

    monkeypatch.setattr("curl_cffi.requests.get", _get)

    # Hold both callers at the point where they claim a temp file: with one
    # shared "<name>.tmp" the second caller's os.replace then finds its source
    # already moved (FileNotFoundError, or PermissionError on Windows while the
    # winner still holds it open) and leaves a stale .tmp behind. Barriers, not
    # sleeps, so the interleaving is identical on every run.
    real_mkstemp = tempfile.mkstemp
    claiming = threading.Barrier(2)
    temps: list[str] = []

    def _mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        temps.append(name)
        claiming.wait(5.0)
        return fd, name

    monkeypatch.setattr(pw.tempfile, "mkstemp", _mkstemp)

    results: dict = {}
    errors: dict = {}

    def _caller(idx):
        try:
            start.wait(5.0)
            results[idx] = pw.ensure_wasm()
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            errors[idx] = exc

    threads = [threading.Thread(target=_caller, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert len(results) == 2
    assert len(set(temps)) == 2, f"both callers claimed the same temp file: {temps}"

    path = tmp_path / pw.validated_wasm_name()
    assert path.exists() and path.read_bytes() == blob
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"stale temp files: {leftovers}"
    assert downloads["n"] == 2, "one caller short-circuited past the download"


# --- Task 4: the WAF singleflight follower's wait --------------------------


def _bridge(tmp_path, name):
    from deepseaport.core import obscura_bridge as ob

    return ob.ObscuraBridge(binary="obscura", profile=str(tmp_path / name))


def test_follower_wait_budget_covers_the_leader_worst_case():
    """Pass 1 (timeout) + dump_cookies(30) + fetch_ua(30) + slack."""
    from deepseaport.core import obscura_bridge as ob

    budget = ob._follower_wait_budget(45)
    assert budget >= 45 + 2 * ob.WARMUP_SUBSTEP_TIMEOUT
    # The old arithmetic was max(1, timeout + 5) = 50s for the default warmup.
    assert budget > max(1, 45 + 5)
    assert ob._follower_wait_budget(-10) > 1.0


def test_follower_waits_for_a_slow_leader_and_gets_the_refreshed_jar(tmp_path):
    """Regression: the follower used to give up and return the STALE jar.

    The old bound was ``max(1, int(timeout) + 5)``, which floors at 1s for any
    timeout <= -4 -- so a negative timeout is how this stays a fast test while
    still putting the leader's work past the old deadline.
    """
    from deepseaport.core import obscura_bridge as ob

    b = _bridge(tmp_path, "slow-leader")
    b.state = ob.WafState(cookies={"aws-waf-token": "stale"}, user_agent="ua",
                          warmed=True)
    b._last_warmup_ok = 0.0
    fresh = json.dumps([{"name": "aws-waf-token", "value": "fresh"}])
    leader_in_run = threading.Event()
    release = threading.Event()
    runs = {"n": 0}

    def _run(*args, **kwargs):
        runs["n"] += 1
        leader_in_run.set()
        release.wait(10.0)
        return subprocess.CompletedProcess(args, 0, stdout=fresh, stderr="")

    b._run = _run  # type: ignore[method-assign]

    results: dict = {}

    def _call(idx, timeout):
        results[idx] = b.warmup(timeout=timeout)

    leader = threading.Thread(target=_call, args=(1, -10))
    leader.start()
    assert leader_in_run.wait(5.0), "leader never reached the subprocess"
    follower = threading.Thread(target=_call, args=(2, -10))
    follower.start()

    # Past the old bound (max(1, -10 + 5) = 1s): pre-fix the follower had
    # already returned the stale jar here.
    time.sleep(1.5)
    assert follower.is_alive(), \
        "follower gave up before the leader finished and would send stale cookies"

    release.set()
    leader.join(10)
    follower.join(10)
    assert not follower.is_alive()
    assert results[1] == {"aws-waf-token": "fresh"}
    assert results[2] == {"aws-waf-token": "fresh"}, \
        "follower returned the pre-refresh jar instead of waiting"
    assert runs["n"] == 1, "the follower ran its own refresh instead of sharing"
    assert b.state.cookies == {"aws-waf-token": "fresh"}


def test_follower_returns_stale_jar_only_when_the_leader_never_finishes(
        monkeypatch, tmp_path, caplog):
    """The bound is still finite: an expired wait is logged, not silent."""
    from deepseaport.core import obscura_bridge as ob

    monkeypatch.setattr(ob, "_follower_wait_budget", lambda timeout: 0.05)
    b = _bridge(tmp_path, "wedged-leader")
    b.state = ob.WafState(cookies={"aws-waf-token": "stale"}, user_agent="ua",
                          warmed=True)
    b._last_warmup_ok = 0.0
    started = threading.Event()

    def _run(*args, **kwargs):
        started.set()
        time.sleep(0.6)
        return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")

    b._run = _run  # type: ignore[method-assign]
    results: dict = {}

    def _call(idx):
        results[idx] = b.warmup(timeout=45)

    leader = threading.Thread(target=_call, args=(1,))
    leader.start()
    assert started.wait(5.0)
    t0 = time.monotonic()
    _call(2)
    waited = time.monotonic() - t0
    leader.join(10)
    assert waited < 0.5, "follower waited on a budget it did not have"
    assert results[2] == {"aws-waf-token": "stale"}
    assert "follower" in caplog.text.lower()
# --- Task: the shared executors must not be able to hold the process --------

def test_shared_executor_workers_are_daemon_threads():
    """`serve` + Ctrl+C with a stream open appeared to hang for the socket budget.

    ``ThreadPoolExecutor`` takes no daemon option: ``initializer`` runs inside
    the worker *after* it is started (``Thread.daemon`` refuses to change once a
    thread is alive), and ``Thread.__init__`` otherwise inherits the SUBMITTER's
    flag -- so worker daemon-ness depended on which thread happened to submit
    first. On top of that ``concurrent.futures.thread`` registers every worker
    it creates with ``threading._register_atexit`` and joins them at interpreter
    exit.

    All three pools are pinned here. It is a one-line check because the bug was
    invisible from any behavioural test: a parked worker only shows up as process
    wall time charged at exit (the same trap
    tests/test_engine_retry_gaps.py documents for its ``time.sleep(30)`` fake).
    """
    from deepseaport import server_concurrency as SC

    for name in ("_CHALLENGE_EXECUTOR", "_CLEANUP_EXECUTOR", "_STREAM_EXECUTOR"):
        pool = getattr(SC, name)
        pool.submit(lambda: None).result(5.0)  # force one real worker thread
        assert pool._threads, f"{name} spawned no worker"
        for t in pool._threads:
            assert t.daemon is True, f"{name} worker {t.name} is not a daemon"


def test_shared_executor_still_bounds_and_returns_results():
    """The daemon swap must not cost the property these pools were built for.

    A bounded pool exists so a parallel-subagent burst QUEUES instead of piling
    unbounded raw threads. Same bound, same results, same Future API the three
    call sites use (``submit`` -> ``.result`` / ``.cancel``).
    """
    from concurrent.futures import Future
    from deepseaport import server_concurrency as SC

    pool = SC._DaemonExecutor(max_workers=2, thread_name_prefix="t")
    assert pool._max_workers == 2

    futs = [pool.submit(lambda i=i: i * 2) for i in range(6)]
    assert sorted(f.result(10.0) for f in futs) == [0, 2, 4, 6, 8, 10]
    assert len(pool._threads) <= 2, "the pool grew past its bound"

    boom = pool.submit(_raise_value_error)
    with pytest.raises(ValueError):
        boom.result(5.0)

    assert pool.submit(lambda: "ok").result(5.0) == "ok"
    # A Future the executor never hands work to must still be a real Future:
    # _await_cancelable calls .cancel() on it.
    assert isinstance(futs[0], Future)


def _raise_value_error():
    raise ValueError("boom")


def test_startup_warm_skips_deepseek_for_an_unknown_provider(monkeypatch):
    """Finding 23. ``provider: "glm"`` fetched the DeepSeek wasm at startup.

    Two separate defects met here:

    1. ``getattr(registry.get(pid), "needs_bridge", True)`` -- ``getattr(None,
       ...)`` is ``True``, so an id the registry does not know counted as
       "wants the browser" and ran an Obscura warmup for it.
    2. ``registry.get(pid) is None`` did ``continue``, leaving ``warmed =
       False``, which then fell into the inline DeepSeek PoW fallback -- a
       wasted wasm fetch for a provider that needs no wasm.

    Neither is visible from a settings object: it only shows up as a startup
    browser/window on a provider that has no accounts and no upstream at all.
    """
    from deepseaport import server_concurrency as SC
    from deepseaport.core import registry as reg
    from deepseaport.accounts import AccountConfig
    from support import settings as _mk_settings

    real_get = reg.get
    monkeypatch.setattr(reg, "get",
                        lambda pid: None if pid == "glm" else real_get(pid))

    calls: dict = {"bridge": 0, "wasm": 0}

    class _Bridge:
        def warmup(self, timeout: int = 45):
            calls["bridge"] += 1
            return {}

        def cookie_header(self):
            return ""

    monkeypatch.setattr(
        "deepseaport.providers.deepseek.pow.ensure_wasm",
        lambda **k: calls.__setitem__("wasm", calls["wasm"] + 1))

    s = _mk_settings(provider="glm",
                     accounts=[AccountConfig(email="a@x.com", token="t1")])
    SC._startup_warm(_Bridge(), s)

    assert calls == {"bridge": 0, "wasm": 0}, (
        "an unknown provider still drove the DeepSeek bridge/wasm path")


def test_startup_warm_skips_bridge_warmup_for_qwen_only(monkeypatch):
    """qwen consumes its own acw_tc jar, not the bridge's: no DeepSeek warmup.

    The startup obscura warmup targets CHAT_HOME (chat.deepseek.com), so a
    qwen-only setup must not boot it and plant cookies qwen then refuses to
    send. qwen's OWN warmup must still run -- it is what solves the qwen-host
    jar. ``needs_bridge`` stays True (one-shot chat still builds a bridge), so
    the startup gate is the separate ``needs_waf_jar``.
    """
    from deepseaport import server_concurrency as SC
    from deepseaport.providers.qwen.provider import QwenProvider
    from support import settings as _mk_settings

    calls = {"bridge": 0, "provider": []}

    class _Bridge:
        def warmup(self, timeout: int = 45):
            calls["bridge"] += 1
            return {}

    monkeypatch.setattr(QwenProvider, "warmup",
                        lambda self, bridge: calls["provider"].append(bridge))
    s = _mk_settings(provider="qwen")
    SC._startup_warm(_Bridge(), s)

    assert calls["bridge"] == 0, (
        "a qwen-only startup booted obscura against the DeepSeek host")
    assert calls["provider"], "qwen's own warmup was skipped"
    assert isinstance(calls["provider"][0], _Bridge)


def test_effective_log_level_prefers_runtime_override():
    """chat mode pins ``_runtime_log_level``; the lifespan used to ignore it.

    The override is restored to its pre-run value (None when unset) on exit,
    so a missing/None attribute MUST fall back to ``log_level``. No attribute
    at all (odd stand-in settings) falls back to INFO.
    """
    from deepseaport import server_concurrency as SC
    from support import settings as _mk_settings

    s = _mk_settings(log_level="INFO")
    assert SC._effective_log_level(s) == "INFO"
    s._runtime_log_level = "ERROR"
    assert SC._effective_log_level(s) == "ERROR"
    s._runtime_log_level = None
    assert SC._effective_log_level(s) == "INFO"

    class _NoLevels:
        pass

    assert SC._effective_log_level(_NoLevels()) == "INFO"
