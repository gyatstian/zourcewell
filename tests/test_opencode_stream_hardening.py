"""opencode streaming + managed-serve hardening: watchdog budget, live tail, child reap.

Three defects, each pinned by a test that fails on the pre-change code:

* **The consumer watchdog was one socket read in both stream modes.** In
  ``stream_mode: buffered`` (and in ``live`` *with* tools, where content
  deltas are deliberately suppressed at completion_service.py:528) the
  producer emits nothing while the whole turn runs, so the 190 s watchdog was
  structurally smaller than the producer's own budget
  (create 30 + turn 180 + one overload retry 180) and a valid, fully
  assembled reply was answered ``{"error": {"message": "upstream timed out"}}``
  and thrown away. ``ARCHITECTURE.md`` makes the multiplier a cross-provider
  contract; DeepSeek honours it (providers/deepseek/completion_service.py:1156)
  and opencode did not.
* **The live path had no coverage at all.** ``client.stream_message`` (the
  hand-rolled one-byte SSE frame splitter and the event dispatch) and
  ``completion_service._attempt_live`` had never run under a test, so the two
  rules that make live streaming safe -- content suppressed when the caller
  sent ``tools``, and the authoritative ``GET /api/session/:id/message``
  result returned instead of the assembled tail -- were unpinned.
* **``managed_serve.ensure()`` orphaned the child it replaced.** ``_proc`` is
  the only handle ``shutdown()`` and the ``atexit`` hook know, so overwriting
  it stranded a managed ``opencode serve`` -- holding its port, its ``work/``
  cwd and its per-boot basic-auth password -- for the life of the bridge. The
  reuse probe is a 2 s health GET, so an *unresponsive* serve (not a dead
  one) missed it and booted a replacement on a new port. ``ensure()`` runs
  from every serve-down error path (completion_service.py:202, 288), so these
  accumulated. The not-ready path compounded it: ``terminate()`` with no
  ``wait()``, and ``_proc`` never assigned, so nothing would ever reap that
  child either.

No network, no opencode binary, no credentials: the serve is a fake
``urlopen()`` and every process is a fake ``Popen``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import time
import types
import urllib.parse

import pytest

from support import collect_sse, settings, sse_deltas, sse_text
from deepseaport.config import Settings
from deepseaport.core import engine as _engine
from deepseaport.providers.opencode import client as OC
from deepseaport.providers.opencode import completion_service as OCS
from deepseaport.providers.opencode import managed_serve as MS
from deepseaport.providers.opencode.models import DEFAULT_MODEL
from deepseaport.server import create_app
from deepseaport.server_concurrency import _CONSUMER_GRACE_SECONDS
from functools import partial
from deepseaport.server_routes import _prepare

SESSION = "ses_live"
# The id the prompt POST answers with: it anchors the poll to THIS turn.
USER_ID = "msg_u"
# Text that only exists in the deltas, and text that only exists in the
# authoritative GET: they must never be confused for one another.
TAIL_TEXT = "Partial tail"
FINAL_TEXT = "Authoritative answer."


# Default kwargs differ per file; the shared base dict lives in support.py.
_settings = partial(settings, provider="opencode", log_level="INFO",
                     warmup_on_startup=True)


def _body(app, prep) -> str:
    """One opencode run as a decoded SSE body: no pool, no item."""
    return sse_text(asyncio.run(collect_sse(OCS, app, None, None, prep)))


def _texts(deltas: list[dict], key: str) -> list[str]:
    return [d[key] for d in deltas if key in d]


# ---------------------------------------------------------------------------
# Fake `opencode serve` (opencode v2 HttpApi) at the urllib seam, so the real
# client code runs
# ---------------------------------------------------------------------------


class _FakeResp:
    """urlopen() result: supports both ``read()`` and the SSE ``read(1)``."""

    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self._pos = 0
        self.status = status

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._payload) - self._pos
        chunk = self._payload[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _frame(event: dict) -> str:
    return "data: " + json.dumps(event) + "\n\n"


def _json_body(obj) -> _FakeResp:
    return _FakeResp(json.dumps(obj).encode("utf-8"), 200)


def _event(kind: str, **data) -> str:
    """One bus frame; v2 carries the payload under ``data``, not properties."""
    return _frame({"type": kind, "data": {"sessionID": SESSION, **data}})


def _fake_serve(monkeypatch, *, messages: list, sse: str | None = None,
                sse_up: bool = True) -> list[str]:
    """Route the v2 endpoints the opencode client uses; return the call log.

    ``sse`` is what ``GET /api/event`` replays byte for byte; without it the
    bus answers 200 with an empty body. ``sse_up=False`` refuses that
    connection outright, which is the "the live tail never started" case the
    blocking fallback exists for. ``messages`` is the CHRONOLOGICAL timeline;
    the ``/message`` route reverses it like the real serve does for
    ``order=desc``, so the client's own re-reverse is what these tests read.
    Anything else is a test bug and raises.
    """
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_USER_AGENT", "opencode/1.18.32")
    seen: list[str] = []

    def _urlopen(req, timeout=None):
        url = str(getattr(req, "full_url", "") or "")
        split = urllib.parse.urlsplit(url)
        path, query = split.path, split.query
        try:
            method = str(req.get_method()).upper()
        except Exception:
            method = "GET"
        seen.append(f"{method} {path}")
        if path.endswith("/api/event"):
            if not sse_up:
                raise OSError("connection refused (no live tail)")
            return _FakeResp((sse or "").encode("utf-8"), 200)
        if path.endswith("/api/info"):
            return _json_body({"version": "2.4.0"})
        if path.endswith("/message"):
            items = list(reversed(messages)) if "order=desc" in query else list(messages)
            return _json_body({"data": items, "cursor": {}})
        if method == "POST" and path.endswith("/api/session"):
            return _json_body({"data": {"id": SESSION}})
        if method == "POST" and path.endswith("/prompt"):
            return _json_body({"data": {"id": USER_ID}})
        if method == "POST" and (path.endswith("/model") or path.endswith("/agent")):
            return _FakeResp(b"", 204)
        if method == "DELETE" and path.endswith(f"/api/session/{SESSION}"):
            return _FakeResp(b"", 204)
        raise AssertionError(f"unexpected serve request: {method} {url}")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return seen


def _live_sse() -> str:
    """The frame stream the serve's /api/event bus writes for one turn.

    The leading comment line and ``event:`` line exercise two branches of the
    hand-rolled splitter that no other test reaches: it must skip an SSE
    comment and ignore an ``event:`` line while still collecting the
    following ``data:`` lines into one frame.
    """
    return "".join([
        ": keep-alive\n",
        "event: session.text.delta\n",
        _event("session.reasoning.delta", messageID="msg_a", delta="weigh"),
        _event("session.text.delta", messageID="msg_a", delta="Par"),
        _event("session.text.delta", messageID="msg_a", delta="tial tail"),
        _event("session.reasoning.delta", messageID="msg_a", delta="ing"),
        # Another session on the same bus: the sessionID gate must drop it.
        _frame({"type": "session.text.delta",
                "data": {"sessionID": "ses_other", "delta": "NOT OURS"}}),
        # An event the client does not dispatch must be tolerated, not fatal.
        _frame({"type": "session.updated",
                "data": {"sessionID": SESSION, "info": {"title": "x"}}}),
        _event("session.execution.succeeded", messageID="msg_a"),
    ])


def _assistant_item(text: str, thinking: str = "", usage: int = 11) -> dict:
    return {"id": "msg_a", "type": "assistant", "agent": "deepseaport-passthrough",
            "model": {"providerID": "opencode",
                      "modelID": "muse-spark-1.3-contributor-free"},
            "time": {"created": 1.0, "streamed": 2.0, "completed": 3.0},
            "content": [{"type": "reasoning", "text": thinking},
                        {"type": "text", "text": text}],
            "finish": "stop", "rawFinish": "stop", "cost": 0.0,
            "tokens": {"input": 5, "output": usage, "reasoning": 3,
                       "cache": {"read": 0, "write": 0}}}


def _user_item() -> dict:
    return {"id": USER_ID, "type": "user",
            "time": {"created": 1.0, "completed": 1.0}, "text": "hi"}


def _final_messages(content: str = FINAL_TEXT, thinking: str = "weighing") -> list:
    """The timeline of the turn that just ran, oldest first."""
    return [_user_item(), _assistant_item(content, thinking)]


# ---------------------------------------------------------------------------
# 1. The consumer watchdog must cover the producer's own budget
# ---------------------------------------------------------------------------


def _spy_stream_context(monkeypatch) -> list[tuple[bool, float]]:
    """Record (buffered, consumer_timeout) for every stream context built."""
    seen: list[tuple[bool, float]] = []
    real = _engine.new_stream_context

    def _spy(timeout, grace=10.0, **kw):
        result = real(timeout, grace, **kw)
        seen.append((bool(kw.get("buffered")), result[4]))
        return result

    monkeypatch.setattr(_engine, "new_stream_context", _spy)
    return seen


def _patch_blocking_serve(monkeypatch, content: str = "buffered reply"):
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: SESSION)
    monkeypatch.setattr(
        OC, "send_message",
        lambda sid, pid, mid, agent, prompt, timeout=180, variant=None, **kw:
        {"content": content, "thinking": "", "usage_total": 2, "finish": "stop"})
    monkeypatch.setattr(OC, "delete_session", lambda sid, timeout=10, **kw: None)


def test_buffered_mode_sizes_the_watchdog_for_the_whole_producer(monkeypatch):
    """Pre-change: ``buffered`` was never passed, so the watchdog was 190 s.

    The producer's own deadline here is create(30) + turn(180) + a retry(180).
    Sizing the consumer off one read is what turned a healthy-but-slow serve
    into a client-visible timeout with the answer discarded.
    """
    seen = _spy_stream_context(monkeypatch)
    _patch_blocking_serve(monkeypatch)
    settings = _settings(stream_mode="buffered")
    app = create_app(settings)
    prep = _prepare({"model": DEFAULT_MODEL, "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]}, settings)

    body = _body(app, prep)

    assert "buffered reply" in body
    assert seen == [(True, pytest.approx(
        (OC.DEFAULT_TIMEOUT + _CONSUMER_GRACE_SECONDS)
        * _engine._BUFFERED_PRODUCER_READS))], seen


def test_live_mode_watchdog_stays_exactly_one_read(monkeypatch):
    """The other half: live keeps the single-read budget byte for byte.

    Every delta resets the per-get clock, so only genuine silence should be
    bounded -- tripling this budget would let a wedged serve hold a request
    open for the full producer budget instead.
    """
    seen = _spy_stream_context(monkeypatch)
    _fake_serve(monkeypatch, messages=_final_messages(), sse=_live_sse())
    settings = _settings(stream_mode="live")
    app = create_app(settings)
    prep = _prepare({"model": DEFAULT_MODEL, "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]}, settings)

    body = _body(app, prep)

    # Live mode without tools streams the deltas and closes on a stop chunk;
    # the authoritative result is not re-sent as a duplicate content frame.
    deltas = sse_deltas(body)
    assert _texts(deltas, "content") == ["Par", "tial tail"], deltas
    assert seen == [(False, pytest.approx(
        OC.DEFAULT_TIMEOUT + _CONSUMER_GRACE_SECONDS))], seen


# ---------------------------------------------------------------------------
# 2. The live tail: delta dispatch, tool suppression, authoritative result
# ---------------------------------------------------------------------------


def test_live_attempt_forwards_deltas_in_order_and_returns_the_final_result(
        monkeypatch):
    """``stream_message``'s frame splitter and event dispatch, end to end.

    The deltas deliberately spell a text (``Partial tail``) that is NOT the
    reply: the value returned must come from ``GET /api/session/:id/message``,
    so a partial tail can never be served as the answer.
    """
    calls = _fake_serve(monkeypatch, messages=_final_messages(), sse=_live_sse())
    content_seen: list[str] = []
    thinking_seen: list[str] = []

    content, thinking, error, usage = OCS._attempt_live(
        "hi", DEFAULT_MODEL, 30,
        on_content=content_seen.append, on_thinking=thinking_seen.append)

    assert error is None
    # Reasoning and text are dispatched by event type, interleaved, in order.
    assert content_seen == ["Par", "tial tail"], content_seen
    assert thinking_seen == ["weigh", "ing"], thinking_seen
    # The foreign session's delta never reached the callbacks.
    assert "NOT OURS" not in "".join(content_seen)
    # Authoritative result, not the assembled tail ("Partial tail").
    assert content == FINAL_TEXT and TAIL_TEXT not in content
    assert thinking == "weighing"
    assert usage == 11
    # v2 is session-scoped: the model, the agent and the prompt all went out
    # on the same session the bus tail was scoped to, and it was cleaned up.
    assert f"POST /api/session/{SESSION}/model" in calls
    assert f"POST /api/session/{SESSION}/agent" in calls
    assert f"POST /api/session/{SESSION}/prompt" in calls
    assert f"DELETE /api/session/{SESSION}" in calls


@contextlib.contextmanager
def _fast_client_clock(monkeypatch):
    """Jump the client's monotonic clock past the wait-until-deadline loop.

    ``stream_message`` waits out its full budget when the bus never goes idle,
    so a test for the post-wait fallback would otherwise burn real seconds.
    """
    now = {"t": 0.0}

    def _monotonic() -> float:
        now["t"] += 30.0
        return now["t"]

    monkeypatch.setattr(OC, "time", types.SimpleNamespace(
        monotonic=_monotonic, sleep=lambda _s: None, time=time.time))
    yield


def test_live_attempt_falls_back_to_blocking_when_the_tail_never_starts(
        monkeypatch):
    """No SSE subscription -> blocking fallback.

    The live tail produced nothing, so the request must still succeed on the
    blocking turn (the same code path as ``stream_mode: buffered``) rather
    than fail with ``OPENCODE_UPSTREAM``.
    """
    monkeypatch.setattr(OC, "create_session", lambda timeout=20, **kw: SESSION)
    calls = _fake_serve(monkeypatch, messages=[], sse_up=False)
    blocking: list[str] = []

    def _send(sid, pid, mid, agent, prompt, timeout=180, variant=None, **kw):
        blocking.append(sid)
        return {"content": "blocking answer", "thinking": "", "usage_total": 1,
                "finish": "stop"}

    monkeypatch.setattr(OC, "send_message", _send)

    with _fast_client_clock(monkeypatch):
        content, _thinking, error, _usage = OCS._attempt_live(
            "hi", DEFAULT_MODEL, 30, on_content=lambda _t: None)

    assert error is None
    assert content == "blocking answer"
    # The live turn really was attempted and only the bus was missing, so the
    # fallback is not masking a broken prompt route.
    assert "GET /api/event" in calls, calls
    assert f"POST /api/session/{SESSION}/prompt" in calls, calls
    # The blocking turn is what answered: the live tail produced nothing.
    assert blocking == [SESSION]


def test_live_stream_suppresses_content_but_not_reasoning_when_tools_are_sent(
        monkeypatch):
    """The two deliberate live rules, on the wire.

    With ``tools`` in the request the reply may be tool JSON, so live content
    deltas would leak as user-visible text -- but reasoning is never tool
    output and always streams. And the text that does reach the client is the
    authoritative GET result, exactly once, never the assembled tail.
    """
    _fake_serve(monkeypatch, messages=_final_messages(), sse=_live_sse())
    settings = _settings(stream_mode="live")
    app = create_app(settings)
    prep = _prepare({
        "model": DEFAULT_MODEL, "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {}}}}],
    }, settings)
    assert prep["tools"], "the request must reach _stream_completion with tools"

    body = _body(app, prep)
    deltas = sse_deltas(body)

    assert _texts(deltas, "content") == [FINAL_TEXT], \
        "live content deltas leaked next to a tool_calls reply: " + body
    assert TAIL_TEXT not in body
    assert _texts(deltas, "reasoning_content") == ["weigh", "ing"], body
    # The reply ends as a normal stop, never a half-written tail.
    assert body.rstrip().endswith("data: [DONE]")


def test_live_stream_without_tools_forwards_content_deltas(monkeypatch):
    """Plain chat: content streams as it arrives, and is not re-sent at the end."""
    _fake_serve(monkeypatch, messages=_final_messages(), sse=_live_sse())
    settings = _settings(stream_mode="live")
    app = create_app(settings)
    prep = _prepare({"model": DEFAULT_MODEL, "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]}, settings)

    deltas = sse_deltas(_body(app, prep))

    assert _texts(deltas, "content") == ["Par", "tial tail"], deltas
    assert _texts(deltas, "reasoning_content") == ["weigh", "ing"], deltas


# ---------------------------------------------------------------------------
# 3. Managed serve: reap the child we replace, and reap the not-ready one
# ---------------------------------------------------------------------------

NEW_PORT = 45999
NEW_BASE = f"http://127.0.0.1:{NEW_PORT}"


class _FakeProc:
    """Popen stand-in recording the lifecycle calls manage_serve makes."""

    def __init__(self, pid: int = 4242, ignore_sigterm: bool = False):
        self.pid = pid
        self.calls: list[str] = []
        self.returncode = None
        self._alive = True
        self._ignore_sigterm = ignore_sigterm

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.calls.append("terminate")
        if not self._ignore_sigterm:
            self._alive = False

    def kill(self):
        self.calls.append("kill")
        self._alive = False

    def wait(self, timeout=None):
        self.calls.append(f"wait({timeout})")
        if self._alive:
            raise subprocess.TimeoutExpired("opencode serve", timeout)
        self.returncode = 0
        return 0


def _posix_reap(monkeypatch):
    """Pin the reap to the POSIX channel and record the tree kill.

    ``_terminate_proc`` picks its channel from ``os.name``, and the two are
    genuinely different: Windows kills the process TREE (the tracked handle is
    the ``cmd.exe`` wrapper and the serve is its child), POSIX terminates the
    child it spawned. The reap-sequence assertions below are about the sequence
    -- terminate, wait, escalate -- so they pin one platform and stay
    identical on every runner; the Windows branch is pinned separately in
    ``test_managed_serve_hardening.py``. Stubbing ``_kill_tree`` here also
    keeps a fake pid away from a real taskkill.
    """
    monkeypatch.setattr(MS.os, "name", "posix")
    trees: list[int] = []
    monkeypatch.setattr(MS, "_kill_tree",
                        lambda pid: trees.append(pid) or True, raising=False)
    return trees


@contextlib.contextmanager
def _managed_state(proc=None, base: str = "", password: str = ""):
    """Isolate (and restore) the module-level managed-serve state.

    The atexit slot is pre-claimed so a test that boots successfully never
    registers a real interpreter-exit callback against a fake Popen.
    """
    saved = (MS._proc, MS._managed_base, MS._managed_password,
             MS._atexit_registered)
    MS._proc, MS._managed_base, MS._managed_password = proc, base, password
    MS._atexit_registered = True
    try:
        yield
    finally:
        (MS._proc, MS._managed_base, MS._managed_password,
         MS._atexit_registered) = saved


def _stub_managed_boot(monkeypatch, tmp_path, healthy_bases=(), proc=None) -> dict:
    """Stub every side effect of a boot; returns the Popen capture dict."""
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_BASE", "")
    monkeypatch.setenv("DEEPSEAPORT_OPENCODE_MANAGED", "1")
    monkeypatch.setattr(MS, "managed_base_dir", lambda: tmp_path / "m")
    monkeypatch.setattr(MS, "_find_free_port", lambda: NEW_PORT)
    monkeypatch.setattr(MS, "_probe",
                        lambda base, pw, timeout=2.0: base in healthy_bases)
    monkeypatch.setattr(MS.shutil, "which", lambda name: "opencode")
    monkeypatch.setattr(
        MS.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            ["opencode", "--version"], 0, "1.18.32", ""))
    captured: dict = {}

    def _popen(argv, **kw):
        captured["argv"] = list(argv)
        captured["env"] = dict(kw.get("env") or {})
        captured["proc"] = proc or _FakeProc()
        return captured["proc"]

    monkeypatch.setattr(MS.subprocess, "Popen", _popen)
    return captured


def test_boot_reaps_the_child_it_replaces(monkeypatch, tmp_path):
    """Pre-change: ``_proc = proc`` dropped the old child on the floor.

    A 2 s health probe misses an unresponsive -- not dead -- serve, so this
    is the ordinary path on a loaded box. ``shutdown()`` and the atexit hook
    only ever knew ``_proc``, so the previous ``opencode serve`` kept its
    port, its ``work/`` cwd and its per-boot basic-auth password forever.
    """
    monkeypatch.setenv("DEEPSEAPORT_KEYS", "sk-parent-secret")
    old = _FakeProc(pid=1111)
    trees = _posix_reap(monkeypatch)
    captured = _stub_managed_boot(monkeypatch, tmp_path,
                                  healthy_bases={NEW_BASE})

    with _managed_state(proc=old, base="http://127.0.0.1:41000",
                        password="old-pw"):
        assert MS.ensure(timeout=1.0) == NEW_BASE
        assert old.calls == ["terminate", "wait(5.0)"], old.calls
        assert trees == [], "the POSIX channel must not shell out to taskkill"
        assert MS._proc is captured["proc"] and MS._proc is not old

    # The reap must not have disturbed the child env boundary: the parent
    # secret is still stripped and the per-boot keys still set after it.
    assert "DEEPSEAPORT_KEYS" not in captured["env"]
    assert captured["env"]["OPENCODE_SERVER_PASSWORD"]
    assert captured["env"]["OPENCODE_CONFIG_DIR"].endswith("cfg")


def test_boot_reaps_the_recorded_pid_when_the_handle_is_gone(monkeypatch, tmp_path):
    """No ``_proc`` (module reloaded, another process wrote serve.json).

    The recorded pid still goes through the identity-gated reaper, which
    refuses to signal anything that is not the managed serve.
    """
    killed: list[int] = []
    _stub_managed_boot(monkeypatch, tmp_path, healthy_bases={NEW_BASE})
    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:41000",
                        "old-pw", 4242, MS._config_hash())
    monkeypatch.setattr(MS, "_kill_pid", lambda pid: killed.append(pid))

    with _managed_state():
        assert MS.ensure(timeout=1.0) == NEW_BASE
        assert killed == [4242], killed


def test_hash_change_does_not_signal_the_recorded_pid_twice(monkeypatch, tmp_path):
    """The config-change restart already reaped it; do not signal it again.

    A second ``_kill_pid`` would fail the identity gate on a now-dead process
    and log a misleading "not signalling it" warning for a reap that worked.
    """
    killed: list[int] = []
    _stub_managed_boot(monkeypatch, tmp_path,
                       healthy_bases={"http://127.0.0.1:41000", NEW_BASE})
    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    MS._write_port_file(tmp_path / "m" / "serve.json", "http://127.0.0.1:41000",
                        "old-pw", 4242, "stale-hash")
    monkeypatch.setattr(MS, "_kill_pid", lambda pid: killed.append(pid))

    with _managed_state():
        assert MS.ensure(timeout=1.0) == NEW_BASE
        assert killed == [4242], killed


@pytest.mark.parametrize("stubborn", [False, True])
def test_not_ready_child_is_terminated_and_reaped(monkeypatch, tmp_path, stubborn):
    """Pre-change: ``proc.terminate()`` with no ``proc.wait()``.

    ``_proc`` is never assigned on this path, so nothing would ever wait on
    that child: a POSIX zombie until the GC collects the Popen, an unowned
    live process on Windows. A child that ignores SIGTERM is killed.
    """
    proc = _FakeProc(pid=2222, ignore_sigterm=stubborn)
    _posix_reap(monkeypatch)
    captured = _stub_managed_boot(monkeypatch, tmp_path, healthy_bases=(),
                                  proc=proc)
    # The readiness loop probes for max(5.0, timeout) with a 0.25 s sleep.
    # A clock that jumps past the deadline keeps that out of the test's
    # runtime; only ensure()'s loop reads it here.
    clock = {"t": 1000.0}

    def _fast_clock() -> float:
        clock["t"] += 30.0
        return clock["t"]

    monkeypatch.setattr(MS.time, "monotonic", _fast_clock)

    with _managed_state():
        assert MS.ensure(timeout=1.0) is None
        assert captured["proc"] is proc
        assert proc.calls[:2] == ["terminate", "wait(5.0)"], proc.calls
        assert MS._proc is None
    if stubborn:
        assert proc.calls == ["terminate", "wait(5.0)", "kill", "wait(5.0)"], \
            proc.calls
    else:
        assert proc.calls == ["terminate", "wait(5.0)"], proc.calls


def test_shutdown_reaps_the_tracked_child(monkeypatch):
    """The reap helper is the single exit path, atexit included."""
    _posix_reap(monkeypatch)
    proc = _FakeProc()
    with _managed_state(proc=proc):
        MS.shutdown()
        assert proc.calls == ["terminate", "wait(5.0)"], proc.calls
        assert MS._proc is None
        MS.shutdown()  # idempotent: nothing left to reap
    assert proc.calls == ["terminate", "wait(5.0)"], proc.calls
