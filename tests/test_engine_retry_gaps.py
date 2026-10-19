"""Regression tests for the engine/retry gaps E1-E5.

Each section names the pre-change failure it reproduces. All offline: no
network, no credentials, no browser.

* E1 ``core.engine.drive_retries`` let an exception from a recovery closure
  escape, so a broken PoW/WAF recovery replaced the upstream status with an
  unhandled 500 on the non-stream path.
* E2 ``core.engine.app_settings`` memoized ``load_settings()`` onto
  ``app.state``, binding a hand-built app to the operator's ``config.json``
  and letting provider saves rewrite it.
* E3 ``classify_error`` mapped a bare HTTP 401 to ``RetryKind.BAN``, and
  ``_terminal_ban`` ignored a negative ban probe, so one transient 401
  persisted ``banned: true`` for a healthy account.
* E5 the buffered stream watchdog was sized from ONE socket read, so a
  healthy-but-slow producer was killed with "upstream timed out" and its
  valid answer discarded.
* E7 the consumer applied that same one-read budget to the LAST mode that
  could not afford it -- live -- before the first payload, where no delta
  exists yet to reset the clock.
* E6 only the ``overload`` retry row was gated on "no live delta has left",
  so a pow/session/token/waf failure AFTER a partial emit re-POSTed the same
  payload (upstream appended a turn) and re-streamed from delta 0 into a
  stream the client had already read.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from support import cached_wasm, collect_sse, settings, sse_text
from deepseaport.providers.deepseek import pow as PoW
from deepseaport.accounts import AccountPool, PooledAccount
from deepseaport.config import AccountConfig, Settings
from deepseaport.core import engine as _engine
from deepseaport.providers.deepseek import client as DS
from deepseaport.providers.deepseek import completion_service as CS
from deepseaport.providers.qwen import completion_service as QCS
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare

CHALLENGE = {
    "algorithm": PoW.ALGORITHM,
    "challenge": "c",
    "salt": "s",
    "difficulty": 1000,
    "expire_at": 2**31,
    "signature": "sig",
    "target_path": "/api/v0/chat/completion",
}

# One socket read for the watchdog tests below. The buffered budget is
# `_BUFFERED_PRODUCER_READS` (3) x this, so a producer must land strictly
# between 1x and 3x of it to prove the old budget killed it and the new one
# does not. Spelled out here rather than inlined because the ratio is the whole
# test and a future editor who changes the producer delay without re-deriving
# it silently disables the assertion.
_WATCHDOG_READ_SECONDS = 0.5
# How long a faked "never speaks" producer is allowed to stay parked. It is a
# hang guard, never a wait that elapses on the happy path: the test releases
# it the moment the watchdog has fired, so the parked `_STREAM_EXECUTOR` worker
# unwinds immediately instead of pinning a non-daemon thread until interpreter
# exit (which is what `time.sleep(30)` used to cost -- invisible in pytest's
# own timing, but ~30s of real process wall time).
_SILENT_PRODUCER_PARK_SECONDS = 10.0

POW_ERROR = "POW_HEADER_ERROR: proof of work header is invalid"
SESSION_ERROR = "INVALID_SESSION_ID: chat_session not found"
WAF_ERROR = "HTTP_403_WAF: aws-waf token missing"
BARE_401 = "HTTP_401: Unauthorized"
OVERLOAD_ERROR = "generation_timeout: Server busy, please try again later."
BAN_ERROR = "5: user is muted (suspended until 16 September 2026 12:49)"


# --- shared fixtures/helpers -------------------------------------------------


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.


def _settings(**kw) -> Settings:
    kw.setdefault("accounts", [AccountConfig(email="a@x.com", token="tok")])
    kw.setdefault("auto_delete_session", False)
    return settings(**kw)


def _app(settings: Settings):
    """create_app + an explicitly injected pool (endpoint reuses it)."""
    app = create_app(settings)
    app.state.pool = AccountPool(settings.accounts, current=settings.active_account)
    return app


def _stub_session(monkeypatch) -> None:
    """Session create + challenge fetch succeed; no session cleanup traffic."""
    monkeypatch.setattr(CS, "_delete_session_bg", lambda *a, **k: None)
    monkeypatch.setattr(DS, "create_session", lambda h: "sess-1")
    monkeypatch.setattr(DS, "fetch_pow", lambda h: dict(CHALLENGE))
    monkeypatch.setattr("deepseaport.providers.deepseek.pow.solve", lambda *a, **k: 42)


def _boom(exc: BaseException):
    """A zero-arg callable that raises ``exc`` (recovery-failure stand-in)."""
    def _raise(*_a, **_k):
        raise exc
    return _raise


def _attempt_seq(*results):
    """``_attempt`` stand-in replaying ``results`` (last one repeats)."""
    calls = {"n": 0}

    def _attempt(*_a, **_k):
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[idx]

    _attempt.calls = calls  # type: ignore[attr-defined]
    return _attempt


def _post(app, **body):
    payload = {"model": "deepseek-flash",
               "messages": [{"role": "user", "content": "hi"}]}
    payload.update(body)
    # raise_server_exceptions=False: an unhandled engine exception must show
    # up as the 500 the client really gets, not as a test-level raise.
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post("/v1/chat/completions", json=payload)


# --- E1: a failing recovery must not become an unhandled 500 ----------------


def test_pow_recovery_failure_keeps_upstream_status_not_500(monkeypatch, tmp_path):
    """E1a. Pre-change: `_recover_pow` -> `_solve` raising RuntimeError escaped
    `drive_retries` and the whole request as a 500 "Internal Server Error";
    the client never learned the upstream reason (POW rejection)."""
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    # The retry's own work dies: `_fresh_pow` -> DS.solve_pow -> HTTP fetch_pow.
    monkeypatch.setattr(CS, "_solve", _boom(RuntimeError("fetch_pow: connection reset")))
    attempt = _attempt_seq(("", "", POW_ERROR, 0))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert "POW_HEADER_ERROR" in detail, detail
    assert "proof of work header is invalid" in detail, detail
    assert "fetch_pow" not in detail, "the recovery fault must not become the answer"
    assert attempt.calls["n"] == 1, \
        "a failed recovery re-attempts nothing; the original error is raised"


def test_waf_recovery_failure_keeps_upstream_status_not_500(monkeypatch, tmp_path):
    """E1b. Pre-change: a raising `_refresh_waf_without_account_lock` escaped
    the same way, so a WAF block answered 500 instead of the upstream 502."""
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_refresh_waf_without_account_lock",
                        _boom(RuntimeError("obscura warmup exploded")))
    monkeypatch.setattr(CS, "_solve", lambda a, token: {"Authorization": "Bearer x"})
    attempt = _attempt_seq(("", "", WAF_ERROR, 0))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert "HTTP_403_WAF" in detail, detail
    assert "obscura warmup" not in detail, \
        "the recovery fault must not replace the upstream reason"
    assert attempt.calls["n"] == 1


def test_recovery_that_succeeds_still_retries(monkeypatch, tmp_path):
    """E1c. The happy path is untouched: a working recovery re-attempts and
    the request succeeds. Guards against "contain everything" over-fixes."""
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    solved = {"n": 0}

    def _solve_ok(app_, token):
        solved["n"] += 1
        return {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(CS, "_solve", _solve_ok)
    attempt = _attempt_seq(("", "", POW_ERROR, 0), ("hi", "", None, 5))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hi"
    assert attempt.calls["n"] == 2
    assert solved["n"] == 1, "a successful recovery must solve a fresh PoW header"


def test_drive_retries_lets_http_exception_through():
    """E1d. A recovery may raise a deliberate status (the ban 403 the
    invalid-token path raises after a positive probe). Containing it would
    downgrade a real 403 ban to the generic upstream 502, so HTTPException
    must still escape `drive_retries`."""
    calls = {"n": 0}

    def _recover():
        calls["n"] += 1
        raise HTTPException(status_code=403, detail="Account a@x.com banned.")

    with pytest.raises(HTTPException) as exc_info:
        _engine.drive_retries(
            2, lambda: False, lambda: "5: user is muted",
            (("ban", lambda e: True, _recover),),
        )
    assert exc_info.value.status_code == 403
    assert calls["n"] == 1


def test_drive_retries_contains_non_status_recovery_failure(caplog):
    """E1e. Unit-level: a RuntimeError from a recovery is logged, ends that
    kind, and leaves the ORIGINAL error for the remaining handlers."""
    overload = "generation_timeout: Server busy, please try again later."
    order: list[str] = []

    def _recover_overload():
        order.append("overload-recover")
        raise RuntimeError("pow solver died")

    def _should_retry(name):
        def _pred(e):
            order.append(f"pred:{name}:{e}")
            return e == overload and name == "overload"
        return _pred

    with caplog.at_level("WARNING", logger="deepseaport.core.engine"):
        _engine.drive_retries(
            3, lambda: False, lambda: overload,
            (
                ("overload", _should_retry("overload"), _recover_overload),
                ("later", _should_retry("later"), lambda: order.append("later-recover")),
            ),
        )
    assert order == ["pred:overload:" + overload, "overload-recover",
                     "pred:later:" + overload]
    assert "later-recover" not in order, "a failed recovery must not be retried"
    assert "pow solver died" in caplog.text


# --- E2: app_settings must not adopt the operator config --------------------


class _BareState:
    """Stand-in for a hand-built app state with no settings."""


class _BareApp:
    def __init__(self) -> None:
        self.state = _BareState()


def _operator_config(tmp_path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "keys": ["sk-operator"],
        "accounts": [{"email": "a@x.com", "token": "tok-operator"}],
        "active_account": "a@x.com",
    }), encoding="utf-8")
    return path


def test_app_settings_fallback_does_not_memoize_onto_app_state(monkeypatch, tmp_path):
    """E2. Pre-change: the fallback assigned `app.state.settings`, so the
    hand-built app ended up holding a Settings bound to the on-disk
    config.json that any later provider `save()` rewrote."""
    path = _operator_config(tmp_path)
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(path))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    app = _BareApp()

    first = _engine.app_settings(app)

    assert isinstance(first, Settings)
    assert first.config_path == str(path)
    assert not hasattr(app.state, "settings"), \
        "fallback settings were memoized onto app.state"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before, \
        "reading settings must not write the config file"

    second = _engine.app_settings(app)
    assert second is not first, "the fallback must not be memoized"
    assert not hasattr(app.state, "settings")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    # app=None tolerance is part of the frozen contract.
    assert isinstance(_engine.app_settings(None), Settings)


def test_ban_persist_path_leaves_bare_app_without_operator_settings(monkeypatch, tmp_path):
    """E2b. Same leak through the real caller: the ban 403 persists, but the
    operator-bound Settings must not stay attached to the app afterwards."""
    from deepseaport.config import load_settings

    path = _operator_config(tmp_path)
    monkeypatch.setenv("DEEPSEAPORT_CONFIG", str(path))
    app = _BareApp()
    item = PooledAccount(cfg=load_settings(str(path)).accounts[0])
    until = time.time() + 86400

    exc = _engine.ban_403_for_item(app, item, until, "user is muted")

    assert exc.status_code == 403
    assert item.cfg.banned is True
    assert not hasattr(app.state, "settings"), \
        "the ban save path left the operator config bound to the app"


# --- E3: a bare 401 is auth, not a ban --------------------------------------


def test_bare_401_classifies_as_invalid_token_not_ban():
    """E3. Pre-change: `any(k in low for k in (..., "401"))` classified a bare
    HTTP_401 as RetryKind.BAN, which persists `banned: true` and answers 403."""
    assert CS.classify_error(BARE_401) is CS.RetryKind.INVALID_TOKEN
    assert CS.classify_error("Unauthorized") is CS.RetryKind.INVALID_TOKEN
    # Unchanged neighbours.
    assert CS.classify_error("40003: Authorization Failed (invalid token)") \
        is CS.RetryKind.INVALID_TOKEN
    assert CS.classify_error(BAN_ERROR) is CS.RetryKind.BAN
    assert CS.classify_error("account restricted by policy") is CS.RetryKind.BAN
    assert CS.classify_error(WAF_ERROR) is CS.RetryKind.WAF
    assert CS.classify_error("rate_limit_reached: too frequent") \
        is CS.RetryKind.RATE_LIMIT


def test_bare_401_does_not_mark_account_banned_or_write_config(monkeypatch, tmp_path):
    """E3b. Pre-change: one transient 401 marked the healthy account
    `banned: true` in config.json and answered 403 with ban wording."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "keys": [],
        "accounts": [{"email": "a@x.com", "token": "tok-401-abcdefghijklmn"}],
        "active_account": "a@x.com",
    }), encoding="utf-8")
    settings = _settings(
        accounts=[AccountConfig(email="a@x.com", token="tok-401-abcdefghijklmn")],
        active_account="a@x.com", config_path=str(path), max_retries=1)
    app = _app(settings)
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_probe_ban", lambda *a, **k: (False, None))
    attempt = _attempt_seq(("", "", BARE_401, 0))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 401, resp.text
    assert "token invalid" in resp.json()["detail"].lower()
    assert attempt.calls["n"] == 1, \
        "no stored password -> the one-shot refresh is a no-op, nothing re-attempts"

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert not on_disk["accounts"][0].get("banned"), \
        "a bare 401 persisted banned=true into config.json"
    item = app.state.pool.get("a@x.com")
    assert item.cfg.banned is False
    assert item.cooldown_remaining == 0, "an unconfirmed 401 must not cool the account"


def test_401_with_banned_probe_is_still_marked_banned(monkeypatch, tmp_path):
    """E3c. Routing 401 to INVALID_TOKEN must not lose a real ban: the terminal
    handler probes first and a positive verdict still becomes 403 + cooldown."""
    until = time.time() + 86400
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_probe_ban", lambda *a, **k: (True, until))
    attempt = _attempt_seq(("", "", BARE_401, 0))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 403, resp.text
    assert "banned" in resp.json()["detail"].lower()
    item = app.state.pool.get("a@x.com")
    assert item.cfg.banned is True
    assert item.cfg.banned_until == until
    assert item.cooldown_remaining > 3600


def test_genuine_ban_text_still_403_and_marks_banned(monkeypatch, tmp_path):
    """E3d. Real ban evidence (biz 5 muted) keeps the 403 + persisted ban."""
    until = time.time() + 86400
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_probe_ban", lambda *a, **k: (True, until))
    attempt = _attempt_seq(("", "", BAN_ERROR, 0))
    monkeypatch.setattr(CS, "_attempt", attempt)

    resp = _post(app)

    assert resp.status_code == 403, resp.text
    assert "banned" in resp.json()["detail"].lower()
    item = app.state.pool.get("a@x.com")
    assert item.cfg.banned is True
    assert item.cfg.banned_until == until


def test_terminal_ban_ignores_timestamp_from_a_negative_probe(monkeypatch, tmp_path):
    """E3e. Pre-change: `_, ban_until = _probe_ban(...)` adopted a timestamp
    from a verdict of False, cooling the account down as if it were muted
    until an expiry the probe had just denied."""
    stray_until = time.time() + 7 * 86400
    app = _app(_settings(max_retries=0))
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))
    monkeypatch.setattr(CS, "_probe_ban", lambda *a, **k: (False, stray_until))
    prep = _prepare({"model": "deepseek-flash",
                     "messages": [{"role": "user", "content": "hi"}]})

    with pytest.raises(HTTPException) as exc_info:
        CS._terminal_ban(app, item, prep, "tok", BAN_ERROR, "", None)

    assert exc_info.value.status_code == 403
    assert item.cfg.banned is True
    assert item.cfg.banned_until == 0.0, \
        "a negative probe verdict must not pin the probe's timestamp"
    assert item.cooldown_remaining <= 300, \
        "negative probe must not apply a multi-day cooldown"


def test_qwen_terminal_ban_ignores_timestamp_from_a_negative_probe(monkeypatch):
    """E3e (Qwen twin). The DeepSeek fix above landed in DeepSeek only; Qwen
    kept ``_, ban_until = _probe_ban(...)`` and adopted the timestamp from a
    False verdict, cooling a healthy account for the stray expiry."""
    stray_until = time.time() + 7 * 86400
    app = _app(_settings(max_retries=0))
    item = PooledAccount(cfg=AccountConfig(email="a@x.com", token="tok"))
    monkeypatch.setattr(QCS, "_probe_ban", lambda *a, **k: (False, stray_until))

    with pytest.raises(HTTPException) as exc_info:
        QCS._terminal_ban(app, item, {}, "tok", BAN_ERROR, "", None)

    assert exc_info.value.status_code == 403
    assert item.cfg.banned is True
    assert item.cfg.banned_until == 0.0, \
        "a negative probe verdict must not pin the probe's timestamp"
    assert item.cooldown_remaining <= 300, \
        "negative probe must not apply a multi-day cooldown"


def test_every_registered_provider_classifies_the_shared_status_shapes():
    """E3f. ``transport.http_error_code`` renders every non-200 as
    "HTTP_<status>: <body>" on every upstream, so a bare status text must
    classify the same way everywhere. The marker tuples live in
    ``core.engine`` and each provider composes them with its own extras;
    derived from ``registry.registered()`` rather than a literal list so this
    check cannot silently miss a provider that ships a divergent classifier.
    """
    import importlib

    from deepseaport.core import registry

    cases = {
        "HTTP_502: ": "OVERLOAD",
        "HTTP_500: Internal Server Error": "OVERLOAD",
        "429 too many requests": "RATE_LIMIT",
        "HTTP_422: ": "UPSTREAM_INVALID",
    }
    provider_ids = sorted(registry.registered())
    assert provider_ids, "registry imported no providers"
    for pid in provider_ids:
        mod = importlib.import_module(
            f"deepseaport.providers.{pid.replace('-', '_')}.completion_service")
        for text, kind_name in cases.items():
            got = mod.classify_error(text)
            assert got is getattr(mod.RetryKind, kind_name), (
                f"{pid}: {text!r} classified {got}, want {kind_name}")


# --- E5: buffered watchdog vs the real producer budget ----------------------


def _stream_settings(**kw) -> Settings:
    kw.setdefault("stream_mode", "buffered")
    kw.setdefault("max_retries", 0)
    kw.setdefault("use_multiple_accounts", False)
    return _settings(**kw)


def test_buffered_watchdog_covers_the_whole_producer(monkeypatch, tmp_path):
    """E5a. Pre-change: consumer_timeout was ONE read (stream_timeout+grace),
    so a buffered producer slower than that was killed with
    {"error": {"message": "upstream timed out"}} and its answer discarded.

    The delay is DERIVED from the read budget rather than written as its own
    literal, because only the ratio is meaningful: the producer has to be
    slower than the OLD one-read watchdog (so the regression is visible) and
    faster than the new 3x one (so the fix is visible). 1.2x of the read puts
    it 0.2x above the old budget and 1.8x below the new -- i.e. ~0.1s and
    ~0.9s of absolute headroom.

    The scale is deliberately NOT shrunk. An earlier revision ran this at
    0.1s/0.12s, which kept the ratio and left 0.02s of headroom -- less than a
    loaded runner's overshoot on a single `time.sleep`, i.e. strictly MORE
    flaky, not less. Cutting the absolute time here buys ~0.4s and spends the
    margin that makes the test safe on `windows-latest`. The 30s in E5b was
    worth paying because it pinned a non-daemon thread; this 0.6s does not.
    """
    read = _WATCHDOG_READ_SECONDS
    producer_seconds = 1.2 * read  # > old watchdog (1x read), < new one (3x read)
    # 1x read + 0 grace = old watchdog; the new budget is 3x.
    monkeypatch.setattr(DS, "DEFAULT_TIMEOUT", read)
    monkeypatch.setattr(CS, "_CONSUMER_GRACE_SECONDS", 0.0)

    app = _app(_stream_settings())
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_solve", lambda a, token: {"Authorization": "Bearer x"})

    def _slow_attempt(*_a, **_k):
        time.sleep(producer_seconds)
        return "slow but valid", "", None, 7

    monkeypatch.setattr(CS, "_attempt", _slow_attempt)
    item = app.state.pool.acquire(timeout=1)
    prep = _prepare({"model": "deepseek-flash", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})

    frames = asyncio.run(collect_sse(CS, app, app.state.pool, item, prep))
    body = sse_text(frames)

    assert "upstream timed out" not in body, \
        "buffered watchdog is still sized from one socket read: " + body
    assert "slow but valid" in body, "a valid buffered answer was discarded: " + body
    assert body.rstrip().endswith("data: [DONE]")

    # The live budget stays one read; the buffered one covers setup + a
    # recovery pass. Pinned directly so the multiplier itself is asserted, not
    # just inferred from the sleep surviving.
    assert _engine._BUFFERED_PRODUCER_READS == 3
    assert _engine.new_stream_context(read, 0.0)[4] == pytest.approx(read)
    assert _engine.new_stream_context(read, 0.0, buffered=True)[4] \
        == pytest.approx(3 * read)


def test_silent_buffered_producer_still_times_out(monkeypatch, tmp_path):
    """E5b. The bigger budget must not disable the watchdog: a producer that
    never speaks still gets the existing error frame + [DONE] sentinel.

    The fake used to be `time.sleep(30)`. That is the single worst time sink in
    the suite: `_attempt` runs on the shared `_STREAM_EXECUTOR`, whose
    ThreadPoolExecutor threads are non-daemon, so a 30s park is charged to the
    *process* at interpreter exit. pytest reports this test as fast (~0.3s)
    while the run really costs 30s -- which is exactly the kind of gap a second
    CI matrix leg inherits.

    It is a park on an Event, not a sleep, so the producer's duration stops
    being a tunable constant at all: the watchdog must fire while the producer
    is silent, and the test releases it the instant `_collect` returns. The
    only thing the numbers below still have to get right is that the producer
    is silent for LONGER than the budget -- 0.05s read x 3 = 0.15s -- which is
    now guaranteed by construction (release happens strictly after the watchdog
    fired) instead of by a 30s cushion. The park bound is a hang guard only.
    """
    read = _WATCHDOG_READ_SECONDS / 5  # 0.05s: buffered budget = 3x = 0.15s
    monkeypatch.setattr(DS, "DEFAULT_TIMEOUT", read)
    monkeypatch.setattr(CS, "_CONSUMER_GRACE_SECONDS", 0.0)

    app = _app(_stream_settings())
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_solve", lambda a, token: {"Authorization": "Bearer x"})
    release = threading.Event()
    monkeypatch.setattr(CS, "_attempt",
                        lambda *_a, **_k: release.wait(_SILENT_PRODUCER_PARK_SECONDS))
    item = app.state.pool.acquire(timeout=1)
    prep = _prepare({"model": "deepseek-flash", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})

    frames = asyncio.run(collect_sse(CS, app, app.state.pool, item, prep))
    # Safe now, not before: the watchdog has fired (that is what returned us),
    # so the parked worker cannot complete first and the assertion below cannot
    # be raced by a producer that finished in time.
    release.set()
    body = sse_text(frames)

    assert '"upstream timed out"' in body, body
    assert body.count("data: [DONE]") == 1, body
    assert body.rstrip().endswith("data: [DONE]")


def test_live_mode_watchdog_is_unchanged(monkeypatch, tmp_path):
    """E5c. Live mode keeps the single-read budget byte for byte: deltas reset
    the per-get clock, so only genuine silence should be bounded."""
    monkeypatch.setattr(DS, "DEFAULT_TIMEOUT", 0.1)
    monkeypatch.setattr(CS, "_CONSUMER_GRACE_SECONDS", 5.0)

    app = _app(_stream_settings(stream_mode="live"))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    monkeypatch.setattr(CS, "_solve", lambda a, token: {"Authorization": "Bearer x"})
    monkeypatch.setattr(CS, "_attempt", lambda *_a, **_k: ("live", "", None, 1))
    item = app.state.pool.acquire(timeout=1)
    prep = _prepare({"model": "deepseek-flash", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})

    seen: list[float] = []
    real = _engine.new_stream_context

    def _spy(timeout, grace=10.0, **kw):
        result = real(timeout, grace, **kw)
        seen.append((bool(kw.get("buffered")), result[4]))
        return result

    monkeypatch.setattr(_engine, "new_stream_context", _spy)
    frames = asyncio.run(collect_sse(CS, app, app.state.pool, item, prep))

    assert seen == [(False, pytest.approx(5.1))], seen
    body = sse_text(frames)
    assert "upstream timed out" not in body
    assert body.rstrip().endswith("data: [DONE]")


# --- E7: the first-payload wait covers producer setup in live mode -----------


class _DelayedQueue:
    """``asyncio.Queue`` stand-in with a per-call (delay, item) script.

    The last script entry repeats, so a long delay on the tail models a queue
    that never speaks again; ``wait_for`` cancels the sleep at the budget,
    which is exactly what the watchdog is for. Delays only have to sit on the
    right side of the budget -- no wall-clock equality is asserted anywhere.
    """

    def __init__(self, script: list[tuple[float, tuple]]) -> None:
        self.calls = 0
        self._script = list(script)

    async def get(self):
        delay, item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        await asyncio.sleep(delay)
        return item


async def _drain_q(q, timeout: float) -> list[bytes]:
    """Run the real shared consumer (live) over a queue stand-in."""
    return [f async for f in _engine.drain_stream_queue(
        q, {}, True, "cid", 0, "m", timeout)]


def test_live_first_payload_wait_covers_producer_setup():
    """E7a. Pre-change: every live get waited ONE read. Nothing resets the
    clock before the first delta, yet the producer is doing the same setup +
    recovery work buffered mode is credited 3x for (``new_stream_context``'s
    own note: a ~180s browser login refresh plus a 90s login can exceed the
    single-read budget), so a healthy-but-slow live stream was killed with
    "upstream timed out" and its answer discarded.

    timeout 0.25s -> old budget 0.25s, new first-payload budget 0.75s; the
    first item lands at 1.5x (0.375s), strictly between them with a wide
    margin on both sides. The second (done) item is immediate, so the fix must
    not slow the happy path.
    """
    timeout = 0.25
    q = _DelayedQueue([(0.375, ("content", "late")), (0.0, ("done", {}))])
    body = sse_text(asyncio.run(_drain_q(q, timeout)))

    assert "late" in body, \
        "watchdog killed a live stream before its first delta: " + body
    assert "upstream timed out" not in body, body
    assert body.count("data: [DONE]") == 1, body
    assert q.calls == 2, q.calls


def test_live_watchdog_still_fires_after_the_first_payload():
    """E7b. The bigger first wait must not leak into the rest of the stream:
    once a payload has landed, only genuine silence is bounded and the timeout
    path stays the old one (error frame + exactly one [DONE] + return).

    First item immediate, second wait never resolves: the frame must arrive at
    the PLAIN budget, and the 30s park must never elapse.
    """
    timeout = 0.2
    q = _DelayedQueue([(0.0, ("content", "x")), (30.0, ("done", {}))])
    started = time.monotonic()
    body = sse_text(asyncio.run(_drain_q(q, timeout)))
    elapsed = time.monotonic() - started

    assert "x" in body, body
    assert '"upstream timed out"' in body, body
    assert body.count("data: [DONE]") == 1, body
    assert body.rstrip().endswith("data: [DONE]")
    # Hang guard for the parked second wait, not a timing assertion.
    assert elapsed < 5.0, elapsed


# --- E6: no retry row fires once a live delta reached the client -------------


def _content_then_error(text, error):
    """``_attempt`` stand-in: content THEN error.

    That ordering is legal, not exotic: ``drain_attempt`` ``append``s content
    events and only breaks on the error event, so a mid-stream failure always
    arrives with a content prefix already emitted. It is what makes the
    pre-change pow/session/token/waf rows unsafe.
    """
    def _attempt(_h, _p, on_content=None, on_thinking=None, **_k):
        if on_content is not None:
            on_content(text)
        return text, "", error, 0

    _attempt.calls = {"n": 0}

    def _counted(*a, **k):
        _attempt.calls["n"] += 1
        return _attempt(*a, **k)

    _counted.calls = _attempt.calls  # type: ignore[attr-defined]
    return _counted


def _stub_recovery_work(monkeypatch):
    """Every seam a DeepSeek recovery reaches, benign, so the only thing that
    can stop one is the ``emitted`` guard in the retry table."""
    monkeypatch.setattr(CS, "_solve", lambda a, token: {"Authorization": "Bearer x"})
    monkeypatch.setattr(CS, "_refresh_waf_without_account_lock", lambda *a, **k: None)
    # The one-shot Obscura refresh, so INVALID_TOKEN's recovery gets as far as
    # recreating the session and re-attempting. Without it that row would no-op
    # and the test would pass for the wrong reason.
    monkeypatch.setattr(CS, "_refresh_token_via_obscura",
                        lambda *a, **k: ("tok-new", False, ""))
    monkeypatch.setattr(CS, "_probe_ban", lambda *a, **k: (False, None))


# One case per retry row, in table order. ``overload`` is included because it
# is the row that already carried the guard: this pins the whole table, not
# just the rows that were broken.
_ALL_KINDS = [POW_ERROR, SESSION_ERROR, BARE_401, WAF_ERROR, OVERLOAD_ERROR]


@pytest.mark.parametrize("error_text", _ALL_KINDS,
                         ids=["pow", "session", "token", "waf", "overload"])
def test_no_deepseek_retry_row_re_attempts_after_a_partial_live_emit(
        monkeypatch, tmp_path, error_text):
    """E6. Pre-change: only the overload row carried
    ``and not _emit_tracker.emitted``, on the assumption that the other kinds
    fail before any content. ``_content_then_error`` breaks that assumption.

    The completion POST is not idempotent -- DeepSeek appended a turn -- and
    the re-stream restarts at delta 0 into a stream the client already read, so
    the client ends up with duplicated text, or two interleaved halves of one
    ``tool_calls`` JSON object, inside a 200 SSE stream that then terminates
    with a clean ``[DONE]``. There is no error in that stream to tell a harness
    not to execute the corrupt call.

    ``_attempt`` count is the assertion because it IS the harm: one attempt is
    one POST, and a second one is the duplicate the client would read.
    ``test_recovery_that_succeeds_still_retries`` (E1c) is the control for the
    other half -- with nothing emitted, the same table does re-attempt.
    """
    app = _app(_settings(max_retries=1))
    cached_wasm(monkeypatch, tmp_path)
    _stub_session(monkeypatch)
    _stub_recovery_work(monkeypatch)
    attempt = _content_then_error("partial ", error_text)
    monkeypatch.setattr(CS, "_attempt", attempt)
    item = app.state.pool.acquire(timeout=1)
    prep = _prepare({"model": "deepseek-flash", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})

    with pytest.raises(HTTPException):
        CS._run_completion_core(app, item, prep, on_content=lambda _t: None)

    assert attempt.calls["n"] == 1, (
        f"{error_text!r} re-POSTed after output already reached the client: "
        f"{attempt.calls['n']} attempts, so the client reads a duplicate")


@pytest.mark.parametrize("error_text",
                         ["WAF_CHALLENGE: aliyun_waf_aa challenge",
                          "401 Unauthorized"],
                         ids=["waf", "invalid_token"])
def test_no_qwen_retry_row_re_attempts_after_a_partial_live_emit(
        monkeypatch, error_text):
    """E6, Qwen's copy of the same table. It carried the guard on ``overload``
    only, so a WAF challenge or a 401 arriving after a delta re-POSTed on the
    same chat. ``overload`` is not repeated here: it is one row of one factory
    and DeepSeek's case above already pins that factory's shared guard."""
    from deepseaport.providers.qwen import client as QC
    from deepseaport.providers.qwen import completion_service as QCS

    st = settings(provider="qwen", max_retries=1,
                  accounts=[AccountConfig(email="a@x.com", token="tok")])
    app = create_app(st)
    app.state.pool = AccountPool(st.accounts, current=st.active_account)
    monkeypatch.setattr(QCS, "_headers",
                        lambda _a, token: {"Authorization": f"Bearer {token}",
                                           "Cookie": f"token={token}"})
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)
    monkeypatch.setattr(QCS, "_refresh_waf_without_account_lock",
                        lambda *a, **k: None)
    monkeypatch.setattr(QCS, "_probe_ban", lambda *a, **k: (False, None))
    ids = iter(["chat-1", "chat-2"])
    monkeypatch.setattr(QC, "create_chat", lambda _h, _m, **_kw: next(ids))
    attempt = _content_then_error("partial ", error_text)
    monkeypatch.setattr(QCS, "_attempt", attempt)
    prep = _prepare({"model": "qwen3.7-plus", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})
    item = app.state.pool.acquire(timeout=1)

    with pytest.raises(HTTPException):
        QCS._run_completion_core(app, item, prep, on_content=lambda _t: None)

    assert attempt.calls["n"] == 1, (
        f"{error_text!r} re-POSTed after output already reached the client: "
        f"{attempt.calls['n']} attempts, so the client reads a duplicate")


# --- allow_failover=False: one attempt, one release, no acquire -------------


class _RecordingPool:
    """``len`` > 1 so an ignored single-account flag WOULD try to fail over."""

    def __init__(self) -> None:
        self.acquires: list[float] = []

    def __len__(self) -> int:
        return 2

    def acquire(self, timeout):
        self.acquires.append(timeout)
        return object()


def test_allow_failover_false_releases_once_and_reraises():
    """Pre-change this branch had no engine-level test: the only tests of the
    flag mock ``_complete_with_failover_sync`` away, so nothing pinned that a
    retryable error on the pinned account still releases its slot and
    propagates instead of failing over.

    The acquire recorder is the discriminating assertion: if the flag were
    ignored, a 2-account pool would acquire and re-run ``_run_once`` before
    giving up, and both the calls and the released list would show the second
    item.
    """
    item = object()
    pool = _RecordingPool()
    released: list[object] = []
    calls: list[object] = []

    def _run_once(account):
        calls.append(account)
        raise HTTPException(status_code=503, detail="overloaded")

    with pytest.raises(HTTPException) as exc_info:
        _engine.run_with_failover(
            pool, item, _run_once,
            is_retryable_exc=lambda _e: True,
            acquire=pool.acquire,
            allow_failover=False,
            release=released.append,
        )

    assert exc_info.value.status_code == 503
    assert calls == [item], "allow_failover=False runs the pinned item once"
    assert released == [item], "release must fire exactly once with the item"
    assert pool.acquires == [], "allow_failover=False must never acquire"


def test_allow_failover_false_releases_once_on_success():
    """The success path of the same branch: one release, no acquire."""
    item = object()
    pool = _RecordingPool()
    released: list[object] = []

    result = _engine.run_with_failover(
        pool, item, lambda _account: "ok",
        is_retryable_exc=lambda _e: True,
        acquire=pool.acquire,
        allow_failover=False,
        release=released.append,
    )

    assert result == "ok"
    assert released == [item], "release must fire exactly once with the item"
    assert pool.acquires == [], "allow_failover=False must never acquire"
