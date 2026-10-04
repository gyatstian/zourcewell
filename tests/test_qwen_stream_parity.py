"""Parity tests for the qwen stream adapter (``_stream_completion``).

``providers/qwen/completion_service.py`` had ZERO coverage of its streaming
path while every agentic-harness request with ``stream: true`` and a qwen
model runs through it. Three defects these tests pin, all of which surface as
"the tool stopped working" rather than as a diagnosable error:

* Q1 the buffered watchdog was sized from ONE socket read.
  ``ARCHITECTURE.md`` ("Cross-provider contract that every adapter must keep",
  and the "``stream_mode: buffered`` sizes its watchdog" rule) requires
  ``new_stream_context(..., buffered=True)`` so the consumer watchdog covers
  the whole producer budget; Qwen passed nothing, so a valid-but-slow answer
  was thrown away and replaced with a client-visible ``upstream timed out``.
  DeepSeek complies at ``providers/deepseek/completion_service.py:1154-1156``.
* Q2 the terminal 401 quoted the raw upstream body, and the qwen credential IS
  a JWT, whose dot-framed segments no run-length redactor can see.
* Q3 every WAF/401 recovery rebound ``chat_id`` without deleting the previous
  one, stranding one chat session per pass upstream.

Fixture idioms are taken from ``tests/test_engine_retry_gaps.py`` (E5, the
DeepSeek counterpart of Q1) and ``tests/test_qwen_provider_gaps.py``. All
offline: no network, no credentials, no browser.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import inspect
import json
import textwrap
import threading
import time

import pytest
from fastapi import HTTPException

from support import collect_sse, settings, sse_payloads, sse_text
from deepseaport.accounts import AccountPool
from deepseaport.config import AccountConfig, Settings
from deepseaport.core import engine as _engine
from deepseaport.providers.qwen import auth as QA
from deepseaport.providers.qwen import client as QC
from deepseaport.providers.qwen import completion_service as QCS
from deepseaport.providers.qwen.sse_parser import StreamEvent
from deepseaport.server import create_app
from deepseaport.server_routes import _prepare

# The qwen credential shape: chat.qwen.ai localStorage ``token`` is a JWT (see
# providers/qwen/README.md "Auth"), so an upstream 401 that echoes the request
# headers echoes a JWT. Only the header/payload are derived; the signature is a
# fixed literal so the string is byte-identical on every run and can be
# asserted on directly.
_JWT_SIG = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


LIVE_JWT = (f"{_b64({'alg': 'HS256', 'typ': 'JWT'})}."
            f"{_b64({'sub': 'u-1', 'exp': 1789555791})}.{_JWT_SIG}")

# The OTHER credential shape: one long unbroken run of token characters (what
# DeepSeek's 64-char userToken is). ``core.engine`` needs both regexes for
# exactly this reason -- ``_OPAQUE_RUN_RE`` catches this one, ``_SEGMENT_FRAMED_RE``
# catches the JWT above.
LIVE_RUN = "aB3dEfGhIjKlMnOpQrStUvWxYz0123456789cDeFgHiJkLmNoP"


# --- shared fixtures/helpers (mirrors tests/test_engine_retry_gaps.py) -------


# Default kwargs differ per file; the shared base dict lives in
# support.py. Every list here is rebuilt per call on purpose.


def _settings(**kw) -> Settings:
    kw.setdefault("accounts", [AccountConfig(email="a@x.com", token="tok")])
    kw.setdefault("max_retries", 0)
    kw.setdefault("use_multiple_accounts", False)
    kw.setdefault("provider", "qwen")
    return settings(**kw)


def _app(settings: Settings):
    """create_app + an explicitly injected pool (Qwen's adapter takes it)."""
    app = create_app(settings)
    app.state.pool = AccountPool(settings.accounts, current=settings.active_account)
    return app


def _prep(**kw) -> dict:
    body = {"model": "qwen3.7-plus", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return _prepare(body)


def _fake_headers(_app, token):
    """Bridge-free headers; also what the cleanup ``finally`` builds."""
    return {"Authorization": f"Bearer {token}", "Cookie": f"token={token}"}


def _item(**cfg) -> AccountPool:
    """A single-account item with its slot already held (as the route does)."""
    return AccountPool([AccountConfig(**cfg)]).acquire(timeout=1)


def _stub(monkeypatch, events_fn, *, chat_ids=("chat-1",)):
    """Qwen's two chat-lifecycle seams, with no traffic.

    ``_delete_chat_bg`` is stubbed BY NAME (the existing qwen suite does the
    same) so this file never depends on the background cleanup executor.
    """
    ids = iter(list(chat_ids))
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QC, "create_chat", lambda _h, _m, **_kw: next(ids))
    monkeypatch.setattr(QC, "stream_completion", events_fn)
    monkeypatch.setattr(QCS, "_delete_chat_bg", lambda *a, **k: None)


def _record_deletes(monkeypatch) -> list:
    """Same seam, but recording what the adapter asked to delete."""
    deleted: list[tuple[dict, str, bool]] = []
    monkeypatch.setattr(
        QCS, "_delete_chat_bg",
        lambda h, cid, enabled=True: deleted.append((dict(h), cid, enabled)))
    return deleted


def _attempts(monkeypatch, script) -> list[dict]:
    """Fake ``stream_completion`` replaying one event list per attempt.

    Counted on ATTEMPTS, not on chat creates: a chat is created before the
    first attempt, so create-count is always one ahead of it. Returns the
    per-attempt header dicts, which is how "the retry used the refreshed
    token" is asserted.
    """
    state = {"n": 0}
    calls: list[dict] = []

    def _events(headers, payload, **_kw):
        n = state["n"]
        state["n"] += 1
        calls.append(headers)
        for ev in script[min(n, len(script) - 1)]:
            yield ev

    monkeypatch.setattr(QC, "stream_completion", _events)
    return calls


# --- Q1: the happy path and the watchdog budget ------------------------------


def test_buffered_happy_path_emits_deltas_then_one_done(monkeypatch):
    """Tools OFF, buffered: the whole reply lands, one [DONE], finish stop.

    This is the frame contract an agentic harness reads; ``finish_reason`` and
    the single terminator are what a broken tail gets wrong.
    """
    _stub(monkeypatch, _fake_events([
        StreamEvent(kind="thinking", text="hmm"),
        StreamEvent(kind="content", text="Hello"),
        StreamEvent(kind="content", text=" world"),
        StreamEvent(kind="finished", text=""),
    ]))
    app = _app(_settings(stream_mode="buffered"))
    item = app.state.pool.acquire(timeout=1)

    body = sse_text(asyncio.run(collect_sse(QCS, app, app.state.pool, item, _prep())))

    assert body.count("data: [DONE]") == 1, body
    assert body.rstrip().endswith("data: [DONE]")
    payloads = sse_payloads(body)
    assert all(p["model"] == "qwen3.7-plus" for p in payloads)
    assert all(p["id"].startswith("chatcmpl-") for p in payloads)
    deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
    assert {"role": "assistant"} in deltas
    assert "".join(d.get("content", "") for d in deltas) == "Hello world"
    # Buffered mode ships the thinking tail as its own delta before content.
    assert any(d.get("reasoning_content") == "hmm" for d in deltas)
    finishes = [p["choices"][0]["finish_reason"] for p in payloads if p["choices"]]
    assert finishes[:-1] == [None] * (len(finishes) - 1), finishes
    assert finishes[-1] == "stop", finishes


def _fake_events(events):
    def _events_fn(_h, _p, **_kw):
        yield from events
    return _events_fn


def test_buffered_mode_passes_buffered_true_to_the_watchdog(monkeypatch):
    """ARCHITECTURE.md's cross-provider contract, asserted on the call.

    The spy is ``_engine.new_stream_context`` -- the function
    ``new_producer_context`` forwards ``buffered`` to -- and it records the
    kwarg plus the budget it produced. Same shape as
    ``test_engine_retry_gaps.test_live_mode_watchdog_is_unchanged``, which pins
    the DeepSeek call site; timings are at the 0.5s scale for the reason that
    test gives (the ratio has to hold against a loaded CI runner).
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_CONSUMER_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(QC, "DEFAULT_TIMEOUT", 0.5)
    _stub(monkeypatch, _fake_events([
        StreamEvent(kind="content", text="ok"),
        StreamEvent(kind="finished", text=""),
    ]))
    app = _app(_settings(stream_mode="buffered"))
    item = app.state.pool.acquire(timeout=1)

    seen: list[tuple[bool, float]] = []
    real = _engine.new_stream_context

    def _spy(timeout, grace=10.0, **kw):
        result = real(timeout, grace, **kw)
        seen.append((bool(kw.get("buffered")), result[4]))
        return result

    monkeypatch.setattr(_engine, "new_stream_context", _spy)
    body = sse_text(asyncio.run(collect_sse(QCS, app, app.state.pool, item, _prep())))

    # True, and the budget is the 3x whole-producer multiple: 0.5s for the
    # single-read watchdog, 1.5s here. This is the exact line Qwen omitted.
    assert seen == [(True, pytest.approx(1.5))], seen
    assert _engine._BUFFERED_PRODUCER_READS == 3
    assert "upstream timed out" not in body
    assert "ok" in body


def test_live_mode_keeps_the_single_read_watchdog(monkeypatch):
    """The mirror: live must pass buffered=False.

    Every delta resets the per-``get`` clock, so one read is the only honest
    bound in live mode; multiplying it there would leave a genuinely dead
    producer hanging for 3x before the client is told anything.
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_CONSUMER_GRACE_SECONDS", 5.0)
    monkeypatch.setattr(QC, "DEFAULT_TIMEOUT", 0.1)
    _stub(monkeypatch, _fake_events([
        StreamEvent(kind="content", text="live"),
        StreamEvent(kind="finished", text=""),
    ]))
    app = _app(_settings(stream_mode="live"))
    item = app.state.pool.acquire(timeout=1)

    seen: list[tuple[bool, float]] = []
    real = _engine.new_stream_context

    def _spy(timeout, grace=10.0, **kw):
        result = real(timeout, grace, **kw)
        seen.append((bool(kw.get("buffered")), result[4]))
        return result

    monkeypatch.setattr(_engine, "new_stream_context", _spy)
    body = sse_text(asyncio.run(collect_sse(QCS, app, app.state.pool, item, _prep())))

    assert seen == [(False, pytest.approx(5.1))], seen
    assert body.rstrip().endswith("data: [DONE]")


def test_silent_producer_still_times_out_with_the_bigger_budget(monkeypatch):
    """Q1 must not have DISABLED the watchdog (test_engine_retry_gaps E5b).

    A producer that never speaks still gets the error frame plus exactly one
    [DONE] sentinel; otherwise "buffered=True" would only be a 3x hang.
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_CONSUMER_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(QC, "DEFAULT_TIMEOUT", 0.1)
    # Parked, not sleeping: the wait is bounded (a leak here can never hang the
    # suite) and released in the finally, so the parked producer thread unwinds
    # immediately instead of holding the interpreter open until exit the way a
    # bare time.sleep(30) does.
    released = threading.Event()

    def _silent(_h, _p, **_kw):
        released.wait(30)
        yield StreamEvent(kind="content", text="never")

    _stub(monkeypatch, _silent)
    app = _app(_settings(stream_mode="buffered"))
    item = app.state.pool.acquire(timeout=1)

    try:
        body = sse_text(asyncio.run(collect_sse(QCS, app, app.state.pool, item, _prep())))
    finally:
        released.set()

    assert '"upstream timed out"' in body, body
    assert body.count("data: [DONE]") == 1, body


# --- the account slot ---------------------------------------------------------


def test_midstream_error_frame_then_done_and_the_slot_is_released(monkeypatch):
    """The highest-value assertion in this file: a mid-stream upstream failure
    must (a) reach the client as an error frame + [DONE] and (b) hand the
    account slot back.

    A leaked ``item.lock`` is unrecoverable by design -- the pool treats a
    permanently busy account as unusable, so ONE failed Qwen stream would drop
    that account out of rotation for the life of the process and every later
    request would fail over or 503. The producer runs on its own thread, so
    the release is observed through an explicit event rather than assumed (the
    same handshake idiom as `test_generator_close_releases_account_lock` in
    tests/test_reliability.py).
    """
    def _events(_h, _p, **_kw):
        # A live-emitted Qwen stream, so the two deltas reach the client and
        # THEN the upstream decides the account is muted.
        yield StreamEvent(kind="content", text="partial ")
        yield StreamEvent(kind="content", text="answer")
        raise HTTPException(status_code=403, detail="user is muted")

    _stub(monkeypatch, _events)
    app = _app(_settings(stream_mode="live"))
    item = app.state.pool.acquire(timeout=1)
    assert item.lock.locked(), "the slot must be held while the stream runs"

    async def scenario() -> tuple[str, bool]:
        body = sse_text(await collect_sse(QCS, app, app.state.pool, item, _prep()))
        for _ in range(300):
            if not item.lock.locked():
                return body, True
            await asyncio.sleep(0.01)
        return body, False

    body, released = asyncio.run(scenario())

    payloads = sse_payloads(body)
    assert payloads[-1]["error"]["message"] == "user is muted"
    assert payloads[-1]["error"]["code"] == 403
    assert body.count("data: [DONE]") == 1, body
    assert body.rstrip().endswith("data: [DONE]")
    deltas = [p["choices"][0]["delta"] for p in payloads if p.get("choices")]
    assert "".join(d.get("content", "") for d in deltas) == "partial answer", \
        "the deltas emitted before the 403 were lost"
    assert released, "account slot leaked: a locked item is unusable for the " \
                     "rest of the process"


def test_generator_close_releases_the_slot(monkeypatch):
    """Client disconnect must free the slot too (the DeepSeek twin is
    ``tests/test_reliability.test_generator_close_releases_account_lock``).

    ``consume_stream`` sets ``cancel_event`` in its finally; without that the
    producer would keep the account for the full reply after the harness hung
    up.
    """
    state = {"emitted": 0, "closed": False}

    def _events(_h, _p, **_kw):
        try:
            while True:
                state["emitted"] += 1
                yield StreamEvent(kind="content", text="x")
                time.sleep(0.005)
        finally:
            state["closed"] = True

    _stub(monkeypatch, _events)
    app = _app(_settings(stream_mode="live"))
    item = app.state.pool.acquire(timeout=1)

    async def scenario() -> bool:
        agen = QCS._stream_completion(app, app.state.pool, item, _prep())
        await agen.__anext__()   # role chunk
        await agen.__anext__()   # first live delta
        await agen.aclose()
        for _ in range(300):
            if not item.lock.locked():
                return True
            await asyncio.sleep(0.01)
        return False

    assert asyncio.run(scenario()), \
        "account slot not released after generator close"
    assert state["closed"], "stream_completion generator was not closed"
    assert state["emitted"] < 1000, "producer ran to completion after disconnect"


# --- Q2: no credential in a client-visible detail ----------------------------


def _terminal_401(monkeypatch, body: str) -> HTTPException:
    """Drive the real terminal 401 ladder with ``body`` as the upstream text."""
    _stub(monkeypatch, _fake_events([
        StreamEvent(kind="error", code="HTTP_401", message=body),
    ]))
    # A ban masquerading as a 401 is probed before the 401 is concluded; deny
    # it so the INVALID_TOKEN terminal handler is the one under test.
    monkeypatch.setattr(QC, "check_ban", lambda _h, timeout=12: (False, None))

    with pytest.raises(HTTPException) as exc_info:
        QCS._run_completion_core(_app(_settings(max_retries=0)),
                                 _item(email="a@x.com", token="STALE"),
                                 _prep())
    assert exc_info.value.status_code == 401
    return exc_info.value


def test_terminal_401_does_not_leak_an_echoed_jwt(monkeypatch):
    """Qwen's credential IS a JWT, and an upstream 401 can echo the header.

    The framed regex in ``core.engine`` exists for exactly this shape: each JWT
    segment is short, so a run-length redactor matches nothing and the whole
    token -- header and payload included -- travels verbatim.
    """
    exc = _terminal_401(
        monkeypatch,
        '{"code":"Unauthorized","details":"token rejected; echoed request '
        f'header: Authorization: Bearer {LIVE_JWT}"}}')
    detail = str(exc.detail)

    assert LIVE_JWT not in detail, f"leaked the JWT: {detail}"
    assert _JWT_SIG not in detail, f"leaked the JWT signature: {detail}"
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in detail
    assert "[redacted]" in detail
    # The diagnostic an operator actually needs is NOT collateral damage.
    assert "HTTP_401" in detail
    assert "Unauthorized" in detail
    assert "token invalid" in detail.lower()
    assert "chat.qwen.ai" in detail


def test_terminal_401_does_not_leak_a_token_shaped_run(monkeypatch):
    """The other credential shape: one long unbroken run of token characters."""
    exc = _terminal_401(
        monkeypatch,
        f'{{"code":"Unauthorized","details":"session for {LIVE_RUN} rejected"}}')

    assert LIVE_RUN not in str(exc.detail)
    assert "[redacted]" in str(exc.detail)


@pytest.mark.parametrize("mod, fn, raises", [
    # (module, function, raw-error variable name)
    ("deepseek", "_invalid_token_401_detail", "error"),
    ("qwen", "_terminal_invalid_token", "error"),
])
def test_401_detail_source_has_no_unscrubbed_variant(mod, fn, raises):
    """No client-visible path out of a 401 handler may quote the raw upstream
    body.

    The second, un-scrubbed formatting path is dead code today and a live leak
    the day someone reorders an ``if``: ``error[:200]`` is the account's own
    token. DeepSeek's helper and qwen's inline handler are separate bodies with
    separate seams, so both are scanned here. The qwen row is the STRONGER one
    -- its handler RAISES as well as returns, and both count as a leak.
    """
    import importlib

    source = textwrap.dedent(inspect.getsource(
        getattr(importlib.import_module(
            f"deepseaport.providers.{mod}.completion_service"), fn)))
    func = ast.parse(source).body[0]
    assert isinstance(func, ast.FunctionDef)

    offenders = [
        node for node in ast.walk(func)
        if isinstance(node, (ast.Return, ast.Raise))
        and any(isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == raises
                for inner in ast.walk(node))
    ]
    assert offenders == [], f"{mod}.{fn}: a path quotes the raw `{raises}` argument"
    assert f"{raises}[:200]" not in source, f"{mod}.{fn}: unscrubbed variant"


def test_qwen_scrub_sites_cover_every_upstream_body_sink():
    """Sweep: no upstream body may reach a detail= or a log unscrubbed.

    ``test_managed_serve_hardening`` scans DeepSeek's helper; qwen's
    ``_terminal_invalid_token`` had no equivalent. Rather than trusting a
    hand-run grep, assert the two properties that make the fix hold: the
    handler routes through ``_engine.scrub_upstream_text``, and no
    ``<something>[:N]`` slice of an upstream-text argument survives anywhere
    in the module -- the two log sinks being the re-signin failure detail
    (the signin body) and the overload retry note.
    """
    src = inspect.getsource(QCS)
    assert "_engine.scrub_upstream_text(error, 200)" in src
    assert '(detail or "")[:200]' not in src, \
        "the re-signin failure log still prints the raw signin body"
    assert '(error or "")[:80]' not in src, \
        "the overload retry log still slices the raw upstream error"
    # Every other terminal path delegates to core.engine, which scrubs inside
    # raise_rate_limit / raise_overload / raise_upstream* / ban_403_for_item.
    # Qwen's own two provider-specific handlers must reach engine directly;
    # the label-only ones are built by the shared factory, so check the factory
    # routes each kind to a scrubbing raise_* rather than re-pinning a list of
    # handler names that the table happens to contain today.
    for name in ("_terminal_invalid_token", "_terminal_upstream"):
        assert "_engine" in inspect.getsource(getattr(QCS, name)), name
    # _terminal_ban answers through the ban helper, which scrubs inside.
    assert "_ban_403_for_item" in inspect.getsource(QCS._terminal_ban)
    shared = inspect.getsource(_engine.generic_terminal_handlers)
    for raiser in ("raise_rate_limit", "raise_overload",
                   "raise_upstream_invalid", "raise_upstream"):
        assert raiser in shared, f"{raiser} is not reachable from the shared table"
    for raiser in ("raise_rate_limit", "raise_overload",
                   "raise_upstream_invalid", "raise_upstream"):
        assert "scrub_upstream_text" in inspect.getsource(getattr(_engine, raiser)), raiser
    assert "_engine.ban_403_for_item" in inspect.getsource(QCS._ban_403_for_item)
    assert "scrub_upstream_text" in inspect.getsource(_engine.ban_403_for_item)


# --- Q3: one chat per recovery pass, all of them deleted ----------------------


def test_waf_recovery_deletes_the_chat_it_replaced(monkeypatch):
    """Q3: a WAF pass must not strand the chat it abandoned upstream.

    ``_prepare_chat`` rebinds ``chat_id`` in place and the ``finally`` deletes
    only the LAST one, so before the fix one forced WAF error meant 2 chats
    created and 1 deleted. With ``max_retries`` up to 5 that is 5 stranded
    sessions per request, none of which ``auto_delete_session`` can reach
    again. The stubbed ``_delete_chat_bg`` records what was asked for.
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_refresh_waf_without_account_lock",
                        lambda *a, **k: None)
    deleted = _record_deletes(monkeypatch)
    ids = iter(["chat-1", "chat-2"])
    monkeypatch.setattr(QC, "create_chat", lambda _h, _m, **_kw: next(ids))
    _attempts(monkeypatch, [
        [StreamEvent(kind="error", code="WAF_CHALLENGE",
                     message="aliyun_waf_aa challenge")],
        [StreamEvent(kind="content", text="recovered"),
         StreamEvent(kind="finished", text="")],
    ])

    out = QCS._run_completion_core(
        _app(_settings(max_retries=1, auto_delete_session=True)),
        _item(email="a@x.com", token="tok"), _prep())

    assert out["content"] == "recovered"
    # chat-1 is the one the WAF pass abandoned; chat-2 is the finally. Both
    # under the headers they were CREATED with (a refreshed token would 401
    # the delete, which is exactly the 401 case below).
    assert [cid for _h, cid, _e in deleted] == ["chat-1", "chat-2"]
    assert all(enabled is True for _h, _cid, enabled in deleted)
    assert all(h.get("Cookie") == "token=tok" for h, _c, _e in deleted)


def test_invalid_token_recovery_deletes_the_chat_it_replaced(monkeypatch):
    """The 401 path leaked the same way, and is the worse of the two: the
    token was just re-signed, so the delete MUST use the pre-refresh headers
    or it 401s upstream and strands the chat anyway."""
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QC, "check_ban", lambda _h, timeout=12: (False, None))
    deleted = _record_deletes(monkeypatch)
    ids = iter(["chat-old", "chat-new"])
    monkeypatch.setattr(QC, "create_chat", lambda _h, _m, **_kw: next(ids))
    attempts = _attempts(monkeypatch, [
        [StreamEvent(kind="error", code="Unauthorized",
                     message="401 Unauthorized")],
        [StreamEvent(kind="content", text="ok"),
         StreamEvent(kind="finished", text="")],
    ])
    # A stored password makes the recover path do a direct re-signin, so the
    # token genuinely changes between the two attempts.
    monkeypatch.setattr(QA, "login_with_expiry",
                        lambda e, p, s=None: ("FRESHJWT", None, False, ""))

    out = QCS._run_completion_core(
        _app(_settings(max_retries=1, auto_delete_session=True)),
        _item(email="a@x.com", password="pw", token="STALE"), _prep())

    assert out["content"] == "ok"
    assert "FRESHJWT" in str(attempts[-1]), \
        "the refreshed token was never used for the retry"
    assert [cid for _h, cid, _e in deleted] == ["chat-old", "chat-new"]
    assert deleted[0][0].get("Cookie") == "token=STALE"
    assert deleted[1][0].get("Cookie") == "token=FRESHJWT"


def test_recreate_respects_auto_delete_off(monkeypatch):
    """``enabled=auto_delete`` is honoured on the recreate path.

    ``auto_delete_session: false`` means the operator is keeping the chats, so
    the recreate path must not delete behind their back -- the same gate the
    ``finally`` uses.
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_refresh_waf_without_account_lock",
                        lambda *a, **k: None)
    seen: list[bool] = []
    ids = iter(["chat-1", "chat-2"])
    monkeypatch.setattr(QC, "create_chat", lambda _h, _m, **_kw: next(ids))
    _attempts(monkeypatch, [
        [StreamEvent(kind="error", code="WAF_CHALLENGE",
                     message="aliyun_waf_aa challenge")],
        [StreamEvent(kind="content", text="ok"),
         StreamEvent(kind="finished", text="")],
    ])
    monkeypatch.setattr(
        QCS, "_delete_chat_bg",
        lambda h, cid, enabled=True: seen.append(enabled))

    out = QCS._run_completion_core(
        _app(_settings(max_retries=1, auto_delete_session=False)),
        _item(email="a@x.com", token="tok"), _prep())

    assert out["content"] == "ok"
    # The recreate pass asked with enabled=False -- ``_delete_chat_bg``'s own
    # ``if not enabled`` guard then no-ops -- and the ``finally`` never called
    # it at all (it gates on ``auto_delete`` first). So the operator's "keep my
    # chats" setting survives a recovery pass untouched.
    assert seen == [False], seen
    assert all(enabled is False for enabled in seen)


def test_failed_recreate_does_not_double_delete(monkeypatch):
    """A create that FAILS leaves the old id in place for the ``finally``.

    ``_prepare_chat`` assigns ``chat_id`` only on success, so the recreate
    helper must not also delete it -- that would be two DELETEs for one chat,
    and on the create-failure path the account would end up with a chat
    nobody tracks.
    """
    monkeypatch.setattr(QCS, "_headers", _fake_headers)
    monkeypatch.setattr(QCS, "_refresh_waf_without_account_lock",
                        lambda *a, **k: None)
    deleted = _record_deletes(monkeypatch)
    state = {"creates": 0}

    def _create(_h, _m, **_kw):
        state["creates"] += 1
        if state["creates"] == 1:
            return "chat-1"
        raise RuntimeError("WAF_CHALLENGE: chat create blocked again")

    monkeypatch.setattr(QC, "create_chat", _create)
    _attempts(monkeypatch, [
        [StreamEvent(kind="error", code="WAF_CHALLENGE",
                     message="aliyun_waf_aa challenge")],
    ])

    with pytest.raises(HTTPException) as exc_info:
        QCS._run_completion_core(
            _app(_settings(max_retries=1, auto_delete_session=True)),
            _item(email="a@x.com", token="tok"), _prep())

    # WAF exhausted its budget -> generic 502, the upstream reason preserved
    # (core.engine._run_recovery contains the failed recovery).
    assert exc_info.value.status_code == 502
    assert [cid for _h, cid, _e in deleted] == ["chat-1"], deleted


# --- monkeypatch seams other modules depend on --------------------------------


def test_delete_chat_bg_seam_keeps_its_name_and_signature():
    """``tests/test_qwen_provider_gaps.py`` stubs ``_delete_chat_bg`` away, and
    the recreate path routes through it, so its NAME and
    ``(headers, chat_id, enabled)`` signature are load-bearing.
    """
    params = list(inspect.signature(QCS._delete_chat_bg).parameters)
    assert params == ["headers", "chat_id", "enabled"]
    assert inspect.signature(QCS._delete_chat_bg).parameters["enabled"].default is True


def test_public_and_patchable_names_survive():
    """The names other modules and tests import must keep existing.

    ``_recreate_chat`` is deliberately a closure, not a module function: it
    needs ``chat_id``/``headers``/``auto_delete`` from the request scope, and
    DeepSeek's equivalent is a closure for the same reason. The guard is that
    the module-level seams around it did not move.
    """
    for name in ("_headers", "_delete_chat_bg", "_attempt", "_run_completion",
                 "_run_completion_core", "_complete_with_failover_sync",
                 "_stream_completion", "_raise_for_terminal_error",
                 "_TERMINAL_HANDLERS", "classify_error"):
        assert hasattr(QCS, name), name
