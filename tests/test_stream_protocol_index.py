"""Stream protocol conformance: tool_calls ``index`` + OpenAI-shaped errors.

Two verified defects, both invisible to a single-call happy path:

* ``core.engine.tool_finish_chunks`` emitted each parallel tool call as a
  complete, standalone delta with NO ``index``. OpenAI's streaming contract
  makes ``index`` mandatory on ``tool_calls`` deltas -- it is the only field
  that says which of N calls a delta belongs to -- so ``openai-python``
  raised inside the stream (mid agent-loop turn) and index-keyed
  accumulators could not merge parallel calls.
* ``core.engine.drain_stream_queue`` hand-rolled three different in-band SSE
  error frames: none carried ``error.type``, and ``err.detail`` (untyped in
  FastAPI) could serialize as a JSON object where the OpenAI schema requires
  a string.

All offline: no network, no credentials, no browser.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

from fastapi import HTTPException

from zourcewell.core import engine as _engine
from zourcewell.core import openai as _openai

# Parser-shaped calls (what ``parse_tool_calls`` returns and what
# ``_openai_response`` puts in ``message.tool_calls``).
CALLS = [
    {"id": "call_001", "type": "function",
     "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}},
    {"id": "call_002", "type": "function",
     "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'}},
    {"id": "call_003", "type": "function",
     "function": {"name": "search", "arguments": '{"q":"deep sea port"}'}},
]

TOOLS = [{"type": "function", "function": {"name": t["function"]["name"]}}
         for t in CALLS]


def _prep(**kw) -> dict:
    base = dict(model="test-model", tools=list(TOOLS), prompt="hi",
                stream_options={"include_usage": True})
    base.update(kw)
    return base


def _stub_parse(content, tools, max_args_chars=None, forgiving=False):
    """Parser seam returning fixed calls (both paths must see these)."""
    return [dict(c) for c in CALLS], ""


def _decode(frames) -> list:
    """SSE bytes -> decoded JSON; the ``[DONE]`` sentinel as a string."""
    out = []
    for raw in frames:
        text = raw.decode("utf-8")
        assert text.startswith("data: "), text
        body = text[len("data: "):].strip()
        out.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return out


async def _drain(prep: dict, items: list, consumer_timeout: float = 5.0,
                 live: bool = False, parse_fn=None) -> list:
    """Run the real shared consumer over a pre-filled queue."""
    out: asyncio.Queue = asyncio.Queue()
    for item in items:
        out.put_nowait(item)
    cid, created = "chatcmpl-testtesttesttesttest12", 1_700_000_000
    return _decode([
        frame async for frame in _engine.drain_stream_queue(
            out, prep, live, cid, created, prep["model"],
            consumer_timeout, parse_fn=parse_fn)])


def _tool_deltas(chunks: list) -> list:
    """tool_calls delta elements, in emission order."""
    deltas = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for choice in chunk.get("choices") or []:
            for tc in (choice.get("delta") or {}).get("tool_calls") or []:
                deltas.append(tc)
    return deltas


# --- tool_calls index -------------------------------------------------------


def test_parallel_tool_deltas_carry_index():
    """3 parallel calls -> index 0,1,2 in emission order, ids/names/args
    untouched, then a bare finish chunk. Pre-fix every delta had no index."""
    chunks = _decode(_engine.tool_finish_chunks(
        lambda delta, finish=None: _engine.stream_chunk(
            "cid", 1, "m", delta, finish),
        CALLS))

    deltas = _tool_deltas(chunks)
    assert [d["index"] for d in deltas] == [0, 1, 2], deltas
    for got, want in zip(deltas, CALLS):
        assert got["id"] == want["id"]
        assert got["type"] == "function"
        assert got["function"] == want["function"]

    last = chunks[-1]["choices"][0]
    assert last["delta"] == {}, last
    assert last["finish_reason"] == "tool_calls", last


def test_single_tool_call_index_is_zero():
    """One call is still indexed 0 -- not omitted, not 1-based."""
    chunks = _decode(_engine.tool_finish_chunks(
        lambda delta, finish=None: _engine.stream_chunk(
            "cid", 1, "m", delta, finish),
        CALLS[:1]))

    deltas = _tool_deltas(chunks)
    assert [d["index"] for d in deltas] == [0], deltas
    assert deltas[0]["id"] == "call_001"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_and_non_stream_agree_on_calls():
    """No stream/non-stream divergence: same ids, names and arguments from
    the same parser result (only the streamed form adds ``index``)."""
    prep = _prep()
    done = {"thinking": "", "content": "tool markup", "usage_total": None}

    chunks = asyncio.run(_drain(prep, [("done", done)], parse_fn=_stub_parse))
    streamed = [{"id": d["id"], "function": d["function"]}
                for d in _tool_deltas(chunks)]
    assert [d["index"] for d in _tool_deltas(chunks)] == [0, 1, 2]

    response = _openai._openai_response(
        prep, {"content": "tool markup", "thinking": ""},
        parse_fn=_stub_parse)
    message = response["choices"][0]["message"]
    non_streamed = [{"id": c["id"], "function": c["function"]}
                    for c in message["tool_calls"]]

    assert streamed == non_streamed, (streamed, non_streamed)
    assert streamed == [{"id": c["id"], "function": c["function"]}
                        for c in CALLS]
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if isinstance(c, dict) and c.get("choices")]
    assert finishes[-1] == "tool_calls", chunks
    assert finishes.count("tool_calls") == 1, chunks
    assert chunks.count("[DONE]") == 1, chunks


def test_no_calls_finish_chunk_unchanged():
    """Zero calls keeps the pre-existing length/stop tail (no index, no
    tool_calls frame) -- the fix must not touch the empty path."""
    chunks = _decode(_engine.tool_finish_chunks(
        lambda delta, finish=None: _engine.stream_chunk(
            "cid", 1, "m", delta, finish),
        None, truncated=True))
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"] == {}
    assert chunks[0]["choices"][0]["finish_reason"] == "length"


# --- SSE error frames -------------------------------------------------------


def _error_frames(chunks: list) -> list:
    return [c["error"] for c in chunks
            if isinstance(c, dict) and "error" in c]


def _assert_openai_error(err: dict, message: str, kind: str,
                         code) -> None:
    assert isinstance(err["message"], str), err
    assert err["message"] == message, err
    assert err["type"] == kind, err
    assert err["code"] == code, err


def test_http_error_frame_is_openai_shaped():
    """4xx detail -> invalid_request_error, 5xx -> upstream_error, both with
    the HTTP status as ``code``. Pre-fix neither had ``type``."""
    prep = _prep()

    chunks = asyncio.run(_drain(
        prep, [("http_error", HTTPException(
            status_code=400, detail="unknown model"))]))
    _assert_openai_error(_error_frames(chunks)[0], "unknown model",
                         "invalid_request_error", 400)
    assert chunks[-1] == "[DONE]" and chunks.count("[DONE]") == 1, chunks

    chunks = asyncio.run(_drain(
        prep, [("http_error", HTTPException(
            status_code=502, detail="upstream error: boom"))]))
    _assert_openai_error(_error_frames(chunks)[0], "upstream error: boom",
                         "upstream_error", 502)
    assert chunks.count("[DONE]") == 1, chunks


def test_dict_detail_message_is_coerced_to_string():
    """A dict ``HTTPException.detail`` must not land as a nested JSON
    object in ``error.message`` (strict SDKs reject it). A list detail --
    what FastAPI 422 validation raises -- likewise."""
    prep = _prep()

    chunks = asyncio.run(_drain(prep, [("http_error", HTTPException(
        status_code=502, detail={"m": 1}))]))
    err = _error_frames(chunks)[0]
    _assert_openai_error(err, str({"m": 1}), "upstream_error", 502)
    assert chunks.count("[DONE]") == 1, chunks

    detail = [{"loc": ["body", "messages"], "msg": "field required"}]
    chunks = asyncio.run(_drain(prep, [("http_error", HTTPException(
        status_code=422, detail=detail))]))
    _assert_openai_error(_error_frames(chunks)[0], str(detail),
                         "invalid_request_error", 422)
    assert chunks.count("[DONE]") == 1, chunks


class _NoDetailError:
    """HTTPException-shaped error whose ``detail`` is None.

    Starlette backfills a missing detail with the status phrase
    (``HTTP_Exception(401)`` reports "Unauthorized"), so the truly-empty
    case is only reachable with a detail-less error object -- which is
    exactly what the ``str()`` coercion has to survive.
    """

    status_code = 401
    detail = None


def test_missing_detail_message_is_still_a_string():
    """No detail at all: ``message`` stays a JSON string (never None/null)
    so the frame deserializes."""
    chunks = asyncio.run(_drain(
        _prep(), [("http_error", _NoDetailError())]))

    err = _error_frames(chunks)[0]
    _assert_openai_error(err, "", "invalid_request_error", 401)
    assert isinstance(err["message"], str)
    assert chunks.count("[DONE]") == 1, chunks


def test_timeout_and_generic_frames_are_openai_shaped():
    """The two status-less frames (watchdog timeout, unclassified queue
    item) carry upstream_error with code=null, and one [DONE] each."""
    chunks = asyncio.run(_drain(_prep(), [], consumer_timeout=0.05))
    _assert_openai_error(_error_frames(chunks)[0], "upstream timed out",
                         "upstream_error", None)
    assert chunks.count("[DONE]") == 1, chunks
    assert chunks[-1] == "[DONE]", chunks

    chunks = asyncio.run(_drain(
        _prep(), [("mystery", RuntimeError("upstream exploded"))]))
    _assert_openai_error(_error_frames(chunks)[0], "upstream exploded",
                         "upstream_error", None)
    assert chunks.count("[DONE]") == 1, chunks
    assert chunks[-1] == "[DONE]", chunks

# --- Finding 14: a disconnecting stream held its account slot ---------------

class _SilentStream:
    """A curl_cffi-shaped streaming response whose queue never produces again.

    This is the shape that held the slot: ``iter_lines`` blocks on an untimed
    ``queue.get``, so with the upstream silent nothing was ever yielded out of
    the generator and ``drain_attempt``'s per-event cancel check could not run.
    ``quit_now`` mimics curl's flag; line splitting mirrors curl_cffi's so the
    parser sees exactly what it sees in production.
    """

    def __init__(self):
        self.queue = queue.Queue()
        self.quit_now = threading.Event()
        self.closed = False
        self.close_calls = 0
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    def _push(self, line: bytes) -> None:
        self.queue.put_nowait(line)

    def iter_lines(self):
        pending = b""
        while True:
            chunk = self.queue.get()  # blocks until pushed: the whole defect
            if isinstance(chunk, BaseException):
                raise chunk
            if not isinstance(chunk, (bytes, bytearray)):
                return  # curl_cffi's end-of-stream sentinel
            if pending:
                chunk = pending + chunk
            lines = chunk.splitlines()
            pending = (lines.pop()
                       if lines and lines[-1] and lines[-1][-1:] == chunk[-1:]
                       else None)
            yield from lines
        if pending is not None:
            yield pending

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_a_disconnect_is_noticed_inside_the_socket_read():
    """The bounded read: cancellation is observed WITHOUT an SSE event.

    ``_cancellable`` swaps the response's queue for one that returns control
    every ``_READ_SLICE_SECONDS`` and raises once the cancel token is set. That
    is the only place it CAN be observed -- curl_cffi 0.16.3's
    ``iter_lines(chunk_size, decode_unicode, delimiter)`` has no ``timeout``
    parameter and ``iter_content`` blocks on an untimed ``queue.get``, verified
    against the installed package.

    Before the swap this cannot finish at all: with the upstream silent no event
    is ever yielded, so the per-event cancel check in ``drain_attempt`` never
    runs and the read parks for the whole DEFAULT_TIMEOUT.
    """
    from zourcewell.core import engine as _engine
    from zourcewell.providers.deepseek import client as DS

    resp = _SilentStream()
    resp._push(b'data: {"p":"response/content","o":"APPEND","v":"hi"}\n')
    cancel = threading.Event()
    seen: list = []
    failure: list = []

    def _stream():
        # Same order as drain_attempt: publish the token, THEN open the stream.
        _engine._CURRENT_CANCEL.event = cancel
        try:
            DS._cancellable(resp)
            seen.extend(DS.T.pump_sse(resp, DS.StreamParser(),
                                      precheck=DS._raw_json_ban_precheck))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the asserts
            failure.append(exc)
        finally:
            _engine._CURRENT_CANCEL.event = None

    thread = threading.Thread(target=_stream, daemon=True)
    thread.start()

    # Let the first event through, then disconnect with the upstream silent.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not seen and not failure:
        time.sleep(0.01)
    assert [e.kind for e in seen] == ["content"], (
        f"the first delta never arrived, so the test proves nothing: {failure}")
    cancel.set()

    thread.join(DS._READ_SLICE_SECONDS + 5.0)
    assert not thread.is_alive(), (
        "the read did not return after cancel; the account slot is still held")
    assert resp.quit_now.is_set(), "the transfer was not asked to stop"


def test_the_300s_budget_is_not_shortened_by_the_bounded_read():
    """Only the GRANULARITY changes, never the total allowance.

    curl still owns the stall budget (``DEFAULT_TIMEOUT`` -> LOW_SPEED_TIME);
    the proxy only re-enters ``get``. Asserted on the constant that the proxy
    is built from plus the one that would move if someone "fixed" this by
    lowering the timeout instead.
    """
    from zourcewell.providers.deepseek import client as DS

    assert DS.DEFAULT_TIMEOUT == 300, "the overall socket budget changed"
    assert 0 < DS._READ_SLICE_SECONDS < DS.DEFAULT_TIMEOUT, (
        "a slice must be a granularity, not a shortened budget")


def test_drain_attempt_publishes_and_restores_the_cancel_token():
    """The token is thread-local and restored, not cleared.

    ``drain_attempt`` runs on a shared producer thread pool, so a leaked token
    would cancel the NEXT request's read; a cleared one would break nesting.
    """
    from zourcewell.core import engine as _engine

    assert _engine.current_cancel_event() is None

    class _Ev:
        def __init__(self, kind):
            self.kind = kind

    seen: list = []

    def _stream(_headers, _payload):
        seen.append(_engine.current_cancel_event())
        yield _Ev("finished")

    outer = threading.Event()
    _engine._CURRENT_CANCEL.event = outer
    try:
        _engine.drain_attempt(_stream, {}, {}, cancel_event=None)
        assert seen == [None], "a stale token leaked into a stream function"

        inner = threading.Event()
        _engine.drain_attempt(_stream, {}, {}, cancel_event=inner)
        assert seen[1] is inner, "the token was not published to the stream fn"
    finally:
        _engine._CURRENT_CANCEL.event = None

    assert _engine.current_cancel_event() is None


def test_cancellable_leaves_a_response_without_a_live_queue_alone():
    """A unit-test double (or a non-stream response) must be untouched.

    Every existing test double in the suite exposes ``iter_lines`` and nothing
    else; swapping in a queue proxy there would break them all, and it is also
    pointless -- with no live queue there is no blocked read to bound.
    """
    from zourcewell.providers.deepseek import client as DS

    class _NoQueue:
        def iter_lines(self):
            return iter([b"data: x\n"])

        def close(self):
            pass

    resp = _NoQueue()
    DS._cancellable(resp)
    assert not hasattr(resp, "queue")

    # Same with a live token in force -- still nothing to bound, because there
    # is no queue to bound.
    from zourcewell.core import engine as _engine

    cancel = threading.Event()
    _engine._CURRENT_CANCEL.event = cancel
    try:
        DS._cancellable(resp)
    finally:
        _engine._CURRENT_CANCEL.event = None
    assert not hasattr(resp, "queue")
