"""F9: a stream must notice a disconnect while the socket is SILENT.

``providers/qwen/client.py`` pumped SSE without the bounded read DeepSeek wraps
around the identical ``transport.pump_sse`` call, so a disconnect during a
pre-first-token stretch (or any idle upstream hop) went unnoticed until the
socket spoke: ``drain_attempt``'s per-event cancel check cannot run while no
event is yielded. The account lock and its worker stayed held for the whole
``DEFAULT_TIMEOUT`` (~300s), queueing every other request for that account and
then failing them.

Both providers are exercised because the machinery is shared now: the DeepSeek
row pins the seam that moved, the Qwen row is the fix. The non-SSE row covers
the same swap on the 200+JSON body drain, which reads the same queue.

Hermetic -- no curl, no network. ``_SilentStream``'s ``queue`` is the read the
pump actually blocks on, and the slice constant is patched down so a full test
costs ~0.1s.
"""

from __future__ import annotations

import importlib
import queue
import threading
import time

import pytest


class _SilentStream:
    """curl_cffi-shaped response whose queue blocks until something is pushed.

    ``iter_lines`` reads through ``self.queue`` so the pump's ``queue`` swap is
    the mechanism under test; ``quit_now`` and ``close`` mimic curl's flag and
    its thread-joining close.
    """

    def __init__(self):
        self.queue = queue.Queue()
        self.quit_now = threading.Event()
        self.close_calls = 0
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    def push(self, chunk) -> None:
        self.queue.put_nowait(chunk)

    def iter_lines(self):
        while True:
            chunk = self.queue.get()  # blocks: the whole defect
            if not isinstance(chunk, (bytes, bytearray)):
                return  # curl_cffi's end-of-stream sentinel
            yield chunk

    def close(self) -> None:
        # The real one joins curl's worker thread (see transport._abandon).
        self.close_calls += 1


class _SilentJsonStream(_SilentStream):
    """200 + application/json variant: the non-SSE body drain's read.

    ``text``/``content`` stay empty like a curl_cffi ``stream=True`` body, so
    both providers take their JSON-drain branch and park on the same queue.
    """

    def __init__(self):
        super().__init__()
        self.headers = {"content-type": "application/json"}
        self.text = ""
        self.content = b""


_CLIENTS = [
    pytest.param("zourcewell.providers.qwen.client", id="qwen"),
    pytest.param("zourcewell.providers.deepseek.client", id="deepseek"),
]


def _sess(monkeypatch, client, resp) -> None:
    class _S:
        def post(self, *a, **k):
            return resp

    monkeypatch.setattr(client, "_SESSION", _S())


@pytest.mark.parametrize("mod", _CLIENTS)
def test_a_disconnect_ends_the_stream_during_a_silent_read(monkeypatch, mod):
    """The cancel token must reach a stream parked in a blocking read.

    Pre-fix (Qwen) this thread never returns: no ``_cancellable`` swap means
    ``queue.get`` stays untimed, so the per-event cancel check in
    ``drain_attempt`` is never reached and the read parks for DEFAULT_TIMEOUT.
    """
    from zourcewell.core import engine as _engine
    from zourcewell.core import transport as T

    client = importlib.import_module(mod)
    monkeypatch.setattr(T, "_READ_SLICE_SECONDS", 0.1)
    resp = _SilentStream()
    _sess(monkeypatch, client, resp)

    cancel = threading.Event()
    events: list = []
    failure: list = []

    def _consume():
        # Same order as drain_attempt: publish the token, THEN open the stream.
        _engine._CURRENT_CANCEL.event = cancel
        try:
            events.extend(client.stream_completion({"h": "1"}, {"prompt": "hi"}))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the asserts
            failure.append(exc)
        finally:
            _engine._CURRENT_CANCEL.event = None

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    time.sleep(0.1)  # let the read park with the upstream silent
    cancel.set()
    thread.join(3.0)

    assert not thread.is_alive(), (
        "the read did not return after cancel; the account lock and its worker "
        "are still held")
    assert failure == [], failure
    assert events == [], "a cancelled stream must not emit anything"
    assert isinstance(resp.queue, T._BoundedQueue), "the read was never bounded"
    assert resp.quit_now.is_set(), "the transfer was not asked to stop"
    assert resp.close_calls == 0, (
        "Response.close() joins curl's worker thread and would block here")
    assert "close" in vars(resp), "close() was never neutralized"


@pytest.mark.parametrize("mod", _CLIENTS)
def test_a_disconnect_ends_a_silent_non_sse_drain(monkeypatch, mod):
    """The 200 non-SSE body drain reads the same blocking queue as the pump.

    Pre-fix the drain parked on the raw ``queue.get``: a disconnect during a
    silent body was invisible until the socket spoke or curl's whole budget
    expired, so the account lock and its worker stayed held. The cancel must
    end the drain within a read slice.
    """
    from zourcewell.core import engine as _engine
    from zourcewell.core import transport as T

    client = importlib.import_module(mod)
    monkeypatch.setattr(T, "_READ_SLICE_SECONDS", 0.1)
    resp = _SilentJsonStream()
    _sess(monkeypatch, client, resp)

    cancel = threading.Event()
    events: list = []
    failure: list = []

    def _consume():
        _engine._CURRENT_CANCEL.event = cancel
        try:
            events.extend(client.stream_completion({"h": "1"}, {"prompt": "hi"}))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the asserts
            failure.append(exc)
        finally:
            _engine._CURRENT_CANCEL.event = None

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    time.sleep(0.1)  # parked in the body drain
    cancel.set()
    thread.join(3.0)

    assert not thread.is_alive(), (
        "the drain did not return after cancel; the account lock and its "
        "worker are still held")
    assert failure == [], failure
    assert [e.kind for e in events] == ["error"], events
    assert isinstance(resp.queue, T._BoundedQueue), "the read was never bounded"
    assert resp.quit_now.is_set(), "the transfer was not asked to stop"
    assert resp.close_calls == 0, (
        "Response.close() joins curl's worker thread and would block here")
    assert "close" in vars(resp), "close() was never neutralized"


@pytest.mark.parametrize("mod", _CLIENTS)
def test_without_a_token_the_read_is_left_exactly_as_it_was(monkeypatch, mod):
    """No token (a direct/unit-test call) means no-op, not a different read.

    ``data: [DONE]`` is the one line both wire parsers finish on, so the same
    control works for each provider.
    """
    from zourcewell.core import transport as T

    client = importlib.import_module(mod)
    monkeypatch.setattr(T, "_READ_SLICE_SECONDS", 0.1)
    resp = _SilentStream()
    raw_queue = resp.queue
    _sess(monkeypatch, client, resp)
    resp.push(b"data: [DONE]\n")

    events = list(client.stream_completion({"h": "1"}, {"prompt": "hi"}))

    assert [e.kind for e in events] == ["finished"]
    assert resp.queue is raw_queue, "a token-free call must not touch the read"
    assert not resp.quit_now.is_set(), "the transfer was stopped for no reason"
    assert resp.close_calls == 1, "the pump's own finally closes exactly once"
    assert "close" not in vars(resp), "close() was replaced without a cancel"


def test_deepseek_reexports_the_shared_bounded_read():
    """The historical ``client.*`` names are the shared objects, not copies.

    ``tests/test_stream_protocol_index.py`` reads ``_cancellable`` /
    ``_READ_SLICE_SECONDS`` off the DeepSeek client, and its stream path must
    keep calling the module-global name so a patch there still wins.
    """
    from zourcewell.core import transport as T
    from zourcewell.providers.deepseek import client as DS

    assert DS._cancellable is T._cancellable
    assert DS._ClientGone is T._ClientGone
    assert DS._abandon is T._abandon
    assert DS._BoundedQueue is T._BoundedQueue
    assert DS._READ_SLICE_SECONDS == T._READ_SLICE_SECONDS
