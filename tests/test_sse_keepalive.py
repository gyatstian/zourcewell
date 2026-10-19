"""F19: the consumer must write a byte during a silent producer stretch.

A silently-working producer is normal -- session create, PoW solve, one
recovery pass (a ~180s browser login, an Obscura warmup) emit nothing until
the reply lands -- but nginx drops an idle SSE body at 60s
(``proxy_read_timeout``) and Cloudflare at 100s, so a healthy buffered reply
died behind a proxy with no error frame and no ``[DONE]``.
``drain_stream_queue`` now waits in slices and sends one SSE comment frame per
elapsed slice, which every client ignores.

Hermetic: scripted queue stand-ins, no network, no producer thread. Every wait
is a millisecond wait or a 0.05s budget.
"""

from __future__ import annotations

import asyncio

from deepseaport.core import engine

PING = b": ping\n\n"
# The shipped interval, spelled out: it must stay well under nginx's 60s idle
# timeout, which is the shorter of the two proxies this project answers behind.
_NGINX_IDLE = 60.0


class _ScriptedQueue:
    """``asyncio.Queue`` stand-in: an ``(arrives_at, item)`` schedule.

    ``arrives_at`` is seconds from the first ``get``, i.e. the producer's own
    clock, not a per-call sleep: a slice timeout must not restart the wait.
    The last entry repeats, so a silent tail stays silent (a real
    ``asyncio.Queue`` loses nothing to a cancelled ``get`` either -- see
    ``test_a_real_queue_loses_nothing_to_a_slice_timeout``).
    """

    def __init__(self, schedule: list[tuple[float, tuple]]) -> None:
        self.calls = 0
        self._schedule = list(schedule)
        self._start: float | None = None
        self._at = 0

    async def get(self):
        loop = asyncio.get_running_loop()
        if self._start is None:
            self._start = loop.time()
        arrives_at, item = self._schedule[min(self._at, len(self._schedule) - 1)]
        self.calls += 1
        wait = arrives_at - (loop.time() - self._start)
        if wait > 0:
            await asyncio.sleep(wait)
        self._at = min(self._at + 1, len(self._schedule) - 1)
        return item


async def _drain(q, timeout: float, live: bool = True) -> list[bytes]:
    probe = {} if live else {"model": "m", "tools": None}
    return [f async for f in engine.drain_stream_queue(
        q, probe, live, "cid", 0, "m", timeout)]


def test_shipped_interval_is_under_a_proxy_idle_timeout():
    assert 0 < engine._KEEPALIVE_INTERVAL < _NGINX_IDLE


def test_pings_bridge_a_silent_stretch_before_the_first_chunk(monkeypatch):
    """Tiny interval: the frames between the role chunk and the producer's
    first event are pings, and the event still arrives (item not lost to the
    cancelled ``get``)."""
    monkeypatch.setattr(engine, "_KEEPALIVE_INTERVAL", 0.02)
    q = _ScriptedQueue([(0.12, ("content", "late")), (0.0, ("done", {}))])
    frames = asyncio.run(_drain(q, 5.0))

    role = engine.stream_chunk("cid", 0, "m", {"role": "assistant"})
    late = engine.stream_chunk("cid", 0, "m", {"content": "late"})
    assert frames[0] == role and frames.count(role) == 1
    assert frames.count(PING) >= 1, frames
    assert late in frames, frames
    assert frames.index(late) > frames.index(PING), \
        "a ping must not replace the payload it was covering for"
    assert frames[-2:] == [engine.stream_chunk("cid", 0, "m", {}, "stop"),
                           b"data: [DONE]\n\n"], frames
    assert all(PING == f for f in frames[1:frames.index(late)]), frames


def test_pings_keep_flowing_after_the_first_payload(monkeypatch):
    """The tail can be silent too (a long thinking pause, the close/tool
    parse), so the slices do not stop at ``got_first``."""
    monkeypatch.setattr(engine, "_KEEPALIVE_INTERVAL", 0.02)
    q = _ScriptedQueue([(0.0, ("content", "x")), (0.1, ("done", {}))])

    frames = asyncio.run(_drain(q, 5.0))

    assert frames.count(PING) >= 1, frames
    assert frames.index(PING) > frames.index(
        engine.stream_chunk("cid", 0, "m", {"content": "x"}))
    assert frames[-1] == b"data: [DONE]\n\n"


def test_a_real_queue_loses_nothing_to_a_slice_timeout(monkeypatch):
    """The slices cancel a pending ``asyncio.Queue.get``; an item that lands
    right after that cancel must still be delivered, not dropped with the
    cancelled waiter."""
    monkeypatch.setattr(engine, "_KEEPALIVE_INTERVAL", 0.02)

    async def _run():
        out: asyncio.Queue = asyncio.Queue()

        async def _producer():
            await asyncio.sleep(0.05)          # silence across two slices
            out.put_nowait(("content", "late"))
            out.put_nowait(("done", {}))

        task = asyncio.ensure_future(_producer())
        try:
            return [f async for f in engine.drain_stream_queue(
                out, {}, True, "cid", 0, "m", 5.0)]
        finally:
            await task

    frames = asyncio.run(_run())

    assert frames.count(PING) >= 1, frames
    assert engine.stream_chunk("cid", 0, "m", {"content": "late"}) in frames
    assert frames[-1] == b"data: [DONE]\n\n"


def test_sub_interval_timeout_keeps_todays_frames_byte_for_byte(monkeypatch):
    """A budget shorter than the interval must not gain a ping: the watchdog
    path stays exactly the role chunk + error frame + ``[DONE]``, which the
    existing hermetic watchdog tests assert on. Runs on the SHIPPED interval.
    """
    q = _ScriptedQueue([(30.0, ("done", {}))])

    frames = asyncio.run(_drain(q, 0.05, live=False))

    assert frames == [
        engine.stream_chunk("cid", 0, "m", {"role": "assistant"}),
        engine.sse_error_frame("upstream timed out"),
        b"data: [DONE]\n\n",
    ], frames
