"""F3: a slot ``slot_releaser`` gave up must not be released again.

``threading.Lock.release()`` has no owner check: releasing a lock this thread
does not hold SUCCEEDS. So a request that released the per-account slot for a
global (browser login / WAF warmup) wait and then failed to get it back could
unlock the lock of the request that took the account meanwhile -- a third
request then acquires an account that is already streaming, which is exactly
the one-in-flight-per-account rule the lock exists for.

Hermetic: threads and a local ``AccountPool`` only. No network, no account, no
browser. One 0.05s reacquire deadline (monkeypatched); everything else is
event handshakes with bounded waits so a regression FAILS instead of hanging
the suite (same convention as test_reliability.py).
"""

from __future__ import annotations

import threading

import pytest

from zourcewell.accounts import AccountPool
from zourcewell.config import AccountConfig
from zourcewell.core import engine
from zourcewell.server_concurrency import _StreamCancelled

_HANDSHAKE = 10.0   # bounded wait, never an assertion bound
_DEADLINE = 0.05    # the reacquire deadline the give-up tests run under


def _pool() -> AccountPool:
    return AccountPool([AccountConfig(email="a@x.com", token="t")])


def _recording_release(released: list):
    """The real ``AccountPool.release``, recorded -- the defect IS its blind
    ``item.lock.release()``, so a plain list.append must not stand in for it."""
    def _release(item):
        released.append(item)
        AccountPool.release(item)
    return _release


def _give_the_slot_up(item, monkeypatch) -> None:
    """A's attempt: release the slot, let B take the account, exit without it.

    Runs on the attempting thread (the main thread in these tests), so the
    give-up it publishes is the one the calling thread's release site reads.
    B keeps ``item.lock``: the whole point is that A must never release it.
    """
    monkeypatch.setattr(engine, "SLOT_REACQUIRE_TIMEOUT", _DEADLINE)
    free, b_holds = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def _b():
        try:
            assert free.wait(_HANDSHAKE), "A never released the slot"
            assert item.lock.acquire(timeout=_HANDSHAKE), \
                "B could not take the freed slot"
            b_holds.set()
        except BaseException as exc:  # noqa: BLE001 - reported by the caller
            errors.append(exc)

    worker = threading.Thread(target=_b, daemon=True)
    worker.start()
    with engine.slot_releaser(item, None):
        free.set()
        assert b_holds.wait(_HANDSHAKE), errors or "B never took the slot"
    worker.join(_HANDSHAKE)
    assert errors == [], errors
    assert item.lock.locked(), "A released a lock it no longer held"


def test_give_up_publishes_the_abandoned_slot(monkeypatch):
    """The raw contract of the releaser: exiting without the slot back marks
    the give-up, and B's lock is untouched by that exit."""
    item = _pool().acquire(timeout=_HANDSHAKE)  # A holds the slot

    _give_the_slot_up(item, monkeypatch)

    assert engine.consume_slot_abandoned(item) is True, (
        "the give-up was not published, so the old comment's claim stands: "
        "run_with_failover would release a lock it no longer owns")
    assert item.lock.locked(), "A's exit released B's lock"
    item.lock.release()


def test_normal_attempt_releases_once_and_publishes_nothing():
    """The give-up must not become the normal path: an uncontended releaser
    reacquires, and the engine still releases exactly once."""
    pool = _pool()
    item = pool.acquire(timeout=_HANDSHAKE)
    released: list[object] = []
    released_for_wait: list[bool] = []

    def _run_once(_item):
        with engine.slot_releaser(item, None) as was_released:
            released_for_wait.append(was_released)
        assert item.lock.locked(), "the uncontended slot was not reacquired"
        return "ok"

    result = engine.run_with_failover(
        pool, item, _run_once, lambda _e: False,
        acquire=pool.acquire, allow_failover=False,
        release=_recording_release(released))

    assert result == "ok"
    assert released_for_wait == [True]
    assert released == [item], "the normal path must release exactly once"
    assert engine.consume_slot_abandoned(item) is False
    assert not item.lock.locked()


def test_cancelled_give_up_propagates_and_skips_the_release():
    """A disconnect during the global wait raises ``_StreamCancelled`` out of
    the releaser and leaves the slot released; the engine must not release
    that item a second time."""
    pool = _pool()
    item = pool.acquire(timeout=_HANDSHAKE)
    released: list[object] = []
    cancel = threading.Event()

    def _run_once(_item):
        with engine.slot_releaser(item, cancel):
            cancel.set()
        return "unreachable"

    with pytest.raises(_StreamCancelled):
        engine.run_with_failover(
            pool, item, _run_once, lambda _e: False,
            acquire=pool.acquire, allow_failover=False,
            release=_recording_release(released))

    assert released == [], "a cancelled give-up must not release the slot"
    assert not item.lock.locked()
    assert engine.consume_slot_abandoned(item) is False, \
        "the release site must consume the give-up, not leave it set"


def test_failover_does_not_release_a_slot_the_attempt_gave_up(monkeypatch):
    """Through the failover wrapper: A gave the slot up mid-attempt, B took
    the account, and the attempt still succeeded -- so B keeps its lock."""
    pool = _pool()
    item = pool.acquire(timeout=_HANDSHAKE)
    released: list[object] = []

    def _run_once(account):
        _give_the_slot_up(account, monkeypatch)
        return "ok"

    result = engine.run_with_failover(
        pool, item, _run_once, lambda _e: False,
        acquire=pool.acquire, allow_failover=False,
        release=_recording_release(released))

    assert result == "ok"
    assert released == [], \
        "the give-up's slot was released a second time (B's lock stolen)"
    assert item.lock.locked(), "B's held lock was unlocked by the release"
    assert engine.consume_slot_abandoned(item) is False
    item.lock.release()


def test_blind_release_would_steal_the_new_holders_lock(monkeypatch):
    """Falsifies the test above: with the give-up guard inert, the engine
    release unlocks the account B is streaming on and a third request can take
    it (``locked()`` False). This is the pre-fix behaviour."""
    pool = _pool()
    item = pool.acquire(timeout=_HANDSHAKE)
    released: list[object] = []
    monkeypatch.setattr(engine, "consume_slot_abandoned", lambda _item: False)

    def _run_once(account):
        _give_the_slot_up(account, monkeypatch)
        return "ok"

    engine.run_with_failover(
        pool, item, _run_once, lambda _e: False,
        acquire=pool.acquire, allow_failover=False,
        release=_recording_release(released))

    assert released == [item], released
    assert not item.lock.locked(), \
        "pre-fix behaviour expected: A's blind release freed B's held lock"


def test_nested_releaser_after_a_give_up_leaves_the_new_holder_alone(monkeypatch):
    """A give-up is not the end of the attempt: the token-refresh row opens a
    SECOND ``slot_releaser`` on the same item (DeepSeek does, after a WAF
    give-up), whose entry release would unlock B one frame later."""
    pool = _pool()
    item = pool.acquire(timeout=_HANDSHAKE)
    released: list[object] = []
    nested: list[bool] = []

    def _run_once(account):
        _give_the_slot_up(account, monkeypatch)  # B holds the account now
        with engine.slot_releaser(item, None) as was_released:
            nested.append(was_released)
        assert item.lock.locked(), "the nested releaser unlocked B's lock"
        return "ok"

    result = engine.run_with_failover(
        pool, item, _run_once, lambda _e: False,
        acquire=pool.acquire, allow_failover=False,
        release=_recording_release(released))

    assert result == "ok"
    assert nested == [False], \
        "a thread that gave the slot up must not release it again"
    assert released == [] and item.lock.locked()
    item.lock.release()
