"""Regression tests for the DeepSeek Obscura token-refresh gaps.

* #11: a completed leader login was discarded when the client disconnected
  during the browser run. ``slot_releaser``'s reacquire check raises
  ``_StreamCancelled`` on the way out of its ``with`` block, ahead of the
  persistence code, so config.json kept the stale token and the next request
  paid the whole browser login (with its captcha risk) again.
* #12: a follower returned whatever token it saw after the wait, so a failed
  or still-running leader still produced a value the caller treated as a
  refresh -- it retried once with the same invalid token and reported a
  misleading outcome.
* #2: the refresh persistence matched settings-wide, so the same email bound
  to another provider received this provider's token.

Offline: no network, no credentials, no browser.
"""

from __future__ import annotations

import threading

from support import app_with_bridge, on_disk, settings
from deepseaport.accounts import PooledAccount
from deepseaport.config import AccountConfig
from deepseaport.providers.deepseek import completion_service as CS
from deepseaport.server_concurrency import _StreamCancelled

# Bounded waits so a regression FAILS instead of hanging the suite. Not
# assertion bounds: on a loaded runner only an Event can tell "slow" apart
# from "broken" (same convention as test_reliability.py).
_HANDSHAKE = 10.0

_STALE = "stale-token-1234567890"
_FRESH = "fresh-token-1234567890"


def _leader_app(tmp_path, email: str):
    cfg = AccountConfig(email=email, password="pw", token=_STALE)
    item = PooledAccount(cfg=cfg)
    app = app_with_bridge(settings(accounts=[cfg],
                                   config_path=str(tmp_path / "config.json")))
    return app, item


def test_disconnected_leader_persists_token_and_wakes_follower(monkeypatch,
                                                               tmp_path):
    """#11. The client disconnect is noticed by slot_releaser's reacquire
    check as the browser run ends. The token that run produced must be
    persisted (and followers woken) before _StreamCancelled is raised."""
    app, item = _leader_app(tmp_path, "leader-cancel@x.com")
    cancel = threading.Event()
    login_entered = threading.Event()
    release_login = threading.Event()

    def fake_login(app_, item_, settings_, key_):
        login_entered.set()
        assert release_login.wait(_HANDSHAKE), "login was never released"
        cancel.set()  # the client disconnects as the browser run finishes
        return _FRESH, False, ""

    monkeypatch.setattr(CS, "_run_obscura_login", fake_login)
    item.lock.acquire()  # as the in-flight attempt holds the account slot

    leader_error: list[BaseException] = []

    def _leader():
        try:
            CS._refresh_token_via_obscura(app, item, cancel_event=cancel)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            leader_error.append(exc)

    leader_thread = threading.Thread(target=_leader, daemon=True)
    leader_thread.start()
    assert login_entered.wait(_HANDSHAKE), "leader never reached the login"

    # A second request joins the same browser run as a follower while the
    # leader is still inside it. Its first wait is the point where the
    # follower has already read the pre-refresh token.
    inflight = CS._TOKEN_REFRESH_INFLIGHT[item.cfg.identifier]
    waited = threading.Event()
    real_wait = inflight.wait

    def _tracked_wait(timeout=None):
        waited.set()
        return real_wait(timeout)

    inflight.wait = _tracked_wait
    follower_result: list[tuple] = []
    follower_thread = threading.Thread(
        target=lambda: follower_result.append(
            CS._refresh_token_via_obscura(app, item)), daemon=True)
    follower_thread.start()
    assert waited.wait(_HANDSHAKE), "follower never parked on the singleflight"

    release_login.set()
    leader_thread.join(_HANDSHAKE)
    follower_thread.join(_HANDSHAKE)
    assert not leader_thread.is_alive(), "leader thread stuck"
    assert not follower_thread.is_alive(), "follower thread stuck"

    assert leader_error and isinstance(leader_error[0], _StreamCancelled), \
        f"expected _StreamCancelled, got {leader_error!r}"
    # The browser run's result survived the abort: memory, Settings, disk.
    assert item.cfg.token == _FRESH
    assert app.state.settings.accounts[0].token == _FRESH
    assert on_disk(tmp_path / "config.json")["accounts"][0]["token"] == _FRESH
    # The follower waited on that run and got the new token, not the stale one.
    assert follower_result == [(_FRESH, False, "")]


def test_follower_returns_none_when_the_token_did_not_change(monkeypatch,
                                                             tmp_path):
    """#12. A failed leader leaves the token unchanged; the follower must
    return None (with the leader's detail), not the stale value as fresh."""
    app, item = _leader_app(tmp_path, "follower-unchanged@x.com")
    key = item.cfg.identifier
    inflight = threading.Event()
    inflight.set()  # the leader already finished with nothing to show
    monkeypatch.setitem(CS._TOKEN_REFRESH_INFLIGHT, key, inflight)
    monkeypatch.setitem(CS._TOKEN_REFRESH_BAN_HINT, key,
                        (False, "no-token: captcha challenge"))

    token, banned, detail = CS._refresh_token_via_obscura(app, item)

    assert token is None, "an unchanged token was reported as refreshed"
    assert banned is False
    assert detail == "no-token: captcha challenge"


def test_follower_returns_the_token_the_leader_persisted(monkeypatch, tmp_path):
    """#12 (control). When the token changed while the follower waited, the
    follower reports it as the refresh."""
    app, item = _leader_app(tmp_path, "follower-changed@x.com")
    key = item.cfg.identifier
    inflight = threading.Event()
    monkeypatch.setitem(CS._TOKEN_REFRESH_INFLIGHT, key, inflight)
    monkeypatch.setitem(CS._TOKEN_REFRESH_BAN_HINT, key, (False, ""))
    waited = threading.Event()
    real_wait = inflight.wait

    def _tracked_wait(timeout=None):
        waited.set()
        return real_wait(timeout)

    inflight.wait = _tracked_wait
    result: list[tuple] = []
    thread = threading.Thread(
        target=lambda: result.append(CS._refresh_token_via_obscura(app, item)),
        daemon=True)
    thread.start()
    assert waited.wait(_HANDSHAKE), "follower never parked"
    item.cfg.token = _FRESH  # the leader persisted its new token
    inflight.set()
    thread.join(_HANDSHAKE)
    assert not thread.is_alive(), "follower thread stuck"

    assert result == [(_FRESH, False, "")]


def test_refresh_writes_only_the_accounts_own_provider_record(monkeypatch,
                                                              tmp_path):
    """#2. Config may hold the same email under two providers -- a distinct
    account per provider. The refresh persistence matched settings-wide, so a
    DeepSeek login wrote its userToken onto the Qwen record: a credential the
    Qwen pool then serves and the DeepSeek pool never gets. Scoped to the
    item's own provider, the sibling record stays untouched."""
    sibling = AccountConfig(email="shared@x.com", password="pw",
                            token="qwen-token", provider="qwen")
    cfg = AccountConfig(email="shared@x.com", password="pw", token=_STALE,
                        provider="deepseek")
    item = PooledAccount(cfg=cfg)
    app = app_with_bridge(settings(accounts=[sibling, cfg],
                                   config_path=str(tmp_path / "config.json")))
    monkeypatch.setattr(CS, "_run_obscura_login",
                        lambda *a, **kw: (_FRESH, False, ""))

    token, banned, detail = CS._refresh_token_via_obscura(app, item)

    assert (token, banned, detail) == (_FRESH, False, "")
    assert item.cfg.token == _FRESH
    assert app.state.settings.accounts[0].token == "qwen-token"
    assert app.state.settings.accounts[1].token == _FRESH
    accounts = on_disk(tmp_path / "config.json")["accounts"]
    assert accounts[0]["token"] == "qwen-token"
    assert accounts[1]["token"] == _FRESH
