"""Every error response is JSON: ``{"detail": ..., "error": {...}}`` (F18).

Non-stream failures returned a bare ``{"detail": ...}`` -- plain "Internal
Server Error" text for an unhandled 500 -- while the SSE path already spoke the
OpenAI error shape. ``install_error_envelope`` adds the OpenAI-shaped ``error``
object to every error response on BOTH apps (live and dry run) without touching
``detail``, which scripts, SDKs and the whole test suite read.

Also here: the Retry-After rule for our own raisers. A 429 raised after the
account-slot queue timed out carries the header ONLY when the pool knows a real
horizon (a cooling account's remaining cooldown); the empty-pool 503 carries
none, because only a config change makes that retryable and a fabricated value
is worse than no value.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from zourcewell.accounts import AccountPool
from zourcewell.config import AccountConfig
from zourcewell.dry_run import create_dry_app
from zourcewell.server import create_app
from zourcewell.server_concurrency import _acquire_slot_or_499
from zourcewell.server_routes import MAX_REQUEST_BYTES, error_envelope
from support import settings as _settings

AUTH = {"Authorization": "Bearer sk-test"}
# Token-shaped: the 500 body must never carry it back (core scrub contract).
SECRET = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"


def _s():
    return _settings(keys=["sk-test"],
                     accounts=[AccountConfig(email="a@x.com", token="tok")])


# --- the envelope itself ------------------------------------------------------


@pytest.mark.parametrize("status,expected", [
    (400, "invalid_request_error"), (409, "invalid_request_error"),
    (413, "invalid_request_error"), (422, "invalid_request_error"),
    (401, "authentication_error"), (403, "permission_error"),
    (404, "not_found_error"), (429, "rate_limit_error"),
    (499, "api_error"), (500, "api_error"), (502, "api_error"),
    (503, "api_error"),
])
def test_status_to_openai_type_map(status, expected):
    assert error_envelope(status, "x")["error"]["type"] == expected


def test_non_string_detail_stays_a_string_in_error_message():
    """A dict detail must not land as a nested object: strict SDKs type
    ``error.message`` as a string (the SSE frame already coerces)."""
    assert error_envelope(400, {"field": "messages"})["error"]["message"] == \
        '{"field": "messages"}'
    assert error_envelope(400, None)["error"]["message"] == ""


# --- HTTPException paths (live app) -------------------------------------------


def test_400_keeps_its_detail_and_gains_the_error_object():
    app = create_app(_s())

    @app.get("/boom-400")
    async def boom_400():
        raise HTTPException(status_code=400, detail="messages[1] is not an object")

    resp = TestClient(app).get("/boom-400")

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "messages[1] is not an object"
    assert body["error"] == {"message": "messages[1] is not an object",
                             "type": "invalid_request_error",
                             "param": None, "code": None}


def test_404_maps_the_router_and_the_route_alike():
    resp = TestClient(create_app(_s())).get("/v1/nope")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Not Found"
    assert resp.json()["error"]["type"] == "not_found_error"


def test_429_keeps_retry_after_and_the_detail():
    app = create_app(_s())

    @app.get("/boom-429")
    async def boom_429():
        raise HTTPException(status_code=429,
                            detail="all DeepSeek accounts busy, retry later",
                            headers={"Retry-After": "7"})

    resp = TestClient(app).get("/boom-429")

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7", "the raiser's header was dropped"
    assert resp.json()["detail"] == "all DeepSeek accounts busy, retry later"
    assert resp.json()["error"]["type"] == "rate_limit_error"


def test_401_from_the_real_route_maps_to_authentication_error():
    resp = TestClient(create_app(_s())).get("/v1/accounts")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid API key"
    assert resp.json()["error"]["type"] == "authentication_error"


def test_body_limit_413_carries_the_envelope():
    """The middleware builds its 413 OUTSIDE the exception handlers, so it goes
    through error_envelope by hand -- it must not drift from the handler shape."""
    resp = TestClient(create_app(_s())).post(
        "/v1/accounts", content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"content-type": "application/json", **AUTH})

    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert "exceeds" in body["detail"]
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == body["detail"]


def test_422_keeps_the_fastapi_detail_list_and_gains_the_envelope():
    resp = TestClient(create_app(_s())).post(
        "/v1/chat/completions", json=[1, 2], headers=AUTH)

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert [e["loc"] for e in body["detail"]] == [["body"]], body["detail"]
    assert body["error"]["type"] == "invalid_request_error"
    assert isinstance(body["error"]["message"], str)


# --- unhandled 500 ------------------------------------------------------------


def test_unhandled_500_returns_the_envelope_and_still_reaches_the_server_log():
    app = create_app(_s())

    @app.get("/boom-500")
    async def boom_500():
        raise RuntimeError(f"kaboom {SECRET}")

    resp = TestClient(app, raise_server_exceptions=False).get("/boom-500")

    assert resp.status_code == 500
    assert resp.json() == {"detail": "internal server error",
                           "error": {"message": "internal server error",
                                     "type": "api_error",
                                     "param": None, "code": None}}
    assert "kaboom" not in resp.text and SECRET not in resp.text

    # The handler must not swallow the exception: ServerErrorMiddleware sends
    # the envelope and then RE-RAISES, which is what keeps the server's
    # traceback log (uvicorn prints it; starlette itself never logs).
    with pytest.raises(RuntimeError):
        TestClient(app).get("/boom-500")


# --- dry-run parity -----------------------------------------------------------


def test_dry_app_gets_the_same_envelope():
    c = TestClient(create_dry_app(_settings(keys=["sk-test"])),
                   raise_server_exceptions=False)

    unauth = c.get("/v1/models")
    assert unauth.status_code == 401
    assert unauth.json()["detail"] == "invalid API key"
    assert unauth.json()["error"]["type"] == "authentication_error"

    missing = c.post("/v1/chat/completions",
                     json={"model": "gpt-99",
                           "messages": [{"role": "user", "content": "hi"}]},
                     headers=AUTH)
    assert missing.status_code == 404
    assert "not available" in missing.json()["detail"]
    assert missing.json()["error"]["type"] == "not_found_error"


# --- Retry-After honesty ------------------------------------------------------


class _Req:
    async def is_disconnected(self) -> bool:
        return False


def _queue_timeout_exc(pool) -> HTTPException:
    async def scenario():
        with pytest.raises(HTTPException) as ei:
            await _acquire_slot_or_499(pool, _Req(), threading.Event(), 0.05, True)
        return ei.value
    return asyncio.run(scenario())


def test_queue_timeout_429_carries_retry_after_only_with_a_real_horizon():
    """A cooling account's remaining cooldown is the one known horizon."""
    pool = AccountPool([AccountConfig(email="a@x.com", token="tok")])
    AccountPool.mark_bad(pool.get("a@x.com"), seconds=600)

    exc = _queue_timeout_exc(pool)

    assert exc.status_code == 429
    assert 1 <= int(exc.headers["Retry-After"]) <= 600


def test_queue_timeout_429_has_no_retry_after_when_accounts_are_merely_busy():
    """A stream ends when the upstream finishes: no horizon, no header."""
    pool = AccountPool([AccountConfig(email="a@x.com", token="tok")])
    held = pool.acquire(timeout=1)
    try:
        exc = _queue_timeout_exc(pool)
    finally:
        AccountPool.release(held)

    assert exc.status_code == 429
    assert not exc.headers, f"fabricated a retry horizon: {exc.headers}"


def test_empty_pool_503_has_no_retry_after():
    """Only a config change fixes an empty pool; waiting fixes nothing."""
    exc = _queue_timeout_exc(AccountPool([]))

    assert exc.status_code == 503
    assert not exc.headers
