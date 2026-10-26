"""Route-edge hardening for the HTTP surface (P1 findings, review round 1).

Every test here fails on the pre-fix code; the finding each one reproduces is
named above it. Grouped by finding:

* **P1-1** ``FastAPI(title=...)`` left ``/openapi.json``, ``/docs`` and
  ``/redoc`` enabled on both the live and the dry-run app: an unauthenticated,
  machine-readable inventory of every route (including the account CRUD
  surface and the ``{identifier}`` path shape) on a bridge that holds a
  credential store.
* **P1-2** The 8 MB body cap only read ``Content-Length``, so a
  ``Transfer-Encoding: chunked`` request declared nothing, skipped the check and
  got fully materialized + JSON-parsed before ``_check_auth`` ran. The chunked
  cases are built three ways, none of which is "set a header": an
  ``httpx.ASGITransport`` async-generator body (real ``more_body`` receive
  messages, no Content-Length), a hand-written ASGI ``receive`` that counts
  what it hands over, and a live uvicorn server on loopback driven by httpx,
  which does the HTTP/1.1 chunked framing itself for an iterator body -- so the
  real socket and the real h11 dechunker are still the thing under test.
* **P1-3** The 413 log line printed the raw request path, and
  ``accounts._matches`` accepts a FULL 64-char token as an identifier, so
  ``DELETE /v1/accounts/<token>`` wrote a live bearer token into the log before
  routing and before auth.
* **P1-4** The account routes took a full bearer token straight from the URL
  path, which is how secrets end up in access logs, proxy logs and ``Referer``
  chains. Fixed at the HTTP edge (a token-shaped segment is never a legitimate
  ``AccountConfig.identifier``), not in the matcher, which the CLI still needs.
* **P1-5** ``create_dry_app`` builds its own FastAPI app and never called
  ``register_routes``, so the body cap was absent and a dry run echoed and
  printed an unbounded body to the terminal.
* **P1-6** The provider fallback in ``add_account`` skipped the registry check
  the explicit field gets, so an unregistered configured provider id was
  persisted onto every account added without an explicit provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from zourcewell.accounts import AccountPool
from zourcewell.config import AccountConfig, Settings
from zourcewell.core import registry
from zourcewell.dry_run import create_dry_app
from zourcewell.server import create_app
from support import settings as _settings
from zourcewell.server_routes import MAX_REQUEST_BYTES

# 63 chars of the exact shape a real DeepSeek userToken has: [A-Za-z0-9_-].
LIVE_TOKEN = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"
AUTH = {"Authorization": "Bearer sk-test"}
DOC_PATHS = ("/openapi.json", "/docs", "/redoc")
CHUNK = 64 * 1024

# Bounds for the loopback uvicorn server, named so they are greppable and
# tunable in one place. None of these is an assertion about a duration: they
# exist only so a server that never comes up (or never comes down) FAILS with a
# clear message instead of hanging CI. Tuned generously for `windows-latest`,
# where binding a socket and starting the loopback server is far slower than on
# a developer machine -- these are hang guards, not budgets to race.
_SERVER_START_TIMEOUT = 20.0
_SERVER_JOIN_TIMEOUT = 20.0
# Same for the httpx client: a connect or a read that never completes must
# raise TimeoutError, not park the run.
_CLIENT_IO_TIMEOUT = 20.0


def _client(settings) -> TestClient:
    return TestClient(create_app(settings), raise_server_exceptions=False)


# --- chunked request builders --------------------------------------------------
#
# A chunked request is one whose body arrives as several `http.request`
# messages with more_body=True and NO Content-Length header. All three
# builders below assert that property before sending, so none of them can
# quietly degrade into a declared-length request.


async def _chunked_request(app, method: str, path: str, data: bytes, *,
                           chunk: int = CHUNK,
                           headers: dict | None = None) -> httpx.Response:
    """Send ``data`` as a genuinely chunked ASGI request (no Content-Length)."""
    async def body():
        for start in range(0, len(data), chunk):
            yield data[start:start + chunk]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://ds.test") as client:
        request = client.build_request(
            method, path, content=body(),
            headers={"content-type": "application/json", **(headers or {})})
        # The proof this is not a lying header: httpx framed the body as
        # Transfer-Encoding: chunked and never computed a Content-Length.
        assert "content-length" not in request.headers
        assert request.headers.get("transfer-encoding") == "chunked"
        return await client.send(request)


def _chunked_post(app, path: str, data: bytes, **kw) -> httpx.Response:
    return asyncio.run(_chunked_request(app, "POST", path, data, **kw))


def _asgi_post(app, path: str, data: bytes, *, chunk: int = CHUNK,
               headers: dict | None = None) -> tuple[int, int]:
    """Call the ASGI app directly with a chunked body.

    Returns ``(status, bytes_handed_to_the_app)``. The byte count is the
    point: it proves the guard stops pulling at the cap instead of letting the
    whole body be materialized.
    """
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 80),
        "headers": [(k.lower().encode(), v.encode())
                    for k, v in {"content-type": "application/json",
                                 **(headers or {})}.items()],
    }
    assert not any(k == b"content-length" for k, _ in scope["headers"])
    state = {"delivered": 0, "sent": 0}
    messages: list[dict] = []

    async def receive() -> dict:
        if state["sent"] >= len(data):
            return {"type": "http.request", "body": b"", "more_body": False}
        piece = data[state["sent"]:state["sent"] + chunk]
        state["sent"] += len(piece)
        state["delivered"] += len(piece)
        return {"type": "http.request", "body": piece, "more_body": True}

    async def send(message: dict) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    status = next((m["status"] for m in messages
                   if m["type"] == "http.response.start"), 0)
    return status, state["delivered"]


class _LiveServer:
    """A real uvicorn server on loopback, driven by httpx.

    httpx does the HTTP/1.1 chunked framing itself when the body is an
    iterator -- ``Transfer-Encoding: chunked``, no ``Content-Length`` -- and a
    real uvicorn/h11 server dechunks it into incremental ``http.request``
    messages, so the property these two tests exist for survives without the
    hand-rolled ``%x\r\n`` writer or a response-body reader.

    Port 0 plus the post-``startup`` lookup replaces binding a socket here:
    ``Server.startup()`` is what binds, and the readiness Event fires after it
    returns, so the port is known by the time ``__enter__`` hands the server
    back. Readiness stays a delivered signal rather than a poll on
    ``time.time()``, which an NTP correction can move backwards.
    """

    def __init__(self, app) -> None:
        outer = self
        self._started = threading.Event()

        class _Server(uvicorn.Server):
            async def startup(self, sockets=None) -> None:
                await super().startup(sockets)
                # Only after the real startup returned, so a waiter can trust
                # `started` the moment this fires.
                outer._started.set()

        self._server = _Server(uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", access_log=False,
            # The test is about the HTTP layer; the app lifespan builds an
            # ObscuraBridge, and spawning a browser from a body-limit test
            # would be absurd.
            lifespan="off"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_LiveServer":
        self._thread.start()
        if not self._started.wait(_SERVER_START_TIMEOUT):
            raise RuntimeError("uvicorn did not start")
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(_SERVER_JOIN_TIMEOUT)

    def post(self, path: str, data: bytes, *, headers: dict | None = None,
             chunk: int = CHUNK) -> httpx.Response:
        """POST ``data`` as ``chunk``-sized pieces, so httpx frames it."""
        with httpx.Client(base_url=f"http://127.0.0.1:{self.port}",
                          timeout=_CLIENT_IO_TIMEOUT) as client:
            request = client.build_request(
                "POST", path, headers=headers,
                content=iter(data[i:i + chunk]
                             for i in range(0, len(data), chunk)))
            # The same proof the ASGI builder makes: httpx framed the body as
            # Transfer-Encoding: chunked and never computed a Content-Length.
            assert "content-length" not in request.headers
            assert request.headers.get("transfer-encoding") == "chunked"
            return client.send(request)


# --- P1-1: the framework's own docs are an unauthenticated route inventory ----


@pytest.mark.parametrize("path", DOC_PATHS)
def test_framework_docs_are_disabled_on_the_live_app(path):
    resp = _client(_settings()).get(path)
    assert resp.status_code == 404, f"{path} -> {resp.status_code}: {resp.text[:200]}"


@pytest.mark.parametrize("path", DOC_PATHS)
def test_framework_docs_are_disabled_on_the_dry_app(path):
    resp = TestClient(create_dry_app(_settings())).get(path)
    assert resp.status_code == 404, f"{path} -> {resp.status_code}: {resp.text[:200]}"


# --- P1-2: the 8 MB cap must survive a chunked body --------------------------


def test_chunked_body_over_the_cap_is_413():
    settings = _settings(keys=["sk-test"])
    app = create_app(settings)
    body = b"x" * (MAX_REQUEST_BYTES + CHUNK)

    resp = _chunked_post(app, "/v1/accounts", body)

    assert resp.status_code == 413, resp.text
    assert "exceeds" in resp.json()["detail"]
    assert settings.accounts == [], "the body was parsed into an account"


def test_chunked_body_over_the_cap_stops_reading_at_the_cap():
    """The counter must abort the read, not just report after the fact.

    Delivered bytes are asserted against the total: a wrapper that buffered
    the whole body (or let FastAPI do it) would hand over every byte.
    """
    total = MAX_REQUEST_BYTES + 4 * CHUNK
    status, delivered = _asgi_post(create_app(_settings(keys=["sk-test"])),
                                   "/v1/accounts", b"x" * total)

    assert status == 413
    assert delivered < total, (
        f"the app pulled all {delivered} bytes; the cap never stopped the read")
    assert delivered <= MAX_REQUEST_BYTES + CHUNK


def test_normal_chunked_body_still_reaches_the_handler():
    settings = _settings()
    payload = json.dumps({"email": "chunky@x.com", "token": "tok-chunky"}).encode()

    resp = _chunked_post(create_app(settings), "/v1/accounts",
                         payload, chunk=7, headers=AUTH)

    assert resp.status_code == 201, resp.text
    assert [a.email for a in settings.accounts] == ["chunky@x.com"]


def test_live_chunked_post_over_the_cap_gets_413():
    """End to end: a real socket, a real h11 dechunker, no Content-Length.

    Pre-fix this returns 422, not 413: with no Content-Length the guard is
    skipped and the whole body is read and JSON-parsed before _check_auth.
    """
    settings = _settings(keys=["sk-test"])
    with _LiveServer(create_app(settings)) as live:
        resp = live.post("/v1/accounts", b"x" * (MAX_REQUEST_BYTES + 1),
                         headers={"Content-Type": "application/json", **AUTH})

    assert resp.status_code == 413, resp.text
    assert "exceeds" in resp.json()["detail"]
    assert settings.accounts == [], "the body was parsed into an account"


def test_live_chunked_post_within_the_cap_is_accepted():
    settings = _settings()
    payload = json.dumps({"email": "live@x.com", "token": "tok-live"}).encode()

    with _LiveServer(create_app(settings)) as live:
        resp = live.post("/v1/accounts", payload,
                         headers={"Content-Type": "application/json", **AUTH})

    assert resp.status_code == 201, resp.text
    assert [a.email for a in settings.accounts] == ["live@x.com"]


# --- P1-3: the 413 log must never record a secret -----------------------------


def test_oversized_body_log_never_records_a_token_path(caplog):
    settings = _settings(keys=["sk-test"],
                        accounts=[AccountConfig(token=LIVE_TOKEN, email="")])

    with caplog.at_level(logging.DEBUG, logger="zourcewell.server"):
        with _client(settings) as client:
            # A lying Content-Length: the fast path rejects it, so this is the
            # exact request that used to log the path before routing/auth.
            resp = client.request(
                "DELETE", f"/v1/accounts/{LIVE_TOKEN}", content=b"x" * 16,
                headers={**AUTH, "content-length": str(MAX_REQUEST_BYTES + 1)})

    assert resp.status_code == 413, resp.text
    assert LIVE_TOKEN not in caplog.text, "the 413 log wrote a live token"
    assert LIVE_TOKEN not in resp.text
    # The operator still gets the route, just not the path that carried it.
    # The fast path runs before routing, so there is no route template yet:
    # the redacted path is what has to carry the diagnostic value.
    assert "rejecting" in caplog.text
    assert f"DELETE /v1/accounts/<redacted>" in caplog.text
    assert "DELETE" in caplog.text


def test_streamed_oversized_body_logs_the_route_template(caplog):
    """The other half of _log_target: a body aborted mid-parse is already
    routed, so the template is available and the raw path is not needed.

    Pre-change this request produced no 413 and no log line at all: with no
    Content-Length the old middleware never reached its logging branch.
    """
    settings = _settings(keys=["sk-test"],
                        accounts=[AccountConfig(token=LIVE_TOKEN, email="")])

    with caplog.at_level(logging.DEBUG, logger="zourcewell.server"):
        resp = asyncio.run(_chunked_request(
            create_app(settings), "PUT",
            f"/v1/accounts/{LIVE_TOKEN}/token",
            b"x" * (MAX_REQUEST_BYTES + CHUNK), headers=AUTH))

    assert resp.status_code == 413, resp.text
    assert "rejecting" in caplog.text
    assert "PUT /v1/accounts/{identifier}/token" in caplog.text
    assert LIVE_TOKEN not in caplog.text


# --- P1-4: a full bearer token is not a legitimate URL identifier -------------


def _token_pool_settings() -> Settings:
    settings = _settings(keys=["sk-test"],
                        accounts=[AccountConfig(token=LIVE_TOKEN,
                                                email="a@x.com")])
    return settings


def _pin_pool(monkeypatch, settings) -> None:
    import zourcewell.server as srv_mod

    monkeypatch.setattr(srv_mod, "_pool",
                        lambda app: AccountPool(settings.accounts, current=""))


def test_delete_account_rejects_a_full_token_in_the_path(monkeypatch, caplog):
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with caplog.at_level(logging.DEBUG, logger="zourcewell.server"):
        with _client(settings) as client:
            resp = client.delete(f"/v1/accounts/{LIVE_TOKEN}", headers=AUTH)

    assert resp.status_code == 400, resp.text
    assert LIVE_TOKEN not in resp.text, "the 400 echoed the token back"
    assert LIVE_TOKEN not in caplog.text
    assert "identifier" in resp.json()["detail"]
    assert len(settings.accounts) == 1, "the account was removed anyway"


def test_set_account_token_rejects_a_full_token_in_the_path(monkeypatch):
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with _client(settings) as client:
        resp = client.put(f"/v1/accounts/{LIVE_TOKEN}/token",
                          json={"token": "fresh"}, headers=AUTH)

    assert resp.status_code == 400, resp.text
    assert LIVE_TOKEN not in resp.text
    assert settings.accounts[0].token == LIVE_TOKEN


def test_unblock_rejects_a_full_token_identifier(monkeypatch):
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with _client(settings) as client:
        resp = client.post("/v1/accounts/unblock",
                           json={"identifier": LIVE_TOKEN}, headers=AUTH)

    assert resp.status_code == 400, resp.text
    assert LIVE_TOKEN not in resp.text


def test_token_shaped_path_is_rejected_before_auth(monkeypatch):
    """The guard runs ahead of _check_auth, so a token never reaches an
    auth-gated code path no matter which key (or none) is presented."""
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with _client(settings) as client:
        resp = client.delete(f"/v1/accounts/{LIVE_TOKEN}")

    assert resp.status_code == 400, resp.text


def test_delete_account_by_email_still_works(monkeypatch):
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with _client(settings) as client:
        resp = client.delete("/v1/accounts/a@x.com", headers=AUTH)

    assert resp.status_code == 200, resp.text
    assert settings.accounts == []


def test_delete_account_by_short_token_identifier_still_works(monkeypatch):
    """``AccountConfig.identifier`` is token[:10] + "..." for a token-only
    account; that is the form the route must keep accepting."""
    settings = _settings(keys=["sk-test"],
                        accounts=[AccountConfig(token=LIVE_TOKEN, email="")])
    _pin_pool(monkeypatch, settings)
    short = settings.accounts[0].identifier
    assert short == LIVE_TOKEN[:10] + "..."

    with _client(settings) as client:
        resp = client.delete(f"/v1/accounts/{short}", headers=AUTH)

    assert resp.status_code == 200, resp.text
    assert settings.accounts == []


def test_31_char_identifier_is_still_accepted_as_a_shape(monkeypatch):
    """Boundary: the guard starts at 32 chars, so a shorter opaque id is still
    a well-formed identifier and gets the normal 404, not a 400."""
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)
    near_miss = "a" * 31

    with _client(settings) as client:
        resp = client.delete(f"/v1/accounts/{near_miss}", headers=AUTH)

    assert resp.status_code == 404, resp.text


def test_jwt_shaped_identifier_is_rejected_on_every_identifier_route(
        monkeypatch, caplog):
    """A Qwen JWT is three dot-separated base64url segments; the old
    ``[A-Za-z0-9_-]`` class did not cover '.', so DELETE /v1/accounts/<jwt>
    deleted the account and echoed the JWT in "removed" (and a PUT/unblock
    reached the same matcher)."""
    jwt = "aaa." + "b" * 40 + ".ccc"
    settings = _settings(keys=["sk-test"],
                         accounts=[AccountConfig(token=jwt, email="")])
    _pin_pool(monkeypatch, settings)

    with caplog.at_level(logging.DEBUG, logger="zourcewell.server"):
        with _client(settings) as client:
            # Ahead of auth: no key presented, and still the 400.
            unauth = client.delete(f"/v1/accounts/{jwt}")
            delete = client.delete(f"/v1/accounts/{jwt}", headers=AUTH)
            put = client.put(f"/v1/accounts/{jwt}/token",
                             json={"token": "fresh"}, headers=AUTH)
            unblock = client.post("/v1/accounts/unblock",
                                  json={"identifier": jwt}, headers=AUTH)
            select = client.post("/v1/accounts/select",
                                 json={"identifier": jwt}, headers=AUTH)

    for resp in (unauth, delete, put, unblock, select):
        assert resp.status_code == 400, resp.text
        assert jwt not in resp.text, "the 400 echoed the token back"
    assert jwt not in caplog.text
    assert len(settings.accounts) == 1, "the account was removed anyway"


def test_short_dotted_identifier_is_still_a_404(monkeypatch):
    """Control for the widened class: '.' alone must not make every dotted id
    token-shaped -- the 32-char floor still decides."""
    settings = _token_pool_settings()
    _pin_pool(monkeypatch, settings)

    with _client(settings) as client:
        resp = client.delete("/v1/accounts/aaa.bbb.ccc", headers=AUTH)

    assert resp.status_code == 404, resp.text


# --- P1-5: the dry-run app has no body cap of its own ------------------------


def test_dry_app_rejects_an_oversized_declared_body():
    resp = TestClient(create_dry_app(_settings())).post(
        "/v1/chat/completions", content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"content-type": "application/json"})

    assert resp.status_code == 413, resp.text
    assert "exceeds" in resp.json()["detail"]


def test_dry_app_rejects_an_oversized_chunked_body(capsys):
    resp = _chunked_post(create_dry_app(_settings()),
                         "/v1/chat/completions",
                         b"x" * (MAX_REQUEST_BYTES + CHUNK))

    assert resp.status_code == 413, resp.text
    assert "dry run #" not in capsys.readouterr().out, (
        "the dry run printed the body it was supposed to reject")


# --- P1-6: the provider fallback skips the registry check ---------------------


def test_add_account_never_persists_an_unregistered_fallback_provider():
    """ZOURCEWELL_PROVIDER (and config.json) are not validated on load, so
    the fallback branch is exactly as reachable as the explicit field."""
    settings = _settings(provider="junk-provider")

    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "e@x.com", "token": "t"})

    assert resp.status_code == 201, resp.text
    stored = settings.accounts[0].provider
    assert stored != "junk-provider"
    assert stored in registry.registered(), (
        f"stored provider {stored!r} is still unmatchable by any pool")


def test_add_account_keeps_a_registered_fallback_provider():
    settings = _settings(provider="deepseek")

    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "e@x.com", "token": "t"})

    assert resp.status_code == 201, resp.text
    assert settings.accounts[0].provider == "deepseek"


# --- an unscrubbed upstream crash must not become a 500 + traceback ------------
#
# The non-stream route caught only _StreamCancelled. A provider client's own
# RuntimeError -- deepseek/client.py raises "create_session: no id (HTTP 500):
# <raw body>" and "pow challenge: bad shape: <raw body>" -- escaped the handler,
# so the client got a 500 and uvicorn wrote the whole traceback, raw upstream
# body included, to the server log. `_client` here uses
# raise_server_exceptions=False, which is what uvicorn does: the exception
# becomes a response and its message is logged.

# 63 chars of the exact shape a real DeepSeek userToken has, inside a JWT body:
# both redaction shapes in scrub_upstream_text have something to match here.
_CRASH_BODY = ('{"errCode":1004,"userToken":"sk-deepseek-'
               + "A1b2C3d4E5f6G7h8" * 3 + 'Zz9"}')
_CRASH_TOKEN = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"


def _crashing_provider(monkeypatch) -> None:
    """Make the deepseek provider raise the RuntimeError its client raises."""
    provider = registry.get("deepseek")
    assert provider is not None

    def _boom(*_a, **_kw):
        raise RuntimeError(f"create_session: no id (HTTP 500): {_CRASH_BODY}")

    monkeypatch.setattr(provider, "complete_request", _boom)


def test_non_stream_upstream_crash_is_a_scrubbed_502(monkeypatch, caplog):
    _crashing_provider(monkeypatch)
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token=_CRASH_TOKEN)])

    with caplog.at_level(logging.DEBUG):
        with _client(settings) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 502, resp.text
    assert _CRASH_TOKEN not in resp.text, "the 502 echoed the upstream body"
    assert "[redacted]" in resp.json()["detail"], resp.text
    assert _CRASH_TOKEN not in caplog.text, "the log line was not scrubbed"
    assert "create_session" in caplog.text, "the log line lost the diagnosis"


def test_a_provider_http_exception_keeps_its_own_status(monkeypatch):
    """The catch-all must not flatten a decision the engine already made."""
    from fastapi import HTTPException

    provider = registry.get("deepseek")

    def _refuse(*_a, **_kw):
        raise HTTPException(status_code=429, detail="no DeepSeek account free")

    monkeypatch.setattr(provider, "complete_request", _refuse)
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="t")])

    with _client(settings) as client:
        resp = client.post("/v1/chat/completions", json={
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 429, resp.text
    assert "no DeepSeek account free" in resp.json()["detail"]
