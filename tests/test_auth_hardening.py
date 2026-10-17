"""Auth + robustness hardening for the config loader and the HTTP surface.

Every test here fails on the pre-fix code. The defects are grouped by the
blast radius they remove:

* ``load_settings`` kept blank API keys (``"keys": [""]``) and let a
  ``DEEPSEAPORT_KEYS=","`` override wipe the file's keys, so the server ran
  with an empty (i.e. open) key list that looked configured.
* ``_check_auth`` compared keys with ``in`` (timing- and blank-key friendly).
* ``/health`` served the account identifier, ``/v1/waf/status`` served the
  absolute obscura binary + profile paths, both without a key.
* ``POST /v1/accounts`` was unbounded in count, field length and provider id;
  every accepted call rewrites the whole config file.
* ``max_retries`` was floored at 0 but never capped, so a config value turned
  one request into 5,000,000 upstream attempts.
* Requests were unbounded in bytes and tool count.
* ``POST /v1/accounts/select`` rewrote config.json on every no-op poll.
* ``serve`` said nothing about a 0.0.0.0 bind.
* The dry-run dump printed a multi-megabyte block on the event loop.

The limits asserted below are the documented ones in the source:
``MAX_RETRIES_LIMIT``, ``MAX_REQUEST_BYTES`` (8 MB), ``MAX_TOOLS_PER_REQUEST``
(128), ``MAX_ACCOUNTS`` (64), ``MAX_IDENTIFIER_CHARS`` (512),
``MAX_SECRET_CHARS`` (4096), ``UVICORN_LIMIT_CONCURRENCY`` (256) and
``UVICORN_TIMEOUT_KEEP_ALIVE`` (15).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import itertools
import json
import logging
import types

import pytest
import uvicorn
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deepseaport import dry_run as dry_run_mod
from deepseaport import server_runner
from deepseaport.accounts import AccountPool
from deepseaport.config import (
    MAX_RETRIES_LIMIT,
    AccountConfig,
    Settings,
    load_settings,
)
from deepseaport.core import registry
from deepseaport.core.provider import ModelSpec, Provider
from deepseaport.server import create_app
from support import settings as _settings, write_config

from deepseaport.server_routes import _check_auth, _prepare
from deepseaport.server_routes import (
    MAX_ACCOUNTS,
    MAX_IDENTIFIER_CHARS,
    MAX_REQUEST_BYTES,
    MAX_SECRET_CHARS,
    MAX_TOOLS_PER_REQUEST,
    _key_matches,
)
from deepseaport.server_runner import (
    UVICORN_LIMIT_CONCURRENCY,
    UVICORN_TIMEOUT_KEEP_ALIVE,
    lan_exposure_notice,
)



def _client(settings) -> TestClient:
    return TestClient(create_app(settings), raise_server_exceptions=False)


def _tools(count: int) -> list:
    return [{"type": "function",
             "function": {"name": f"tool_{i}", "description": "d",
                          "parameters": {"type": "object", "properties": {}}}}
            for i in range(count)]


# A1 -- keys normalization ------------------------------------------------------


@pytest.mark.parametrize("raw_keys", [[""], ["  "], ["ok", "  "]])
def test_blank_config_keys_are_dropped_at_load(tmp_path, raw_keys):
    cfg = write_config(tmp_path, {"keys": raw_keys})
    keys = load_settings(str(cfg)).keys
    assert keys == [k for k in raw_keys if k.strip()]


@pytest.mark.parametrize("raw_keys", [[""], ["  "]])
def test_blank_config_keys_leave_documented_open_gate(tmp_path, raw_keys):
    """A blank key used to authenticate the empty token of a header-less call.

    Post-fix the config normalizes to *no* keys, which is the documented
    local-dev default: the gate is open, but nothing is left masquerading as
    configured auth. Add one real key to get a gate (next test).
    """
    cfg = write_config(tmp_path, {"keys": raw_keys})
    settings = load_settings(str(cfg))
    assert settings.keys == []
    assert _client(settings).get("/v1/accounts").status_code == 200


def test_blank_key_next_to_a_real_one_keeps_the_gate(tmp_path):
    """The mixed case: the blank entry is dropped, the real key still gates."""
    cfg = write_config(tmp_path, {"keys": ["real", "  ", ""]})
    settings = load_settings(str(cfg))
    assert settings.keys == ["real"]
    client = _client(settings)
    assert client.get("/v1/accounts").status_code == 401
    assert client.get("/v1/accounts",
                      headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/accounts",
                      headers={"Authorization": "Bearer real"}).status_code == 200


def test_env_keys_parsing_to_zero_keeps_file_keys_and_warns(tmp_path, caplog):
    cfg = write_config(tmp_path, {"keys": ["filekey"]})
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        import os

        os.environ["DEEPSEAPORT_KEYS"] = ","
        try:
            settings = load_settings(str(cfg))
        finally:
            del os.environ["DEEPSEAPORT_KEYS"]
    assert settings.keys == ["filekey"]
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "DEEPSEAPORT_KEYS" in warned
    assert "AUTH DISABLED" in warned


def test_env_keys_parsing_to_zero_still_enforces_file_key(tmp_path):
    cfg = write_config(tmp_path, {"keys": ["filekey"]})
    import os

    os.environ["DEEPSEAPORT_KEYS"] = " , "
    try:
        client = _client(load_settings(str(cfg)))
    finally:
        del os.environ["DEEPSEAPORT_KEYS"]
    assert client.get("/v1/accounts").status_code == 401
    ok = client.get("/v1/accounts", headers={"Authorization": "Bearer filekey"})
    assert ok.status_code == 200


def test_env_keys_clean_parse_still_overrides(monkeypatch, tmp_path):
    """The override keeps working when it parses to at least one key."""
    cfg = write_config(tmp_path, {"keys": ["filekey"]})
    monkeypatch.setenv("DEEPSEAPORT_KEYS", " a ,b,, c ")
    assert load_settings(str(cfg)).keys == ["a", "b", "c"]


# A2 -- _check_auth -------------------------------------------------------------


def test_wrong_key_and_missing_header_are_401():
    client = _client(_settings(keys=["secret"]))
    assert client.get("/v1/accounts",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/accounts").status_code == 401
    assert client.get("/v1/accounts",
                      headers={"Authorization": "Bearer secret"}).status_code == 200


def test_blank_configured_key_never_matches_the_empty_token():
    """A "" key is not a wildcard for every header-less request."""
    with pytest.raises(HTTPException) as ei:
        _check_auth(_settings(keys=[""]), "")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei2:
        _check_auth(_settings(keys=["", "real"]), "   ")
    assert ei2.value.status_code == 401


def test_key_match_is_length_and_content_exact():
    assert _key_matches("secret", ["secret"]) is True
    assert _key_matches("sec", ["secret"]) is False
    assert _key_matches("secretx", ["secret"]) is False
    assert _key_matches("anything", []) is False


def test_non_ascii_key_compares_instead_of_raising():
    """compare_digest raises TypeError on non-ASCII str; the gate must 401.

    Checked through _check_auth, not over HTTP: httpx refuses to put a
    non-ASCII value in a header, so the wire form cannot be constructed here.
    """
    _check_auth(_settings(keys=["kéy"]), "Bearer kéy")  # must not raise
    with pytest.raises(HTTPException) as ei:
        _check_auth(_settings(keys=["kéy"]), "Bearer nope")
    assert ei.value.status_code == 401


# A5 -- max_retries clamp -------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [(1000000, MAX_RETRIES_LIMIT), (-1, 0),
                                          (3, 3), ("not-a-number", 1)])
def test_max_retries_clamps_to_documented_range(tmp_path, raw, expected):
    cfg = write_config(tmp_path, {"max_retries": raw})
    assert load_settings(str(cfg)).max_retries == expected


def test_max_retries_env_also_clamped(monkeypatch, tmp_path):
    cfg = write_config(tmp_path, {})
    monkeypatch.setenv("DEEPSEAPORT_MAX_RETRIES", "99")
    assert load_settings(str(cfg)).max_retries == MAX_RETRIES_LIMIT


# A3 -- /health redaction -------------------------------------------------------


def test_health_does_not_leak_the_account_identifier():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="tok")],
                         active_account="a@x.com")
    data = _client(settings).get("/health").json()
    assert "a@x.com" not in json.dumps(data)
    assert data["current"] is True
    assert data["ok"] is True and data["ready"] is True


# A4 -- /v1/waf/status auth -----------------------------------------------------


def test_waf_status_requires_a_key():
    client = _client(_settings(keys=["secret"]))
    assert client.get("/v1/waf/status").status_code == 401
    ok = client.get("/v1/waf/status", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_waf_status_stays_open_without_configured_keys():
    """No keys = documented local-dev default, gate included."""
    assert _client(_settings(keys=[])).get("/v1/waf/status").status_code == 200


# A4b -- /v1/models auth (the one unauthenticated /v1/* route) ------------------


def test_models_requires_a_key():
    """Pre-change /v1/models answered unauthenticated while the README said
    every /v1/* route needs the key. /health stays open; this does not."""
    client = _client(_settings(keys=["secret"]))
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200, ok.text
    assert [m["id"] for m in ok.json()["data"]]


def test_models_stays_open_without_configured_keys():
    """No keys = documented local-dev default, gate included; the payload is
    unchanged (version + object + data)."""
    resp = _client(_settings(keys=[])).get("/v1/models")
    assert resp.status_code == 200, resp.text
    assert resp.json()["object"] == "list"
    assert resp.json()["data"]


# A6 -- POST /v1/accounts bounds ------------------------------------------------


def _accounts(count: int) -> list:
    """Distinct accounts: a shared token makes AccountPool treat them as dups."""
    return [AccountConfig(email=f"a{i}@x.com", token=f"tok{i}") for i in range(count)]


def test_add_account_over_cap_conflicts_and_keeps_the_list():
    settings = _settings(accounts=_accounts(MAX_ACCOUNTS))
    client = _client(settings)
    resp = client.post("/v1/accounts", json={"email": "new@x.com", "token": "t"})
    assert resp.status_code == 409, resp.text
    assert str(MAX_ACCOUNTS) in resp.json()["detail"]
    assert len(settings.accounts) == MAX_ACCOUNTS
    assert len(client.get("/v1/accounts").json()["data"]) == MAX_ACCOUNTS


def test_add_account_accepts_the_last_slot():
    settings = _settings(accounts=_accounts(MAX_ACCOUNTS - 1))
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "last@x.com", "token": "t"})
    assert resp.status_code == 201, resp.text
    assert len(settings.accounts) == MAX_ACCOUNTS


@pytest.mark.parametrize("pid", ["nope-not-real", "bad id!", "deepseeek"])
def test_add_account_rejects_unknown_provider(pid):
    settings = _settings()
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "new@x.com", "token": "t",
                                        "provider": pid})
    assert resp.status_code == 400, resp.text
    assert "provider" in resp.json()["detail"]
    assert settings.accounts == []


def test_add_account_accepts_known_provider():
    settings = _settings()
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "new@x.com", "token": "t",
                                        "provider": "deepseek"})
    assert resp.status_code == 201, resp.text
    assert settings.accounts[0].provider == "deepseek"


@pytest.mark.parametrize("field,cap", [("email", MAX_IDENTIFIER_CHARS),
                                       ("mobile", MAX_IDENTIFIER_CHARS),
                                       ("password", MAX_SECRET_CHARS),
                                       ("token", MAX_SECRET_CHARS)])
def test_add_account_rejects_oversized_fields(field, cap):
    settings = _settings()
    body = {"email": "new@x.com", "token": "t", field: "x" * (cap + 1)}
    resp = _client(settings).post("/v1/accounts", json=body)
    assert resp.status_code == 400, resp.text
    assert field in resp.json()["detail"]
    assert settings.accounts == []


def test_add_account_accepts_fields_at_the_cap():
    settings = _settings()
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "e" * MAX_IDENTIFIER_CHARS,
                                        "token": "t" * MAX_SECRET_CHARS})
    assert resp.status_code == 201, resp.text
    assert len(settings.accounts[0].email) == MAX_IDENTIFIER_CHARS


# A6b -- the account cap is per PROVIDER ---------------------------------------


def _full_provider_settings(count: int, provider: str = "deepseek") -> Settings:
    """``count`` accounts bound to ``provider``, on a 2-provider deployment.

    ``_is_duplicate`` keys on provider as well as identity, so the tokens only
    have to be distinct within this provider.
    """
    return _settings(providers=["deepseek", "qwen"],
                     accounts=[AccountConfig(email=f"{provider}-{i}@x.com",
                                             token=f"{provider}-tok-{i}",
                                             provider=provider)
                               for i in range(count)])


def test_the_account_cap_is_counted_per_provider_not_per_config():
    """MAX_ACCOUNTS bounds a POOL, and pools.accounts_for_provider partitions.

    The old check counted ``len(settings.accounts)``, so this deployment -- 64
    DeepSeek accounts plus 63 Qwen ones, 127 in total -- was refused the
    account it needed even though neither pool was full.
    """
    settings = _full_provider_settings(MAX_ACCOUNTS, "deepseek")
    settings.accounts.extend(
        AccountConfig(email=f"qwen-{i}@x.com", token=f"qw-tok-{i}", provider="qwen")
        for i in range(MAX_ACCOUNTS - 1))
    assert len(settings.accounts) > MAX_ACCOUNTS

    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "qw-new@x.com", "token": "tok-new",
                                        "provider": "qwen"})

    assert resp.status_code == 201, resp.text
    assert settings.accounts[-1].provider == "qwen"


def test_the_cap_still_409s_for_the_provider_that_is_actually_full():
    settings = _full_provider_settings(MAX_ACCOUNTS, "deepseek")
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "ds-new@x.com", "token": "tok-new"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert str(MAX_ACCOUNTS) in detail, detail
    assert "deepseek" in detail, f"the 409 does not say which pool is full: {detail}"
    assert len(settings.accounts) == MAX_ACCOUNTS


def test_the_cap_counts_the_coerced_provider_when_none_was_chosen():
    """An unregistered configured provider is stored as deepseek, so it is
    deepseek's pool that fills up -- and deepseek's cap that applies."""
    settings = _settings(provider="junk-provider",
                         accounts=[AccountConfig(email=f"a{i}@x.com", token=f"t{i}")
                                   for i in range(MAX_ACCOUNTS)])
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "new@x.com", "token": "t"})
    assert resp.status_code == 409, resp.text
    assert "deepseek" in resp.json()["detail"]


# A6c -- PUT .../token enforces the same secret cap ---------------------------


def test_set_account_token_enforces_the_secret_cap():
    """PUT had no cap at all: it persisted whatever arrived, verbatim.

    Same credential-store argument as POST (see _reject_oversized_fields), and
    the helper is a pure predicate, so calling it on a second route is free.
    """
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="old")])
    resp = _client(settings).put(
        "/v1/accounts/a@x.com/token",
        json={"token": "x" * (MAX_SECRET_CHARS + 1)})

    assert resp.status_code == 400, resp.text
    assert str(MAX_SECRET_CHARS) in resp.json()["detail"]
    assert settings.accounts[0].token == "old", "the oversized token was stored"


def test_set_account_token_accepts_a_token_at_the_cap():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="old")])
    token = "t" * MAX_SECRET_CHARS
    resp = _client(settings).put("/v1/accounts/a@x.com/token", json={"token": token})

    assert resp.status_code == 200, resp.text
    assert settings.accounts[0].token == token


# A6d -- the duplicate 409 echoes no token fragment ----------------------------


def test_the_duplicate_409_echoes_no_part_of_the_token():
    """``AccountConfig.identifier`` is ``token[:10] + "..."`` for a token-only
    account, and ``AccountPool.add`` put it straight into the 409 detail.

    Same rule as ``_reject_token_shaped_identifier``: nothing derived from the
    token reaches an HTTP response, because a response is logged, cached and
    forwarded far more freely than a URL is scrubbed.
    """
    token = "sk-deepseek-" + "A1b2C3d4E5f6G7h8" * 3 + "Zz9"
    settings = _settings(accounts=[AccountConfig(token=token)])

    resp = _client(settings).post("/v1/accounts", json={"token": token})

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "already in pool" in detail
    assert token[:10] not in detail, f"the 409 echoed a token fragment: {detail!r}"
    assert "..." not in detail, f"the identifier shape leaked: {detail!r}"


def test_the_duplicate_409_still_names_the_account_when_it_has_an_identity():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="t-1")])
    resp = _client(settings).post("/v1/accounts",
                                  json={"email": "a@x.com", "token": "t-2"})
    assert resp.status_code == 409, resp.text
    assert "a@x.com" in resp.json()["detail"]


# A7 -- body size + tool count --------------------------------------------------


def test_oversized_body_rejected_413_before_parse():
    """8 MB+ of non-JSON: 413 from Content-Length, never a parse attempt."""
    resp = _client(_settings()).post(
        "/v1/chat/completions",
        content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"content-type": "application/json"})
    assert resp.status_code == 413, resp.text
    assert "exceeds" in resp.json()["detail"]


def test_body_at_the_cap_is_not_rejected_by_the_guard():
    """Only the guard's verdict is asserted: the payload itself is not JSON."""
    resp = _client(_settings()).post(
        "/v1/chat/completions",
        content=b"x" * MAX_REQUEST_BYTES,
        headers={"content-type": "application/json"})
    assert resp.status_code != 413


def test_ordinary_request_and_stream_survive_the_middleware():
    class _FakeProvider(Provider):
        id = "fakep"
        display_name = "FakeP"

        def models(self) -> dict[str, ModelSpec]:
            return {"fakep-x": ModelSpec(id="fakep-x", owned_by="fakep")}

        # The real providers own the slot lock and release it; the fake has to
        # do the same or the second request queues behind the first for 90s.
        def complete_request(self, app, pool, prep, item, cancel_event):
            AccountPool.release(item)
            return {"content": "P", "thinking": "", "usage_total": 1}

        async def stream_response(self, app, pool, item, prep):
            try:
                yield b'data: {"choices": [{"delta": {"content": "P"}}]}\n\n'
                yield b"data: [DONE]\n\n"
            finally:
                AccountPool.release(item)

    _fake = _FakeProvider()
    registry._REGISTRY[registry._key(_fake)] = _fake
    try:
        # fakep-x only resolves, and only has a pool, when fakep is active and
        # owns an account.
        client = _client(_settings(
            providers=["deepseek", "fakep"],
            accounts=[AccountConfig(email="f@x.com", token="t", provider="fakep")]))
        plain = client.post("/v1/chat/completions", json={
            "model": "fakep-x", "messages": [{"role": "user", "content": "hi"}]})
        assert plain.status_code == 200, plain.text
        streamed = client.post("/v1/chat/completions", json={
            "model": "fakep-x", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
        assert streamed.status_code == 200, streamed.text
        assert "text/event-stream" in streamed.headers["content-type"]
        assert streamed.text.rstrip().endswith("data: [DONE]")
    finally:
        registry._REGISTRY.pop("fakep", None)


def test_tool_count_over_the_cap_is_400():
    body = {"model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _tools(MAX_TOOLS_PER_REQUEST + 1)}
    resp = _client(_settings()).post("/v1/chat/completions", json=body)
    assert resp.status_code == 400, resp.text
    assert "too many tools" in resp.json()["detail"]


def test_prepare_caps_tools_at_the_limit():
    settings = _settings()
    at_limit = _prepare({"model": "deepseek-flash",
                         "messages": [{"role": "user", "content": "hi"}],
                         "tools": _tools(MAX_TOOLS_PER_REQUEST)}, settings)
    assert len(at_limit["tools"]) == MAX_TOOLS_PER_REQUEST
    with pytest.raises(HTTPException) as ei:
        _prepare({"model": "deepseek-flash",
                  "messages": [{"role": "user", "content": "hi"}],
                  "tools": _tools(MAX_TOOLS_PER_REQUEST + 1)}, settings)
    assert ei.value.status_code == 400


def test_tool_cap_only_bites_when_tools_are_rendered():
    """Gated-off tool lists cost nothing, so they stay accepted."""
    body = {"model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _tools(MAX_TOOLS_PER_REQUEST + 50)}
    off = _prepare(dict(body), _settings(enable_tools=False))
    assert off["tools"] == []
    disabled_by_choice = _prepare(dict(body, tool_choice="none"), _settings())
    assert disabled_by_choice["tools"] == []


# A8 -- /v1/accounts/select no-op ----------------------------------------------


def _count_saves(settings) -> list:
    """Count Settings.save() calls made through this instance.

    The route resolves ``settings.save`` at call time, so an instance
    attribute is a valid seam (an unbound function: no descriptor binding for
    instance attributes, hence the zero-arg closure).
    """
    calls: list = []
    real = type(settings).save

    def _spy():
        calls.append(settings)
        return real(settings)

    settings.save = _spy
    return calls


def test_select_account_twice_saves_once():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="t1"),
                                   AccountConfig(email="b@x.com", token="t2")])
    client = _client(settings)
    first = client.post("/v1/accounts/select", json={"identifier": "b@x.com"})
    assert first.status_code == 200 and first.json() == {"ok": True,
                                                         "current": "b@x.com"}
    calls = _count_saves(settings)
    second = client.post("/v1/accounts/select", json={"identifier": "b@x.com"})
    assert second.status_code == 200
    assert second.json() == {"ok": True, "current": "b@x.com"}
    assert calls == []  # re-selecting CURRENT is a read, not a config rewrite


def test_select_account_empty_body_twice_saves_once():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="t1")])
    client = _client(settings)
    assert client.post("/v1/accounts/select", json={}).json() == {"ok": True,
                                                                 "current": None}
    calls = _count_saves(settings)
    assert client.post("/v1/accounts/select", json={}).json() == {"ok": True,
                                                                 "current": None}
    assert calls == []


def test_select_account_still_saves_on_a_real_change():
    settings = _settings(accounts=[AccountConfig(email="a@x.com", token="t1"),
                                   AccountConfig(email="b@x.com", token="t2")])
    client = _client(settings)
    calls = _count_saves(settings)
    assert client.post("/v1/accounts/select",
                       json={"identifier": "b@x.com"}).status_code == 200
    assert len(calls) == 1
    client.post("/v1/accounts/select", json={})
    assert len(calls) == 2  # clearing a real selection still persists


# A9 -- LAN exposure notice ----------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "::1", "localhost",
                                  "[::1]"])
def test_lan_exposure_notice_silent_on_loopback(host):
    assert lan_exposure_notice(_settings(keys=[]), host, 5001) == ""


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5"])
def test_lan_exposure_notice_warns_for_non_loopback(host):
    notice = lan_exposure_notice(_settings(keys=["secret"]), host, 5001)
    assert notice
    assert host in notice and "5001" in notice
    assert "/v1/accounts" in notice
    assert "settings.keys" in notice


def test_lan_exposure_notice_names_the_key_count():
    notice = lan_exposure_notice(_settings(keys=["a", "b"]), "0.0.0.0", 5001)
    assert "2" in notice and "AUTH DISABLED" not in notice


def test_lan_exposure_notice_says_auth_disabled_without_keys():
    notice = lan_exposure_notice(_settings(keys=[]), "0.0.0.0", 5001)
    assert "AUTH DISABLED" in notice


def test_run_server_prints_exposure_and_auth_state_before_starting(capsys,
                                                                  monkeypatch):
    """The notice must be visible before the first request can hit the port."""
    seen: dict = {}

    def _fake_config(app, **kw):
        seen["config"] = kw
        return types.SimpleNamespace()

    class _FakeServer:
        def __init__(self, config):
            seen["config_obj"] = config

        def run(self):
            seen["out_at_run"] = capsys.readouterr().out

    monkeypatch.setattr(uvicorn, "Config", _fake_config)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(server_runner, "_watch_stop_keys", lambda *a, **k: None)
    settings = _settings(keys=["secret"], listen=True)
    assert server_runner._run_server(
        settings, argparse.Namespace(host="", port=0),
        ensure_free=lambda s, h, p: p) == 0
    printed = seen["out_at_run"]
    assert "0.0.0.0" in printed and "5001" in printed
    assert "1 API key(s) required" in printed
    assert "non-loopback" in printed
    assert seen["config"]["limit_concurrency"] == UVICORN_LIMIT_CONCURRENCY
    assert seen["config"]["timeout_keep_alive"] == UVICORN_TIMEOUT_KEEP_ALIVE


def test_run_server_says_auth_disabled_on_loopback(capsys, monkeypatch):
    monkeypatch.setattr(uvicorn, "Config", lambda app, **kw: types.SimpleNamespace())
    monkeypatch.setattr(uvicorn, "Server", lambda config: types.SimpleNamespace(
        run=lambda: None))
    monkeypatch.setattr(server_runner, "_watch_stop_keys", lambda *a, **k: None)
    assert server_runner._run_server(
        _settings(keys=[]), argparse.Namespace(host="127.0.0.1", port=0),
        ensure_free=lambda s, h, p: p) == 0
    out = capsys.readouterr().out
    assert "127.0.0.1" in out
    assert "DISABLED" in out
    assert "non-loopback" not in out  # no LAN warning on a loopback bind


# A11 -- uvicorn limits on every Config we own ----------------------------------


def test_background_server_config_carries_limits(monkeypatch):
    seen: dict = {}

    monkeypatch.setattr(uvicorn, "Config",
                        lambda app, **kw: seen.update(kw) or types.SimpleNamespace())
    monkeypatch.setattr(uvicorn, "Server", lambda config: types.SimpleNamespace(
        run=lambda: None))
    app = create_app(_settings(keys=[]))
    server_runner._start_background_server(app, "127.0.0.1", 5001)
    assert seen["limit_concurrency"] == UVICORN_LIMIT_CONCURRENCY
    assert seen["timeout_keep_alive"] == UVICORN_TIMEOUT_KEEP_ALIVE
    assert seen["access_log"] is False  # unchanged chat behaviour


def test_dry_run_config_carries_limits_and_keeps_access_log_off(monkeypatch,
                                                                tmp_path):
    seen: dict = {}
    monkeypatch.setattr(uvicorn, "Config",
                        lambda app, **kw: seen.update(kw) or types.SimpleNamespace())
    monkeypatch.setattr(uvicorn, "Server", lambda config: types.SimpleNamespace(
        run=lambda: None))
    monkeypatch.setattr("deepseaport.cli._ensure_free_port",
                        lambda s, h, p: p)
    monkeypatch.setattr(server_runner, "_watch_stop_keys", lambda *a, **k: None)
    assert dry_run_mod.run_dry_run(
        _settings(config_path=str(tmp_path / "c.json")),
        argparse.Namespace(host="127.0.0.1", port=0)) == 0
    assert seen["limit_concurrency"] == UVICORN_LIMIT_CONCURRENCY
    assert seen["timeout_keep_alive"] == UVICORN_TIMEOUT_KEEP_ALIVE
    assert seen["access_log"] is False


# A12 -- dry-run print off the event loop ---------------------------------------


def test_dry_run_dump_is_printed_off_the_event_loop(monkeypatch, capsys):
    """json.dumps + flush used to run inline in the async handler."""
    # Private counter: the module-level one is process-global and monotonic,
    # and this test must not steal the id another test asserts on.
    monkeypatch.setattr(dry_run_mod, "_REQUEST_IDS", itertools.count(1))
    seen: list = []
    real_to_thread = asyncio.to_thread

    async def _spy(fn, *args, **kwargs):
        seen.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(dry_run_mod.asyncio, "to_thread", _spy)
    settings = Settings(config_path="", append_top="HDR")
    resp = TestClient(dry_run_mod.create_dry_app(settings)).post(
        "/v1/chat/completions",
        json={"model": "deepseek-flash",
              "messages": [{"role": "user", "content": "hello"}]})
    assert resp.status_code == 200
    assert dry_run_mod._print_dry_run in seen
    content = resp.json()["choices"][0]["message"]["content"]
    out = capsys.readouterr().out
    assert "NOT sent to DeepSeek" in out
    assert content in out  # printed output and reply stay byte-identical


def test_print_dry_run_stays_a_sync_module_function(capsys):
    """TUI/CLI call it directly; it must not become a coroutine."""
    assert not inspect.iscoroutinefunction(dry_run_mod._print_dry_run)
    prep = {"model": "deepseek-flash", "stream": False, "tools": [],
            "prompt": "hello", "thinking": False, "search": False}
    dry_run_mod._print_dry_run(99, {"model": "deepseek-flash"}, prep)
    out = capsys.readouterr().out
    assert "dry run #99" in out
def test_blank_config_keys_warn_that_auth_is_disabled(tmp_path, caplog):
    """`"keys": [""]` used to be a silent bypass; now it must say so.

    A hand-edited config whose only key entries are blank is the "I want to
    require *something*" mistake. Dropping the blanks makes it behave like
    `"keys": []` (the documented open dev default), so the operator has to be
    told that the port is not actually protected.
    """
    import json
    import logging

    from deepseaport.config import load_settings

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"keys": ["", "   "], "accounts": []}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        settings = load_settings(str(cfg))
    assert settings.keys == []
    assert "AUTH DISABLED" in caplog.text
    # The count phrase, not a bare "2": the old assertion was satisfied by any
    # line number in the file containing a 2 (the logger name, a source line),
    # so it proved nothing about the two blank entries being counted.
    assert "has 2 API key entr(y/ies) but none of them is usable" in caplog.text

    # And the count really is the input length, not a constant: one blank entry
    # says 1, three say 3. A hardcoded "2" in the message would pass the
    # previous assertion for any input.
    for n in (1, 3):
        other = tmp_path / f"c{n}.json"
        other.write_text(json.dumps({"keys": [""] * n, "accounts": []}),
                         encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
            assert load_settings(str(other)).keys == []
        assert f"has {n} API key entr(y/ies)" in caplog.text, caplog.text


def test_absent_keys_stay_silent(tmp_path, caplog):
    """An ordinary open-by-default config must not cry wolf."""
    import json
    import logging

    from deepseaport.config import load_settings

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        assert load_settings(str(cfg)).keys == []
    assert "AUTH DISABLED" not in caplog.text


def test_real_keys_stay_silent(tmp_path, caplog):
    import json
    import logging

    from deepseaport.config import load_settings

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"keys": ["sk-ok", ""], "accounts": []}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        assert load_settings(str(cfg)).keys == ["sk-ok"]
    assert "AUTH DISABLED" not in caplog.text
