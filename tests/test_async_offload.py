"""Blocking route work must run off the event loop (F8), plus the additive
/v1/models and /health payload keys (F29).

**F8** ``server_routes.py`` called ``_prepare`` (message render and, on
opencode, a model-catalogue socket read -- 2s live, up to 30s while the managed
serve boots) and ``build_models_list`` INLINE from async handlers; the dry-run
app did the same for both. One cold-cache request froze the whole server --
``/health`` and every ``_watch_disconnect`` watcher included -- for the length
of the read. All four call sites now hop through ``asyncio.to_thread``.

The seam: the helper is monkeypatched to record ``threading.get_ident()``, and
the route is driven through ``httpx.ASGITransport`` from ``asyncio.run`` on the
MAIN thread -- which is therefore also the event-loop thread -- so "ran on the
loop" is exactly ``ident == threading.get_ident()``.

**F29** rides along here because it is the same two routes: ``/v1/models``
entries gained the model's computed thinking/search flags as extra top-level
keys, and ``/health`` gained per-provider account counts.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from zourcewell import dry_run as dry_run_mod
from zourcewell import server_routes
from zourcewell.config import AccountConfig
from zourcewell.dry_run import create_dry_app
from zourcewell.server import MODELS, create_app
from support import settings as _settings

MSGS = [{"role": "user", "content": "hi"}]
MAIN = threading.get_ident()


def _recording_prepare(seen: dict):
    """Stand-in for _prepare: records the thread, then raises like the real one.

    A raise is the cheapest proof that the worker's HTTPException (the "400/404
    fast" path) survives the thread hop intact.
    """
    def prepare(body, settings=None):
        seen["prepare"] = threading.get_ident()
        raise HTTPException(status_code=400, detail="recorded by the stand-in")
    return prepare


def _recording_models(seen: dict):
    def models_list(settings, now=None):
        seen["models"] = threading.get_ident()
        return []
    return models_list


async def _request(app, method: str, path: str, json=None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://ds.test") as client:
        return await client.request(method, path, json=json)


def _assert_off_loop(seen: dict, key: str) -> None:
    assert key in seen, f"{key} was never called: the seam moved"
    assert seen[key] != MAIN, (
        f"{key} ran on the event loop thread: a blocking read here freezes "
        f"/health and every disconnect watcher")


# --- F8: live route -----------------------------------------------------------


def test_live_prepare_runs_off_the_event_loop(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(server_routes, "_prepare", _recording_prepare(seen))
    app = create_app(_settings())

    resp = asyncio.run(_request(app, "POST", "/v1/chat/completions",
                                json={"model": "deepseek-flash", "messages": MSGS}))

    _assert_off_loop(seen, "prepare")
    # Same 400 the inline call produced: the hop changed the thread, nothing else.
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "recorded by the stand-in"


def test_live_models_list_runs_off_the_event_loop(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(server_routes, "build_models_list",
                        _recording_models(seen))
    app = create_app(_settings())

    resp = asyncio.run(_request(app, "GET", "/v1/models"))

    _assert_off_loop(seen, "models")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == []


# --- F8: dry-run app (its own copies of both calls) ---------------------------


def test_dry_prepare_runs_off_the_event_loop(monkeypatch):
    seen: dict = {}
    # dry_run imported the names by value: patch ITS namespace, not the source.
    monkeypatch.setattr(dry_run_mod, "_prepare", _recording_prepare(seen))
    app = create_dry_app(_settings())

    resp = asyncio.run(_request(app, "POST", "/v1/chat/completions",
                                json={"model": "deepseek-flash", "messages": MSGS}))

    _assert_off_loop(seen, "prepare")
    assert resp.status_code == 400, resp.text


def test_dry_models_list_runs_off_the_event_loop(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(dry_run_mod, "build_models_list",
                        _recording_models(seen))
    app = create_dry_app(_settings())

    resp = asyncio.run(_request(app, "GET", "/v1/models"))

    _assert_off_loop(seen, "models")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == []


# --- F29: additive payload keys ----------------------------------------------


def test_models_entries_carry_the_thinking_and_search_flags():
    """MODELS holds (thinking, search, model_type) per id; the flags were
    computed and then dropped from the listing."""
    data = TestClient(create_app(_settings())).get("/v1/models").json()["data"]
    got = {m["id"]: m for m in data}

    for mid, (thinking, search, _model_type) in MODELS.items():
        assert got[mid]["thinking"] is thinking, mid
        assert got[mid]["search"] is search, mid
    # Additive: the pre-existing keys are untouched.
    assert got["deepseek-flash"]["owned_by"] == "deepseek"
    assert got["deepseek-flash-reasoner"]["thinking"] is True


def test_health_reports_per_provider_account_counts():
    s = _settings(accounts=[
        AccountConfig(email="a@x.com", token="tok"),
        AccountConfig(email="b@x.com", provider="qwen"),
        AccountConfig(email="c@x.com", token="tok", provider="qwen",
                      banned=True),
    ])

    data = TestClient(create_app(s)).get("/health").json()

    assert data["providers"] == {
        "deepseek": {"accounts": 1, "accounts_with_token": 1, "banned": 0},
        "qwen": {"accounts": 2, "accounts_with_token": 1, "banned": 1},
    }
    # Additive: the pre-existing totals keep their values.
    assert data["accounts"] == 3
    assert data["accounts_with_token"] == 2
    assert data["banned"] == 1
