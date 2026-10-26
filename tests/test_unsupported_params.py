"""OpenAI params the web upstreams cannot honor: accept, drop, log once (F23).

``_extract_passthrough``'s accept-ignore-log-once tuple covered temperature /
top_p / parallel_tool_calls / n / stop / penalties / response_format, but not
``logprobs``, ``top_logprobs``, ``seed`` or ``logit_bias`` (0 hits anywhere in
``src/``): a client asking for token logprobs or a reproducible sample got a
silent drop -- no log line, no ``unsupported_params`` entry, no way to find out
the field was ignored. They now follow the house rule: accepted, ignored, and
named once per request in the "ignoring unsupported OpenAI params" warning.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from zourcewell.core import registry
from zourcewell.server import create_app
from zourcewell.server_routes import _extract_passthrough, _prepare
from support import settings as _settings

MODEL = "deepseek-flash"
MSGS = [{"role": "user", "content": "hi"}]
PARAMS = {"logprobs": True, "top_logprobs": 3, "seed": 7,
          "logit_bias": {"50256": -100}}


def _body(**kw) -> dict:
    body = {"model": MODEL, "messages": MSGS}
    body.update(kw)
    return body


def _settings_with_account(**kw):
    from zourcewell.config import AccountConfig
    kw.setdefault("accounts",
                  [AccountConfig(email="a@x.com", token="tok")])
    return _settings(**kw)


def _stub_engine(monkeypatch) -> None:
    """No upstream call: the request prep is the subject here."""
    provider = registry.get("deepseek")
    assert provider is not None
    monkeypatch.setattr(
        provider, "complete_request",
        lambda *a, **k: {"content": "ok", "thinking": "", "usage_total": 1})


def test_all_four_are_reported_in_unsupported_params():
    prep = _prepare(_body(**PARAMS), _settings())

    for name, value in PARAMS.items():
        assert prep["unsupported_params"].get(name) == value, name


def test_absent_params_are_not_reported():
    assert _prepare(_body(), _settings())["unsupported_params"] == {}


def test_each_is_named_in_the_single_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="zourcewell.server"):
        _prepare(_body(**PARAMS), _settings())

    lines = [r for r in caplog.records
             if "ignoring unsupported OpenAI params" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    message = lines[0].getMessage()
    for name in PARAMS:
        assert name in message, f"{name} missing from {message!r}"


def test_the_route_accepts_them_off_the_same_single_warning(monkeypatch, caplog):
    """End to end: a body carrying all four is served, warned about once."""
    _stub_engine(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="zourcewell.server"):
        resp = TestClient(create_app(_settings_with_account())).post(
            "/v1/chat/completions", json=_body(**PARAMS))

    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    lines = [r for r in caplog.records
             if "ignoring unsupported OpenAI params" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    assert all(name in lines[0].getMessage() for name in PARAMS)


def test_a_clean_body_warns_about_nothing(monkeypatch, caplog):
    _stub_engine(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="zourcewell.server"):
        resp = TestClient(create_app(_settings_with_account())).post(
            "/v1/chat/completions", json=_body())

    assert resp.status_code == 200, resp.text
    assert "ignoring unsupported OpenAI params" not in caplog.text


def test_stream_options_is_not_swept_into_the_dropped_bucket():
    """Control for the widened tuple: stream_options IS honored locally
    (include_usage), so it must not appear in unsupported_params."""
    prep = _extract_passthrough(_body(stream_options={"include_usage": True}))

    assert prep["stream_options"] == {"include_usage": True}
    assert "stream_options" not in prep["unsupported"]


@pytest.mark.parametrize("value", [False, 0, 0.0, "", []])
def test_falsy_but_present_is_still_reported(value):
    """House style, same as parallel_tool_calls: present means reported."""
    prep = _prepare(_body(seed=value), _settings())

    assert "seed" in prep["unsupported_params"]
