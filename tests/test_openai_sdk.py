"""The real OpenAI SDK against this app, over an in-process ASGI transport.

"OpenAI-compatible" is a claim about what SDK clients can actually do with the
API, so this file drives the SDK instead of asserting our own JSON shapes.
Hermetic: ``ASGITransport`` runs the app in the same process (no socket), the
settings carry ``keys=[]`` (auth off) and a registered fake provider owns the
completion seam. Not ``live`` -- nothing here touches the network.

Verified against openai 3.33, whose own hard dependency is ``httpx2``; its
``ASGITransport`` is async-only, hence ``AsyncOpenAI`` + ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from support import settings
from zourcewell.accounts import AccountConfig, AccountPool
from zourcewell.core import registry
from zourcewell.core.provider import ModelSpec, Provider
from zourcewell.server import create_app

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

_CHUNK = (b'data: {"id": "chatcmpl-sdk", "object": "chat.completion.chunk",'
          b' "created": 1, "model": "sdkp-echo", "choices": [{"index": 0,'
          b' "delta": %s, "finish_reason": %s}]}\n\n')


class _SdkProvider(Provider):
    """A registered fake upstream: no sockets, no accounts to rotate.

    ``complete_request``/``stream_response`` own the account slot's release
    (the real providers do; the route's acquire would otherwise hold it -- see
    tests/test_auth_hardening.py::test_ordinary_request_and_stream_survive_the_middleware).
    """

    id = "sdkp"
    display_name = "SDK fake"

    def models(self) -> dict[str, ModelSpec]:
        return {"sdkp-echo": ModelSpec(id="sdkp-echo", owned_by="sdkp")}

    def complete_request(self, app, pool, prep, item, cancel_event):
        AccountPool.release(item)
        return {"content": "pong", "thinking": "", "usage_total": 4}

    async def stream_response(self, app, pool, item, prep):
        try:
            yield _CHUNK % (b'{"role": "assistant", "content": "po"}', b"null")
            yield _CHUNK % (b'{"content": "ng"}', b'"stop"')
            yield b"data: [DONE]\n\n"
        finally:
            AccountPool.release(item)


@pytest.fixture
def app():
    fake = _SdkProvider()
    registry._REGISTRY[registry._key(fake)] = fake
    try:
        yield create_app(settings(
            keys=[], config_path="", log_level="WARNING",
            providers=["deepseek", "sdkp"],
            accounts=[AccountConfig(email="sdk@x.com", token="t", provider="sdkp")]))
    finally:
        registry._REGISTRY.pop("sdkp", None)


def _run(app, scenario):
    """One async scenario against an SDK client bound to ``app`` in-process."""

    async def _main():
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
                                  base_url="http://test")
        sdk = openai.AsyncOpenAI(base_url="http://test/v1", api_key="test-key",
                                 http_client=http)
        try:
            return await scenario(sdk)
        finally:
            await http.aclose()

    return asyncio.run(_main())


def test_models_list_is_typed(app):
    listed = _run(app, lambda sdk: sdk.models.list())

    assert listed.object == "list"
    assert all(isinstance(m, openai.types.Model) for m in listed.data)
    assert "sdkp-echo" in [m.id for m in listed.data]


def test_chat_completion_is_a_typed_object(app):
    completion = _run(app, lambda sdk: sdk.chat.completions.create(
        model="sdkp-echo", messages=[{"role": "user", "content": "hi"}]))

    assert isinstance(completion, openai.types.chat.ChatCompletion)
    assert completion.choices[0].message.content == "pong"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None and completion.usage.total_tokens > 0


def test_stream_iterates_typed_chunks(app):
    async def scenario(sdk):
        stream = await sdk.chat.completions.create(
            model="sdkp-echo", stream=True,
            messages=[{"role": "user", "content": "hi"}])
        pieces, kinds = [], set()
        async for chunk in stream:
            kinds.add(type(chunk))
            for choice in chunk.choices:
                if choice.delta.content:
                    pieces.append(choice.delta.content)
        return pieces, kinds

    pieces, kinds = _run(app, scenario)

    assert pieces == ["po", "ng"], "SSE frames must reach the SDK as deltas"
    assert kinds == {openai.types.chat.ChatCompletionChunk}


def test_tool_calls_reply_is_typed_and_parsed(app, monkeypatch):
    calls = [{"id": "call_sdk", "type": "function",
              "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}]
    monkeypatch.setattr("zourcewell.server.parse_tool_calls",
                        lambda content, tools, **kw: (calls, ""))

    completion = _run(app, lambda sdk: sdk.chat.completions.create(
        model="sdkp-echo",
        messages=[{"role": "user", "content": "time?"}],
        tools=[{"type": "function", "function": {
            "name": "get_time", "description": "d",
            "parameters": {"type": "object", "properties": {}}}}]))

    assert completion.choices[0].finish_reason == "tool_calls"
    tool_call = completion.choices[0].message.tool_calls[0]
    assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageToolCall)
    assert tool_call.id == "call_sdk"
    assert tool_call.function.name == "get_time"
    assert json.loads(tool_call.function.arguments) == {"tz": "UTC"}


def test_model_404_is_a_typed_error_with_an_error_envelope(app):
    """SDK clients classify failures from the response body: the SDK raises
    ``NotFoundError`` for a 404 either way, but with only ``{"detail": ...}``
    its ``error.type`` stays unset. ``server_routes.install_error_envelope``
    adds the envelope; this asserts the shape an SDK client actually reads."""
    with pytest.raises(openai.NotFoundError) as ei:
        _run(app, lambda sdk: sdk.chat.completions.create(
            model="gpt-99", messages=[{"role": "user", "content": "hi"}]))

    exc = ei.value
    assert exc.response.status_code == 404
    body = exc.response.json()
    assert body["error"]["type"], f"no error.type in the 404 body: {body!r}"
