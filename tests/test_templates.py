"""Registered provider gate (qwen) + shared core.openai shim.

Offline, no network.
"""

from __future__ import annotations


def test_qwen_provider_registered():
    from deepseaport.core import registry

    prov = registry.get("qwen")
    assert prov is not None
    assert prov.id == "qwen"
    assert set(prov.models()) == {"qwen3.7-plus", "qwen3.8-max", "qwen3.8-omni-flash"}
    assert prov.default_model() == "qwen3.7-plus"


def test_no_template_providers_remain_registered():
    """No unimplemented template provider may be registered.

    The kimi stub was removed; this guards against a half-added provider
    re-appearing in the registry without an implementation.
    """
    from deepseaport.core import registry

    reg = registry.registered()
    assert "kimi" not in reg
    assert "deepseek" in reg


def test_openai_shim_identity_through_reexports():
    from deepseaport.core import openai as O
    import deepseaport.providers.deepseek.completion_service as C
    import deepseaport.server as S

    assert O._openai_response is S._openai_response
    assert C._openai_response is S._openai_response


def test_openai_shim_honors_patched_server_parser(monkeypatch):
    import deepseaport.server as S

    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "get_time", "arguments": "{}"}}]
    monkeypatch.setattr(S, "parse_tool_calls",
                        lambda content, tools, **kw: (calls, ""))
    prep = {"model": "deepseek-flash", "prompt": "hello world test prompt",
            "tools": [{"function": {"name": "get_time"}}]}
    out = S._openai_response(prep, {"content": "{}", "thinking": "",
                                    "usage_total": 3})
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"] == calls
    assert out["choices"][0]["finish_reason"] == "tool_calls"
