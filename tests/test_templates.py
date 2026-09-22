"""Phase 6: GLM/Qwen/Kimi templates + shared core.openai shim (no network)."""

from __future__ import annotations

import pytest

from deepseaport.core.provider import Provider


def _check_template(module_name: str, cls_name: str, pid: str):
    import importlib

    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    assert issubclass(cls, Provider)
    inst = cls()
    assert inst.id == pid
    assert inst.models() == {}
    assert inst.default_model() == ""
    assert isinstance(mod.create_provider(), cls)
    assert inst.normalize_token('{"value":"tok64"}') == "tok64"
    assert inst.probe_ban({}, timeout=5) == (False, None)
    with pytest.raises(NotImplementedError):
        inst.login("a@x.com", "pw", None)
    with pytest.raises(NotImplementedError):
        inst.complete(None, None, {}, None)
    # warmup is a no-op returning None.
    assert inst.warmup(None) is None


def test_glm_template():
    _check_template("deepseaport.providers.glm", "GlmProvider", "glm")


def test_qwen_provider_registered():
    from deepseaport.core import registry

    prov = registry.get("qwen")
    assert prov is not None
    assert prov.id == "qwen"
    assert set(prov.models()) == {"qwen3.7-plus", "qwen3.8-max", "qwen3.8-omni-flash"}
    assert prov.default_model() == "qwen3.7-plus"


def test_kimi_template():
    _check_template("deepseaport.providers.kimi", "KimiProvider", "kimi")


def test_templates_not_auto_registered():
    from deepseaport.core import registry

    reg = registry.registered()
    assert "glm" not in reg
    assert "kimi" not in reg


def test_openai_shim_identity_through_reexports():
    from deepseaport.core import openai as O
    import deepseaport.completion_service as C
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
