"""Provider abstraction + registry seam tests (no network, no credentials)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from deepseaport.core import registry
from deepseaport.core.provider import ModelSpec, Provider
from deepseaport.providers.deepseek.models import DEFAULT_MODEL, MODELS_SPEC
from deepseaport.providers.deepseek.provider import DeepSeekProvider
from deepseaport.server import MODELS
from deepseaport.server_routes import _prepare


def _body(model: str = "deepseek-flash") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_registry_default_and_get():
    assert registry.default().id == "deepseek"
    assert registry.get("deepseek") is not None
    assert registry.get("nope") is None
    assert registry.get("DEEPSEEK") is not None


def test_deepseek_models_match_compat_table():
    provider = DeepSeekProvider()
    assert set(provider.models()) == set(MODELS)
    assert provider.default_model() == DEFAULT_MODEL
    assert isinstance(provider.models()["deepseek-flash"], ModelSpec)


def test_model_spec_case_insensitive():
    provider = DeepSeekProvider()
    assert provider.model_spec("DEEPSEEK-FLASH") is not None
    assert provider.model_spec("deepseek-flash").thinking is False
    assert provider.accepts_model("deepseek-flash-reasoner")
    assert not provider.accepts_model("gpt-99")
    assert provider.model_spec("gpt-99") is None


def test_server_models_exact():
    assert MODELS == {
        "deepseek-flash": (False, False, "default"),
        "deepseek-flash-reasoner": (True, False, "default"),
        "deepseek-flash-search": (False, True, "default"),
        "deepseek-flash-reasoner-search": (True, True, "default"),
    }
    assert len(MODELS) == 4


def test_tuple_spec_matches_models():
    assert DeepSeekProvider().tuple_spec() == MODELS


@pytest.mark.parametrize("model,thinking,search", [
    ("deepseek-flash", False, False),
    ("deepseek-flash-reasoner", True, False),
    ("deepseek-flash-search", False, True),
    ("deepseek-flash-reasoner-search", True, True),
])
def test_prepare_provider_and_flags(model, thinking, search):
    prep = _prepare(_body(model))
    assert prep["provider"] == "deepseek"
    assert prep["thinking"] is thinking
    assert prep["search"] is search
    assert prep["model_type"] == "default"


def test_resolve_model_reasoner_thinking_true():
    resolved = registry.resolve_model("deepseek-flash-reasoner")
    assert resolved is not None
    provider, spec = resolved
    assert provider.id == "deepseek"
    assert spec.thinking is True
    assert spec.search is False


def test_resolve_model_unknown_none():
    assert registry.resolve_model("gpt-99") is None


def test_prepare_unknown_model_404():
    with pytest.raises(HTTPException) as exc:
        _prepare(_body("gpt-99"))
    assert exc.value.status_code == 404


def test_registry_models_load_registered_owned_by():
    provider = registry.get("deepseek")
    assert provider is not None
    assert all(s.owned_by == "deepseek" for s in provider.models().values())


def test_models_spec_defaults():
    spec = MODELS_SPEC["deepseek-flash-reasoner"]
    assert spec.id == "deepseek-flash-reasoner"
    assert spec.thinking is True
    assert spec.capabilities == {}