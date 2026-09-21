"""DeepSeek model catalogue as first-class ModelSpec entries."""

from __future__ import annotations

from ...core.provider import ModelSpec

MODELS_SPEC: dict[str, ModelSpec] = {
    "deepseek-flash": ModelSpec(
        id="deepseek-flash", thinking=False, search=False,
        model_type="default", owned_by="deepseek"),
    "deepseek-flash-reasoner": ModelSpec(
        id="deepseek-flash-reasoner", thinking=True, search=False,
        model_type="default", owned_by="deepseek"),
    "deepseek-flash-search": ModelSpec(
        id="deepseek-flash-search", thinking=False, search=True,
        model_type="default", owned_by="deepseek"),
    "deepseek-flash-reasoner-search": ModelSpec(
        id="deepseek-flash-reasoner-search", thinking=True, search=True,
        model_type="default", owned_by="deepseek"),
}

DEFAULT_MODEL = "deepseek-flash"