"""Qwen model catalogue as first-class ModelSpec entries.

Capabilities come from the live verified catalogue (GET /api/v2/models/):
thinking/search flags below mirror ``info.meta.capabilities``.
"""

from __future__ import annotations

from ...core.provider import ModelSpec

MODELS_SPEC: dict[str, ModelSpec] = {
    "qwen3.7-plus": ModelSpec(
        id="qwen3.7-plus", thinking=True, search=True,
        model_type="default", owned_by="qwen"),
    "qwen3.8-max": ModelSpec(
        id="qwen3.8-max", thinking=True, search=True,
        model_type="default", owned_by="qwen"),
    "qwen3.8-omni-flash": ModelSpec(
        id="qwen3.8-omni-flash", thinking=True, search=False,
        model_type="default", owned_by="qwen"),
}

DEFAULT_MODEL = "qwen3.7-plus"