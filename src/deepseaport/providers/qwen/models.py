"""Qwen model catalogue (template: fill in once the web API is reversed)."""

from __future__ import annotations

# PLACEHOLDERS to verify, not facts — confirm against the live web app before use:
#   qwen3-max, qwen3-235b-a22b, qwen-plus
# Fill MODELS_SPEC with core.provider.ModelSpec entries, e.g.:
#   MODELS_SPEC = {
#       "qwen3-max": ModelSpec(id="qwen3-max", thinking=False, search=False,
#                              model_type="default", owned_by="qwen"),
#   }
# ModelSpec fields: id, thinking, search, model_type, owned_by, capabilities.

MODELS_SPEC: dict = {}

DEFAULT_MODEL = ""
