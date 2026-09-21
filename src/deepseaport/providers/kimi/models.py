"""Kimi model catalogue (template: fill in once the web API is reversed)."""

from __future__ import annotations

# PLACEHOLDERS to verify, not facts — confirm against the live web app before use:
#   kimi-k2, kimi-k2-0711, moonshot-v1-8k
# Fill MODELS_SPEC with core.provider.ModelSpec entries, e.g.:
#   MODELS_SPEC = {
#       "kimi-k2": ModelSpec(id="kimi-k2", thinking=False, search=False,
#                            model_type="default", owned_by="kimi"),
#   }
# ModelSpec fields: id, thinking, search, model_type, owned_by, capabilities.

MODELS_SPEC: dict = {}

DEFAULT_MODEL = ""
