"""GLM model catalogue (template: fill in once the web API is reversed)."""

from __future__ import annotations

# PLACEHOLDERS to verify, not facts — confirm against the live web app before use:
#   glm-4.5, glm-4.5-air, glm-4.5-flash
# Fill MODELS_SPEC with core.provider.ModelSpec entries, e.g.:
#   MODELS_SPEC = {
#       "glm-4.5": ModelSpec(id="glm-4.5", thinking=False, search=False,
#                            model_type="default", owned_by="glm"),
#   }
# ModelSpec fields: id, thinking, search, model_type, owned_by, capabilities.

MODELS_SPEC: dict = {}

DEFAULT_MODEL = ""
