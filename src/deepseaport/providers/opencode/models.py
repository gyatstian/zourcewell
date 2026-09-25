"""Opencode model catalogue as first-class ModelSpec entries.

All eight ``opencode/*`` free models below were verified live against a
local ``opencode serve`` instance (session -> message -> ``finish: stop``
with real text, no auth). They match the Zen catalog
(``https://opencode.ai/zen/v1/models``). All are flagged thinking-capable:
``reasoning`` parts were observed by default (muse-spark live: 203
reasoning tokens) and the rest are adapter-attested thinking models, so
``reasoning_effort`` is accepted as the serve message ``variant`` for all.
``provider.model_spec``
additionally accepts any catalogue-unknown ``provider/model`` id as a
passthrough (user-configured proxy providers on the same serve instance),
so ``/v1/chat/completions`` works with those too; ``/v1/models`` serves
this static verified set.
"""

from __future__ import annotations

from ...core.provider import ModelSpec

MODELS_SPEC: dict[str, ModelSpec] = {
    "opencode/muse-spark-1.3-contributor-free": ModelSpec(
        id="opencode/muse-spark-1.3-contributor-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/muse-spark-1.2-contributor-free": ModelSpec(
        id="opencode/muse-spark-1.2-contributor-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/ling-3.0-flash-fin-free": ModelSpec(
        id="opencode/ling-3.0-flash-fin-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/space-bunny-free": ModelSpec(
        id="opencode/space-bunny-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/mimo-v2.6-flash-free": ModelSpec(
        id="opencode/mimo-v2.6-flash-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/nemotron-3-ultra-free": ModelSpec(
        id="opencode/nemotron-3-ultra-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/nemotron-3.5-lightning-free": ModelSpec(
        id="opencode/nemotron-3.5-lightning-free", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
    "opencode/big-pickle": ModelSpec(
        id="opencode/big-pickle", thinking=True, search=False,
        model_type="default", owned_by="opencode"),
}

DEFAULT_MODEL = "opencode/muse-spark-1.3-contributor-free"
