"""Qwen provider template (import-safe: no network, no side effects)."""

from __future__ import annotations

import logging

from ...core.provider import Provider

logger = logging.getLogger("deepseaport.providers")


class QwenProvider(Provider):
    """Qwen web2api provider (template: implement protocol/client first)."""

    id = "qwen"
    display_name = "Qwen"

    def models(self) -> dict:
        """Return this provider's model catalogue (TODO: fill models.py)."""
        return {}

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Return (is_banned, mute_until) (TODO: reverse users/current probe)."""
        return False, None

    def login(self, email: str, password: str, settings) -> object:
        """Authenticate via the Qwen web login (not reversed yet)."""
        raise NotImplementedError("Qwen web login not implemented: reverse the web login first, see README")

    def normalize_token(self, raw) -> str:
        """Normalize a pasted token via the shared extractor."""
        from ...core.tokens import extract_token
        return extract_token(raw)

    def warmup(self, bridge) -> None:
        """No-op (override only if the provider needs WAF/PoW warmup like DeepSeek)."""
        return None

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one blocking completion (not implemented yet)."""
        raise NotImplementedError(
            "Qwen completion not implemented: override complete_request / "
            "stream_response reusing core.tools, core.transport, "
            "core.openai._openai_response and core.pools.make_pool "
            "(deepseaport.accounts pool)")

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion: override with the Qwen engine (see complete)."""
        raise NotImplementedError(
            "Qwen completion not implemented: override complete_request / "
            "stream_response reusing core.tools, core.transport, "
            "core.openai._openai_response and core.pools.make_pool "
            "(deepseaport.accounts pool)")

    def stream_response(self, app, pool, item, prep):
        """Async-gen streaming: override with the Qwen SSE mapping."""
        raise NotImplementedError(
            "Qwen streaming not implemented: override stream_response reusing "
            "core.transport SSE helpers and core.openai._openai_response")


def create_provider() -> QwenProvider:
    """Factory used by the registry to instantiate the Qwen provider."""
    return QwenProvider()
