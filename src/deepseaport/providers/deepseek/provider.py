"""DeepSeek provider implementation.

Every method delegates to the local DeepSeek engine
(``.completion_service``). A tiny ``sys.modules`` check honors historical
``deepseaport.server`` monkeypatch seams without importing the server
module (no provider->server->engine cycle).
"""

from __future__ import annotations

import logging

from ...core.provider import ModelSpec, Provider
from .models import DEFAULT_MODEL, MODELS_SPEC

logger = logging.getLogger("deepseaport.providers")


class DeepSeekProvider(Provider):
    """DeepSeek web2api provider (models, ban probe, login, PoW warmup).

    Execution uses the local DeepSeek engine (``.completion_service``);
    a ``sys.modules`` lookup honors ``deepseaport.server`` monkeypatches
    without importing the server module.
    """

    id = "deepseek"
    display_name = "DeepSeek"

    def models(self) -> dict[str, ModelSpec]:
        """Return a copy of the DeepSeek model catalogue."""
        return dict(MODELS_SPEC)

    def default_model(self) -> str:
        """Return the canonical DeepSeek default model id."""
        return DEFAULT_MODEL

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Delegate to the DeepSeek client's users/current ban probe."""
        try:
            from ...providers.deepseek import client as DS
            return DS.check_ban(headers, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.debug("deepseek probe_ban failed: %s", exc)
            return False, None

    def login(self, email: str, password: str, settings) -> object:
        """Delegate to the DeepSeek browser login flow."""
        from ...providers.deepseek import auth
        return auth.browser_login(email, password, settings)

    def normalize_token(self, raw) -> str:
        """Normalize a pasted token via the shared extractor."""
        from ...core.tokens import extract_token
        return extract_token(raw)

    def warmup(self, bridge) -> None:
        """Fetch/prewarm the DeepSeek PoW wasm; best-effort, never raises."""
        try:
            cookies = bridge.cookie_header()
            user_agent = bridge.state.user_agent
        except Exception as exc:  # noqa: BLE001
            logger.warning("startup WAF warmup failed: %s", exc)
            return
        try:
            from ...providers.deepseek import pow as PoW
            PoW.ensure_wasm(cookies=cookies, user_agent=user_agent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PoW wasm fetch failed (will retry on demand): %s", exc)
            return
        try:
            from ...providers.deepseek import pow as PoW
            PoW.prewarm()
        except Exception as exc:  # noqa: BLE001
            logger.debug("PoW prewarm failed (lazy compile on demand): %s", exc)

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one completion via the local DeepSeek failover engine."""
        from .completion_service import (
            _complete_with_failover_sync as _local,
            _server_override as _ov,
        )
        fn = _ov("_complete_with_failover_sync", _local)
        return fn(
            app, pool, prep, first_item,
            cancel_event=cancel_event, on_content=on_content, on_thinking=on_thinking)

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion via the local DeepSeek failover engine."""
        from .completion_service import (
            _complete_with_failover_sync as _local,
            _server_override as _ov,
        )
        fn = _ov("_complete_with_failover_sync", _local)
        return fn(app, pool, prep, item, cancel_event)

    def stream_response(self, app, pool, item, prep):
        """Async-gen streaming via the local DeepSeek streaming engine."""
        from .completion_service import (
            _stream_completion as _local_stream,
            _server_override as _ov,
        )
        fn = _ov("_stream_completion", _local_stream)
        return fn(app, pool, item, prep)


def create_provider() -> DeepSeekProvider:
    """Factory used by the registry to instantiate the DeepSeek provider."""
    return DeepSeekProvider()