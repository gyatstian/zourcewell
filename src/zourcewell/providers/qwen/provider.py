"""Qwen provider implementation.

Every method delegates to the local Qwen engine (``.completion_service``),
which is the canonical monkeypatch surface for tests; the provider does not
import ``zourcewell.server`` (no cycle).
"""

from __future__ import annotations

import logging

from ...core.provider import ModelSpec, Provider
from .models import MODELS_SPEC

logger = logging.getLogger("zourcewell.providers")


class QwenProvider(Provider):
    """Qwen web2api provider (models, ban probe, login, WAF warmup).

    Execution uses the local Qwen engine (``.completion_service``), the
    canonical monkeypatch surface; the server module is never imported.
    Login is direct-API POST /api/v1/auths/signin (no browser, no captcha);
    tokens are JWTs that ride the ``token`` cookie on every hot path.
    """

    id = "qwen"
    display_name = "Qwen"

    # Consumes its own acw_tc cache (prime/fetch on the qwen host), never the
    # bridge's jar: the shared bridge jar is warmed against the DeepSeek host.
    needs_waf_jar = False

    def models(self) -> dict[str, ModelSpec]:
        """Return a copy of the Qwen model catalogue."""
        return dict(MODELS_SPEC)

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Delegate to the Qwen client's auths/ probe (best-effort, never raises)."""
        try:
            from ...providers.qwen import client as QC
            return QC.check_ban(headers, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.debug("qwen probe_ban failed: %s", exc)
            return False, None

    def login(self, email: str, password: str, settings) -> object:
        """Direct-API signin; returns (token, banned, detail) tuple, never raises.

        Delegates to ``.auth.login`` (POST /api/v1/auths/signin with a
        SHA-256-hex password). The token is a JWT for AccountConfig.token;
        it authenticates as the ``token`` cookie (Bearer alone is WAF-gated).
        """
        from . import auth
        return auth.login(email, password, settings)

    def warmup(self, bridge) -> None:
        """Best-effort qwen edge priming: solved-jar + open GET models/.

        Solved-jar fetch to chat.qwen.ai (existing / only) via obscura —
        never the DeepSeek host — plants full acw_tc 64-char + x-ap/sca/cna/
        tfstk/isg, then GET models/ (verified open, NOT WAF-gated) harvests
        bare fallback. WinError5 graceful, names-only, never raises.
        """
        if bridge is None:
            return None
        try:
            from . import client as QC
            from . import protocol as P
            try:
                QC.prime_qwen_bridge(bridge)
            except Exception:
                pass
            # UA only: the bridge's jar is the DeepSeek-host WAF jar, so it
            # must not ride a request to chat.qwen.ai. Qwen cookies are the
            # local cache harvested from the qwen host by the prime above.
            try:
                ua = bridge.state.user_agent
            except Exception:
                ua = ""
            cookies = ""
            try:
                try:
                    cookies = QC._merged_waf_cookies("")
                except Exception:
                    pass
                QC.fetch_models(
                    P.base_headers(user_agent=ua, waf_cookies=cookies,
                                   bearer=""),
                    timeout=20)
            except OSError as exc:
                if QC._is_winerror5(exc):
                    logger.warning("qwen warmup WinError5, continuing")
        except Exception:
            pass
        return None

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one completion via the local Qwen failover engine."""
        from .completion_service import _complete_with_failover_sync
        return _complete_with_failover_sync(
            app, pool, prep, first_item,
            cancel_event=cancel_event, on_content=on_content, on_thinking=on_thinking)

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion via the local Qwen failover engine."""
        from .completion_service import _complete_with_failover_sync
        return _complete_with_failover_sync(app, pool, prep, item, cancel_event)

    def stream_response(self, app, pool, item, prep):
        """Async-gen streaming via the local Qwen streaming engine."""
        from .completion_service import _stream_completion
        return _stream_completion(app, pool, item, prep)


def create_provider() -> QwenProvider:
    """Factory used by the registry to instantiate the Qwen provider."""
    return QwenProvider()
