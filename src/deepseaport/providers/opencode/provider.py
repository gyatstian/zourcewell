"""Opencode provider implementation (local CLI via ``opencode serve``).

Unlike the web2api providers this one needs no accounts, no tokens, no
browser, and no warmup: free models work with zero CLI auth, and anything
needing keys is authenticated inside the CLI itself. Every
method delegates to the local engine (``.completion_service``), the
canonical monkeypatch surface; ``deepseaport.server`` is never imported.
"""

from __future__ import annotations

import logging

from ...core.provider import ModelSpec, Provider
from .models import DEFAULT_MODEL, MODELS_SPEC

logger = logging.getLogger("deepseaport.providers")


class OpencodeProvider(Provider):
    """Local opencode CLI provider (models + no-op auth + serve engine)."""

    id = "opencode"
    display_name = "opencode"

    # No AccountPool slot needed: the route layer skips acquisition for
    # providers with this flag (single completion at a time is NOT enforced;
    # the serve instance multiplexes sessions itself).
    requires_accounts = False

    # No WAF bridge either: localhost serve needs no Obscura warmup.
    needs_bridge = False

    def models(self) -> dict[str, ModelSpec]:
        """Return a copy of the static opencode catalogue (see models.py)."""
        return dict(MODELS_SPEC)

    def default_model(self) -> str:
        """Return the canonical default opencode model id."""
        return DEFAULT_MODEL

    def model_spec(self, model: str) -> ModelSpec | None:
        """Exact match first, else passthrough for ``provider/model`` ids.

        The local serve instance can address any provider/model the CLI is
        authenticated for, so catalogue-unknown ids containing a slash are
        accepted as non-thinking/search passthrough specs instead of 404ing.
        Bare ids outside the catalogue stay unknown (likely typos — use
        the ``provider/model`` form).
        """
        key = str(model or "").strip()
        if not key:
            return None
        for mid, spec in MODELS_SPEC.items():
            if str(mid).strip().lower() == key.lower():
                return spec
        if "/" not in key:
            return None
        # Passthrough: split at the first slash downstream (see
        # protocol.split_model; the model part may contain slashes).
        return ModelSpec(id=key, thinking=False, search=False,
                         model_type="default", owned_by="opencode")

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Local serve has no bans; always (False, None), never raises."""
        return False, None

    def login(self, email: str, password: str, settings) -> object:
        """Not applicable: free models need no login; key-gated providers
        authenticate inside the opencode CLI itself, never here."""
        raise NotImplementedError(
            "opencode has no web login: free models need nothing, "
            "key-gated providers use `opencode auth login` in the CLI")

    def normalize_token(self, raw) -> str:
        """Normalize a pasted token via the shared extractor (unused)."""
        from ...core.tokens import extract_token
        return extract_token(raw)

    def warmup(self, bridge) -> None:
        """Boot the managed isolated serve when nothing answers (never raises).

        The managed instance carries deepseaport's minimal agent + strip
        plugin via child-only env; the user's own opencode config is never
        touched. An explicit DEEPSEAPORT_OPENCODE_BASE opts out of this and
        uses the user's serve as-is.
        """
        try:
            from . import client as OC
            from . import managed_serve as MS
            if OC.health(timeout=10):
                return None
            base = MS.ensure()
            if base and OC.health(timeout=10):
                logger.info("opencode managed serve ready at %s", base)
            else:
                logger.warning(
                    "opencode serve not reachable (managed boot failed; "
                    "first request retries, or set DEEPSEAPORT_OPENCODE_BASE)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("opencode warmup ignored: %s", exc)
        return None

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one completion through the local serve engine."""
        from .completion_service import _complete_with_failover_sync
        return _complete_with_failover_sync(
            app, pool, prep, first_item,
            cancel_event=cancel_event, on_content=on_content, on_thinking=on_thinking)

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion via the local serve engine."""
        from .completion_service import _complete_with_failover_sync
        return _complete_with_failover_sync(app, pool, prep, item, cancel_event)

    def stream_response(self, app, pool, item, prep):
        """Async-gen (buffered) streaming via the local serve engine."""
        from .completion_service import _stream_completion
        return _stream_completion(app, pool, item, prep)


def create_provider() -> OpencodeProvider:
    """Factory used by the registry to instantiate the opencode provider."""
    return OpencodeProvider()
