"""First-class provider abstraction: model catalogue and account/auth seams.

A provider describes one upstream (``deepseek``, ``qwen``, ``opencode``) as data
plus delegating methods. The shared completion mechanics live in
``core.engine``; this module has no default engine and never references
``zourcewell.server``. ``complete``, ``complete_request`` and
``stream_response`` all raise ``NotImplementedError``, so every provider must
override all three and delegate to its own ``completion_service``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one model exposed by a provider."""

    id: str
    thinking: bool = False
    search: bool = False
    model_type: str = "default"
    owned_by: str = ""


class Provider(ABC):
    """Base class for an upstream provider (models + account/completion seams)."""

    id: str = ""
    display_name: str = ""

    # False for local providers that need no AccountPool slot (e.g. a
    # locally running CLI serve instance that multiplexes sessions itself).
    # Routes skip slot acquisition and pass item=None; preflight passes
    # when only such providers are active.
    requires_accounts: bool = True

    # False for providers needing no WAF bridge (no bridge in one-shot chat;
    # the startup warmup is gated on needs_waf_jar below). Local CLI
    # providers set this.
    needs_bridge: bool = True

    # True when this provider consumes the Obscura bridge's harvested WAF
    # cookie jar. The startup warmup targets one host (CHAT_HOME, DeepSeek by
    # default), so a provider whose WAF cookies come from elsewhere (qwen's
    # own acw_tc cache) must not trigger it.
    needs_waf_jar: bool = True

    @abstractmethod
    def models(self) -> dict[str, ModelSpec]:
        """Return this provider's model catalogue keyed by model id."""
        raise NotImplementedError

    def models_hint(self) -> dict[str, ModelSpec]:
        """Catalogue for a menu row: no I/O, nothing started, nothing waited on.

        ``models()`` may be a LIVE read that boots a local process to answer
        (opencode's managed serve), and a menu has no business starting one --
        nor paying a read whose failure costs a socket timeout -- for a list.
        Defaults to ``models()``, an in-process read for every provider that
        does not override it, so a hint may be older than ``models()`` -- that
        is the trade it makes.
        """
        return self.models()

    def model_spec(self, model: str) -> ModelSpec | None:
        """Case-insensitive lookup of a model spec; None when unknown."""
        key = str(model or "").strip().lower()
        if not key:
            return None
        for mid, spec in self.models().items():
            if str(mid).strip().lower() == key:
                return spec
        return None

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Return (is_banned, mute_until) for the given auth headers."""
        raise NotImplementedError

    def login(self, email: str, password: str, settings) -> object:
        """Authenticate and return a LoginResult-like object for an account."""
        raise NotImplementedError

    def warmup(self, bridge) -> None:
        """Best-effort provider-specific startup warmup (default no-op)."""
        return None

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one blocking completion through the existing engine (delegates)."""
        raise NotImplementedError

    # NOTE: there is deliberately no `stream()` here. It used to sit beside
    # `stream_response()` with a near-identical signature and has never been
    # called by anything -- all three shipped providers override
    # `stream_response`, and that is the name the route layer uses. A template
    # author reading the ABC would have had to guess which of the two to
    # implement. If a future provider genuinely needs a second streaming seam,
    # add it then, with a caller.

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion. No default engine: override per provider."""
        raise NotImplementedError(
            f"{type(self).__name__}.complete_request not implemented")

    def stream_response(self, app, pool, item, prep):
        """Async-gen streaming. No default engine: override per provider."""
        raise NotImplementedError(
            f"{type(self).__name__}.stream_response not implemented")

    def tuple_spec(self) -> dict[str, tuple[bool, bool, str]]:
        """Return ``{id: (thinking, search, model_type)}`` for compat MODELS."""
        return {mid: (s.thinking, s.search, s.model_type)
                for mid, s in self.models().items()}