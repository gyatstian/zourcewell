"""First-class provider abstraction: model catalogue and account/auth seams.

A provider describes one upstream (DeepSeek today) as data plus delegating
methods. The completion engine is intentionally left in ``deepseaport.server``
for this phase; provider methods late-import it so monkeypatch seams survive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one model exposed by a provider."""

    id: str
    thinking: bool = False
    search: bool = False
    model_type: str = "default"
    owned_by: str = ""
    capabilities: dict = field(default_factory=dict)


class Provider(ABC):
    """Base class for an upstream provider (models + account/completion seams)."""

    id: str = ""
    display_name: str = ""

    @abstractmethod
    def models(self) -> dict[str, ModelSpec]:
        """Return this provider's model catalogue keyed by model id."""
        raise NotImplementedError

    def default_model(self) -> str:
        """Return the default model id (first catalogue key, else empty)."""
        models = self.models()
        for mid in models:
            return mid
        return ""

    def model_spec(self, model: str) -> ModelSpec | None:
        """Case-insensitive lookup of a model spec; None when unknown."""
        key = str(model or "").strip().lower()
        if not key:
            return None
        for mid, spec in self.models().items():
            if str(mid).strip().lower() == key:
                return spec
        return None

    def accepts_model(self, model: str) -> bool:
        """True when the model is present in this provider's catalogue."""
        return self.model_spec(model) is not None

    def probe_ban(self, headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
        """Return (is_banned, mute_until) for the given auth headers."""
        raise NotImplementedError

    def login(self, email: str, password: str, settings) -> object:
        """Authenticate and return a LoginResult-like object for an account."""
        raise NotImplementedError

    def normalize_token(self, raw) -> str:
        """Return a usable raw token from any pasted representation."""
        raise NotImplementedError

    def warmup(self, bridge) -> None:
        """Best-effort provider-specific startup warmup (default no-op)."""
        return None

    def complete(self, app, pool, prep, first_item, cancel_event=None,
                 on_content=None, on_thinking=None) -> dict:
        """Run one blocking completion through the existing engine (delegates)."""
        raise NotImplementedError

    async def stream(self, app, pool, item, prep):
        """Async generator of completion deltas (providers may not stream)."""
        raise NotImplementedError

    def complete_request(self, app, pool, prep, item, cancel_event):
        """Blocking completion. Default: the shared server engine (deepseek behavior). Override per provider."""
        from .. import server as _server
        return _server._complete_with_failover_sync(app, pool, prep, item, cancel_event)

    def stream_response(self, app, pool, item, prep):
        """Async-gen streaming. Default: the shared server streaming. Override per provider."""
        from .. import server as _server
        return _server._stream_completion(app, pool, item, prep)

    def tuple_spec(self) -> dict[str, tuple[bool, bool, str]]:
        """Return ``{id: (thinking, search, model_type)}`` for compat MODELS."""
        return {mid: (s.thinking, s.search, s.model_type)
                for mid, s in self.models().items()}