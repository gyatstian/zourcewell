"""Kimi provider package: factory + provider class (no registration side effects)."""

from __future__ import annotations

from .provider import KimiProvider, create_provider

__all__ = ["KimiProvider", "create_provider"]
