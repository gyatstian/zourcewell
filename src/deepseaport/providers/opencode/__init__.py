"""Opencode provider package: factory + provider class (no registration side effects)."""

from .provider import OpencodeProvider, create_provider

__all__ = ["OpencodeProvider", "create_provider"]
