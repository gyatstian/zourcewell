"""Shared structural types used across transport and provider modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StreamEvent:
    kind: Literal["thinking", "content", "finished", "usage", "error"]
    text: str = ""
    code: str = ""
    message: str = ""
    # Ban expiry (mute_until unix ts) when the error carries a ban payload.
    # Lets callers cool the account down without an extra users/current HTTP.
    ban_until: float | None = None