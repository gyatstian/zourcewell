"""Direct-API Qwen authentication (no browser, no captcha).

Live-verified 2026-09-22: POST /api/v1/auths/signin with a SHA-256-hex
password returns a flat user object with a JWT ``token`` + ``token_type``
``"Bearer"`` + ``expires_at``. The same JWT is planted as a ``token`` cookie
on ``.qwen.ai`` (the web session key). Self-contained: mirrors the DeepSeek
``LoginResult``/``unpack_login_result`` contract without importing deepseek.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import client as QC
from . import protocol as P

logger = logging.getLogger("deepseaport.auth")

__all__ = [
    "LoginResult",
    "login",
    "unpack_login_result",
]


@dataclass(frozen=True)
class LoginResult:
    token: str | None = None
    banned: bool = False
    detail: str = ""

    def as_tuple(self) -> tuple[str | None, bool, str]:
        return self.token, self.banned, self.detail


def unpack_login_result(res) -> tuple[str | None, bool, str]:
    """Normalize a LoginResult / legacy tuple / legacy string into a tuple."""
    if res is None:
        return None, False, ""
    if isinstance(res, LoginResult):
        return res.as_tuple()
    if isinstance(res, tuple):
        token = res[0] if len(res) > 0 else None
        banned = bool(res[1]) if len(res) > 1 else False
        detail = str(res[2] or "") if len(res) > 2 else ""
        token = token.strip() if isinstance(token, str) else None
        return (token or None), banned, detail
    if isinstance(res, str):
        return (res.strip() or None), False, ""
    return None, False, ""


def login(email: str, password: str, settings=None) -> tuple[str | None, bool, str]:
    """Sign in via the direct API; returns (token, banned, detail). Never raises.

    - missing credentials -> (None, False, "email and password are required").
    - signin success -> (token, False, ""). The JWT is the value to store in
      AccountConfig.token; it also authenticates as the ``token`` cookie.
    - WAF challenge (RuntimeError text contains WAF/challenge/captcha) -> a
      challenge note, not a ban.
    - other RuntimeError -> banned=True only when the message carries generic
      mute/suspend/ban wording (best-effort; ban envelope unverified).
    - unexpected exceptions -> (None, False, str(exc)).
    """
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        return None, False, "email and password are required"
    try:
        user = QC.signin(email, password)
    except RuntimeError as exc:
        msg = str(exc)
        low = msg.lower()
        if "waf" in low or "challenge" in low or "captcha" in low:
            return None, False, msg
        banned = bool(P.is_ban_error_text(msg))
        return None, banned, msg
    except Exception as exc:  # noqa: BLE001
        logger.warning("qwen signin unexpected error for %s: %s", email, exc)
        return None, False, str(exc)
    token = str(user.get("token") or "").strip() or None
    if token is None:
        return None, False, "signin returned no token"
    return token, False, ""