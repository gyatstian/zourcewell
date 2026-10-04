"""Direct-API Qwen authentication (no browser, no captcha).

Live-verified 2026-09-22: POST /api/v1/auths/signin with a SHA-256-hex
password returns a flat user object with a JWT ``token`` + ``token_type``
``"Bearer"`` + ``expires_at``. The same JWT is planted as a ``token`` cookie
on ``.qwen.ai`` (the web session key). Self-contained: ``login`` and
``login_with_expiry`` return plain tuples, and ``unpack_login_result`` /
``unpack_login_result_ex`` normalize them, without importing deepseek.
"""

from __future__ import annotations

import base64
import json
import logging
import time

from ...core.engine import scrub_upstream_text
from . import client as QC
from . import protocol as P

logger = logging.getLogger("deepseaport.auth")

# Pre-expiry leeway so a token dying mid-stream is refreshed proactively
# (clock skew + chat-create -> completions gap). 5 min.
EXPIRY_LEEWAY_SECONDS = 300.0

__all__ = [
    "EXPIRY_LEEWAY_SECONDS",
    "effective_expiry",
    "is_expired",
    "jwt_exp",
    "login",
    "login_with_expiry",
    "parse_expires_at",
    "token_expired",
    "unpack_login_result",
    "unpack_login_result_ex",
]


def _coerce_ts(value) -> float | None:
    """Best-effort unix-ts coercion: numeric str ok; ms -> s; <=0 -> None."""
    try:
        if value is None or value == "":
            return None
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    # Heuristic: ms timestamps are 1e12+; normal s timestamps are ~1e9-1e11.
    if ts >= 1e12:
        ts = ts / 1000.0
    return ts if ts > 0 else None


def parse_expires_at(user: dict | None) -> float | None:
    """Extract ``expires_at`` from a signin user dict. Never raises."""
    try:
        if not isinstance(user, dict):
            return None
        for key in ("expires_at", "expiresAt", "expire_at", "exp"):
            if key in user:
                ts = _coerce_ts(user.get(key))
                if ts is not None:
                    return ts
        # Some envelopes nest the user object one level down.
        for key in ("data", "user"):
            inner = user.get(key)
            if isinstance(inner, dict):
                ts = parse_expires_at(inner)
                if ts is not None:
                    return ts
    except Exception:
        pass
    return None


def jwt_exp(token: str | None) -> float | None:
    """Best-effort JWT ``exp`` claim decode (no verification). Never raises."""
    try:
        if not token or not isinstance(token, str) or token.count(".") < 2:
            return None
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        if isinstance(payload, dict):
            return _coerce_ts(payload.get("exp"))
    except Exception:
        pass
    return None


def effective_expiry(user: dict | None, token: str | None) -> float | None:
    """Signin ``expires_at`` wins; JWT ``exp`` is the fallback. Never raises."""
    try:
        ts = parse_expires_at(user)
        if ts is not None:
            return ts
        return jwt_exp(token)
    except Exception:
        return None


def is_expired(expires_at: float | None, now: float | None = None,
               leeway: float = EXPIRY_LEEWAY_SECONDS) -> bool:
    """True when a known expiry is past (or within leeway). Unknown -> False."""
    try:
        if not expires_at:
            return False
        ts = float(expires_at)
        if ts <= 0:
            return False
        base = float(now) if now is not None else time.time()
        return ts <= base + max(0.0, float(leeway or 0.0))
    except (TypeError, ValueError):
        return False


def token_expired(cfg, now: float | None = None,
                  leeway: float = EXPIRY_LEEWAY_SECONDS) -> bool:
    """True when an account's persisted expiry (or JWT exp) is past. Never raises."""
    try:
        known = float(getattr(cfg, "token_expires_at", 0.0) or 0.0)
        if known > 0 and is_expired(known, now=now, leeway=leeway):
            return True
        if known > 0:
            return False
        # No persisted expiry (pasted tokens, old configs): JWT exp fallback.
        return is_expired(jwt_exp(getattr(cfg, "token", "") or ""), now=now,
                          leeway=leeway)
    except Exception:
        return False


def unpack_login_result(res) -> tuple[str | None, bool, str]:
    """Normalize a legacy tuple / legacy string into a tuple."""
    if res is None:
        return None, False, ""
    if isinstance(res, tuple):
        # 4-tuple (token, expires_at, banned, detail) from login_with_expiry.
        if len(res) == 4 and isinstance(res[1], (int, float, type(None))):
            token = res[0].strip() if isinstance(res[0], str) else None
            try:
                _ = float(res[1]) if res[1] else None
                return (token or None), bool(res[2]), str(res[3] or "")
            except (TypeError, ValueError):
                pass
        token = res[0] if len(res) > 0 else None
        banned = bool(res[1]) if len(res) > 1 else False
        detail = str(res[2] or "") if len(res) > 2 else ""
        token = token.strip() if isinstance(token, str) else None
        return (token or None), banned, detail
    if isinstance(res, str):
        return (res.strip() or None), False, ""
    return None, False, ""


def unpack_login_result_ex(res) -> tuple[str | None, float | None, bool, str]:
    """Normalize into (token, expires_at, banned, detail). Never raises."""
    try:
        if res is None:
            return None, None, False, ""
        if isinstance(res, tuple):
            if len(res) == 4:
                token = res[0].strip() if isinstance(res[0], str) else None
                exp: float | None = None
                try:
                    exp = float(res[1]) if res[1] else None
                    if exp is not None and exp <= 0:
                        exp = None
                except (TypeError, ValueError):
                    exp = None
                return (token or None), exp, bool(res[2]), str(res[3] or "")
            token, banned, detail = unpack_login_result(res)
            return token, None, banned, detail
        if isinstance(res, str):
            return (res.strip() or None), None, False, ""
    except Exception:
        pass
    return None, None, False, ""


def login_with_expiry(email: str, password: str,
                      settings=None) -> tuple[str | None, float | None, bool, str]:
    """Sign in via the direct API; returns (token, expires_at, banned, detail).

    Never raises. ``expires_at`` is the signin unix ts (JWT ``exp`` fallback,
    None when unknown) for AccountConfig.token_expires_at. See ``login`` for
    the failure contract.
    """
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        return None, None, False, "email and password are required"
    try:
        user = QC.signin(email, password)
    except RuntimeError as exc:
        msg = str(exc)
        low = msg.lower()
        if "waf" in low or "challenge" in low or "captcha" in low:
            return None, None, False, msg
        banned = bool(P.is_ban_error_text(msg))
        return None, None, banned, msg
    except Exception as exc:  # noqa: BLE001
        logger.warning("qwen signin unexpected error for %s: %s",
                       email, scrub_upstream_text(exc, 300))
        return None, None, False, str(exc)
    token = str(user.get("token") or "").strip() or None
    if token is None:
        return None, None, False, "signin returned no token"
    return token, effective_expiry(user, token), False, ""


def login(email: str, password: str, settings=None) -> tuple[str | None, bool, str]:
    """Sign in via the direct API; returns (token, banned, detail). Never raises.

    - missing credentials -> (None, False, "email and password are required").
    - signin success -> (token, False, ""). The JWT is the value to store in
      AccountConfig.token; it also authenticates as the ``token`` cookie.
      The signin ``expires_at`` is captured by ``login_with_expiry`` — this
      3-tuple wrapper discards it for backward compat (use
      ``login_with_expiry`` when you need the expiry for persistence).
    - WAF challenge (RuntimeError text contains WAF/challenge/captcha) -> a
      challenge note, not a ban.
    - other RuntimeError -> banned=True only when the message carries generic
      mute/suspend/ban wording (best-effort; ban envelope unverified).
    - unexpected exceptions -> (None, False, str(exc)).
    """
    token, _exp, banned, detail = login_with_expiry(email, password, settings)
    return token, banned, detail