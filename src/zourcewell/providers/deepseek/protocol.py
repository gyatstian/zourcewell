"""DeepSeek web protocol: headers, payloads, SSE parsing, error mapping."""

from __future__ import annotations

import math
import os

from ...core.transport import has_ban_word as _shared_has_ban_word
from ...core.transport import parse_json_response as _shared_parse_json

# SSE events are built through this module (deepseek/client.py uses
# ``P.StreamEvent``); the parsers themselves live in sse_parser.py.
from .sse_parser import StreamEvent  # noqa: F401,E402

__all__ = [
    "StreamEvent",
    "parse_json",
    "base_headers",
    "bridge_headers",
    "completion_payload",
    "is_ban_payload",
    "extract_ban_timestamp",
    "is_ban_error_text",
    "format_ban_day_month",
    "format_ban_datetime",
    "format_ban_label",
    "ban_message",
    "with_ban_expiry",
    "detect_ban",
    "is_invalid_token_payload",
]

# Browser fingerprint used for every curl_cffi request (shared session).
IMPERSONATE = "chrome"

# Env override for host rotation survival (default preserves pinned host).
# Full discovery/fallback is deferred: this knob lets ops repoint without a
# release when chat.deepseek.com moves.
HOST = os.environ.get("ZOURCEWELL_HOST", "chat.deepseek.com").strip() or "chat.deepseek.com"
BASE = os.environ.get("ZOURCEWELL_BASE", f"https://{HOST}").strip() or f"https://{HOST}"
USERS_CURRENT_URL = f"{BASE}/api/v0/users/current"
SESSION_URL = f"{BASE}/api/v0/chat_session/create"
DELETE_SESSION_URL = f"{BASE}/api/v0/chat_session/delete"
POW_URL = f"{BASE}/api/v0/chat/create_pow_challenge"
COMPLETION_PATH = "/api/v0/chat/completion"
COMPLETION_URL = f"{BASE}{COMPLETION_PATH}"

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Ban family: completion returns biz_code 5 + "user is muted" with
# biz_data {"is_muted": 1, "mute_until": <unix ts>}; users/current returns
# the same shape under biz_data.chat when the token is still valid.
BANNED_BIZ_CODES = {"5", "5.0"}

# Ban word-match is shared (core.transport.BAN_WORD_RE): identical regex
# previously tripled here/Qwen/transport. The ban-envelope verdicts below
# stay DeepSeek-specific.


def _has_ban_word(text: str) -> bool:
    return _shared_has_ban_word(text)


def _as_float(value) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        num = float(value)
        if not math.isfinite(num) or num <= 0:
            return None
        # mute_until is unix seconds; a millisecond value scales down.
        if num > 1e10:
            num /= 1000.0
        # Beyond this datetime cannot represent it: unknown expiry, not a crash.
        if num > 253402300799:
            return None
        return num
    except (TypeError, ValueError):
        return None


def extract_ban_timestamp(data: dict) -> float | None:
    """Return mute_until unix timestamp when payload carries a ban, else None.

    Handles both shapes:
      completion: data.data.biz_data.{is_muted,mute_until}
      users/current: data.data.biz_data.chat.{is_muted,mute_until}
    """
    try:
        if not isinstance(data, dict):
            return None
        inner = data.get("data")
        if not isinstance(inner, dict):
            return None
        biz = inner.get("biz_data")
        if not isinstance(biz, dict):
            return None
        # Direct shape (completion ban).
        if biz.get("is_muted") in (1, True, "1"):
            ts = _as_float(biz.get("mute_until"))
            if ts is not None:
                return ts
            # is_muted without timestamp still counts as banned, but no expiry.
            return None
        # Nested shape (users/current).
        chat = biz.get("chat")
        if isinstance(chat, dict) and chat.get("is_muted") in (1, True, "1"):
            ts = _as_float(chat.get("mute_until"))
            # Return ts (may be None when server omits expiry).
            return ts
        return None
    except Exception:
        return None


def is_ban_payload(data: dict) -> bool:
    """True when envelope is a ban/mute, even without a parseable timestamp."""
    try:
        if not isinstance(data, dict):
            return False
        if extract_ban_timestamp(data) is not None:
            return True
        inner = data.get("data") if isinstance(data.get("data"), dict) else {}
        code = str(data.get("code", "") or "")
        biz_code = str((inner or {}).get("biz_code", "") or "")
        msg = f"{data.get('msg', '')} {(inner or {}).get('biz_msg', '')}"
        if biz_code in BANNED_BIZ_CODES or code in BANNED_BIZ_CODES:
            return True
        if _has_ban_word(msg):
            return True
        biz = (inner or {}).get("biz_data")
        if isinstance(biz, dict):
            if biz.get("is_muted") in (1, True, "1"):
                return True
            chat = biz.get("chat")
            if isinstance(chat, dict) and chat.get("is_muted") in (1, True, "1"):
                return True
        return False
    except Exception:
        return False


def is_ban_error_text(text: str) -> bool:
    return _has_ban_word(text or "")


# Invalid/expired token signal (biz 40003 "Authorization Failed"): not a
# ban, just unauthenticated. Banned accounts can still surface as 40003 on
# session/create, which is why the engine probes users/current before
# concluding "invalid" — but the probe itself treats 40003 as not-banned.
INVALID_TOKEN_CODES = {"40003"}


def is_invalid_token_payload(data: dict) -> bool:
    """True when the envelope is an invalid/expired token (biz 40003)."""
    try:
        if not isinstance(data, dict):
            return False
        if str(data.get("code", "") or "") in INVALID_TOKEN_CODES:
            return True
        inner = data.get("data")
        if isinstance(inner, dict) and str(inner.get("biz_code", "") or "") in INVALID_TOKEN_CODES:
            return True
        return False
    except Exception:
        return False


def detect_ban(data: dict) -> tuple[bool, float | None]:
    """Single ban-verdict authority: (banned, mute_until-or-None).

    Thin over is_ban_payload + extract_ban_timestamp so every caller shares
    one verdict instead of re-pairing the two helpers (check_ban,
    _ban_stream_event, and the SSE feed each had their own variant).
    40003 precedence stays with the caller (check_ban guards invalid-token
    first); detect_ban itself is a pure payload verdict.
    """
    try:
        if is_ban_payload(data):
            return True, extract_ban_timestamp(data)
        return False, None
    except Exception:
        return False, None


def _ban_dt(mute_until: float | None):
    """Local datetime for a unix ts, or None when it cannot be represented."""
    import datetime as _dt

    try:
        return _dt.datetime.fromtimestamp(float(mute_until))
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def format_ban_day_month(mute_until: float | None) -> str:
    """'16 September' from unix ts (local time, matches web UI wording); '' when unusable."""
    dt = _ban_dt(mute_until)
    if dt is None:
        return ""
    return f"{dt.day} {dt.strftime('%B')}"


def format_ban_datetime(mute_until: float | None) -> str:
    """'16 September 2026 12:49' from unix ts (local time); '' when unusable."""
    dt = _ban_dt(mute_until)
    if dt is None:
        return ""
    return f"{dt.day} {dt.strftime('%B')} {dt.year} {dt.strftime('%H:%M')}"


def format_ban_label(mute_until: float | None) -> str:
    """'(BANNED: 16 September)' — red wrapper applied by caller."""
    day_month = format_ban_day_month(mute_until)
    return f"(BANNED: {day_month})" if day_month else "(BANNED)"


def ban_message(mute_until: float | None, *, prefix: str = "") -> str:
    """Full human error: ban reason + expiry, mirrors web UI wording."""
    base = prefix or "Due to violation of user policies, your account has been suspended"
    stamp = format_ban_datetime(mute_until)
    if not stamp:
        return f"{base} (expiry unknown). If you have any questions, please contact us."
    return (f"{base} until {stamp}. "
            "If you have any questions, please contact us.")


def with_ban_expiry(msg: str, data: dict) -> str:
    """Append the ban expiry suffix when the envelope carries a ban."""
    try:
        if is_ban_payload(data):
            stamp = format_ban_datetime(extract_ban_timestamp(data))
            if stamp:
                return f"{msg} (suspended until {stamp})".strip()
    except Exception:
        pass
    return msg


def parse_json(resp) -> dict:
    """Best-effort ``resp.json()``: {} on any error, never raises."""
    return _shared_parse_json(resp)


def _build_headers(*, user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    """Single header builder shared by base_headers/bridge_headers."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": f"{BASE}/",
        "User-Agent": user_agent or CHROME_UA,
    }
    if waf_cookies:
        headers["Cookie"] = waf_cookies
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def base_headers(user_agent: str = "", waf_cookies: str = "", bearer: str = "") -> dict[str, str]:
    return _build_headers(user_agent=user_agent, waf_cookies=waf_cookies, bearer=bearer)


def bridge_headers(bearer: str = "") -> dict[str, str]:
    """Base headers enriched with a warm bridge's WAF cookies + browser UA.

    Single source for out-of-server callers (account pool ban probes): without
    the WAF cookie + matching UA, chat.deepseek.com blocks the request at the
    edge and a probe reads as a false "not banned". Falls back to plain
    base_headers when no bridge is registered (tests/offline).
    """
    try:
        from ...core.obscura_bridge import get_default_bridge
        bridge = get_default_bridge()
    except Exception:
        bridge = None
    if bridge is None:
        return base_headers(bearer=bearer)
    return base_headers(user_agent=bridge.state.user_agent,
                        waf_cookies=bridge.cookie_header(), bearer=bearer)


def completion_payload(session_id: str, prompt: str, thinking: bool, search: bool,
                       model_type: str = "default") -> dict:
    # Upstream 2026-09-15 rejects source="web" with 422:
    #   unknown variant `web`, expected `default` or `landing`.
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": model_type,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "source": "default",
        "action": None,
        "preempt": False,
    }
