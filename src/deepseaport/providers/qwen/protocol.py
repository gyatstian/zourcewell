"""Qwen web protocol: headers, payloads, WAF/ban detection.

The Qwen web API (chat.qwen.ai, ``/api/v2``) drops the DeepSeek ``{p,v}``
continuation protocol and the biz-error envelope: it is OpenAI-style SSE on
``/api/v2/chat/completions``. Header/payload/SSE shapes below are verified
live against the real chat.qwen.ai API; anything still unverified is marked
``# TODO(unverified)`` and left out instead of invented. SSE line parsing
lives in sse_parser.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

logger = logging.getLogger("deepseaport.protocol")

# Structural split (behavior-preserving): SSE parsing lives in sse_parser.py;
# re-exported so `deepseaport.protocol.QwenStreamParser` etc. keep working.
from .sse_parser import QwenStreamParser, StreamEvent, parse_sse_line  # noqa: F401,E402

__all__ = [
    "API_BASE",
    "AUTH_URL",
    "BASE",
    "CHAT_COMPLETIONS_STOP_URL",
    "CHAT_COMPLETIONS_URL",
    "CHAT_DELETE_URL",
    "CHATS_NEW_URL",
    "CHROME_UA",
    "COMPLETION_VERSION",
    "HOST",
    "IMPERSONATE",
    "MODELS_URL",
    "MODEL_META",
    "SIGNIN_URL",
    "StreamEvent",
    "QwenStreamParser",
    "parse_sse_line",
    "base_headers",
    "bridge_headers",
    "chat_new_payload",
    "completion_payload",
    "cookie_names",
    "describe_cookies",
    "detect_ban",
    "extract_ban_timestamp",
    "is_ban_error_text",
    "is_waf_challenge",
    "parse_json",
]

# Browser fingerprint used for every curl_cffi request (shared session).
IMPERSONATE = "chrome"

# Env overrides for host/base rotation survival (defaults preserve the pinned
# host). Full discovery/fallback is deferred: these knobs let ops repoint
# without a release when chat.qwen.ai moves.
HOST = os.environ.get("DEEPSEAPORT_QWEN_HOST", "chat.qwen.ai").strip() or "chat.qwen.ai"
BASE = os.environ.get("DEEPSEAPORT_QWEN_BASE", f"https://{HOST}").strip() or f"https://{HOST}"
API_BASE = f"{BASE}/api/v2"
MODELS_URL = f"{API_BASE}/models/"
CHATS_NEW_URL = f"{API_BASE}/chats/new"
CHAT_COMPLETIONS_URL = f"{API_BASE}/chat/completions"
CHAT_COMPLETIONS_STOP_URL = f"{API_BASE}/chat/completions/stop"
CHAT_DELETE_URL = f"{API_BASE}/chats/{{id}}"
# v1 signin: POST {email, password:<sha256-hex>} -> flat user object with a
# JWT ``token`` + ``token_type:"Bearer"`` + ``expires_at`` (live-verified).
SIGNIN_URL = f"{BASE}/api/v1/auths/signin"
# Best-effort auth/ban probe (read endpoint, NOT challenge-gated): unauth
# returns {success:false, data:{code:"Unauthorized", ...}}; a valid session
# returns success:true (live-verified probe shape).
AUTH_URL = f"{API_BASE}/auths/"

# Web app version header (mirrors the app's JS bundle). Distinct from the
# COMPLETION_VERSION used in the completions body — the app sends its own
# build version as the ``Version`` header and "2.1" inside completions.
VERSION = os.environ.get("DEEPSEAPORT_QWEN_VERSION", "0.2.91").strip() or "0.2.91"
# Completions body ``version`` field (live-verified from sendPromptOpenAI).
COMPLETION_VERSION = (
    os.environ.get("DEEPSEAPORT_QWEN_COMPLETION_VERSION", "2.1").strip() or "2.1")

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Model meta mirrored from the live /api/v2/models/ info.meta. Controls which
# optional keys the completion payload's feature_config carries:
#   - thinking_format is emitted only when the model meta has one AND thinking
#     is enabled (verified: qwen3.7-plus/qwen3.8-max use "summary").
#   - auto_search is emitted only when the model meta has the key; its value
#     is the request's search flag (verified: both search-capable models).
# qwen3.8-omni-flash carries neither key -> omit both.
MODEL_META: dict[str, dict] = {
    "qwen3.7-plus": {"thinking_format": "summary", "auto_search": True},
    "qwen3.8-max": {"thinking_format": "summary", "auto_search": True},
    "qwen3.8-omni-flash": {},
}

# Ban family: no verified Qwen ban envelope exists. Best-effort word matching
# on generic suspension wording. Letter boundaries (not \b: "_" must count as a
# boundary so "is_muted"/"mute_until" still hit) keep true bans while ignoring
# "commute". Shape is UNVERIFIED — see the TODO list in README.md.
# TODO(unverified): Qwen ban envelope shape unknown.
_BAN_RE = re.compile(
    r"(?<![a-z])(muted?|suspend\w*|bann?ed?|violat\w*)(?![a-z])",
    re.IGNORECASE,
)


def _has_ban_word(text: str) -> bool:
    try:
        return bool(text and _BAN_RE.search(text))
    except Exception:
        return False


def is_ban_error_text(text: str) -> bool:
    """Generic best-effort ban-word match; Qwen envelope shape unverified."""
    return _has_ban_word(text or "")


def extract_ban_timestamp(data: dict) -> float | None:
    """Always None: no verified Qwen ban payload carries mute_until.
    # TODO(unverified): reverse the real ban payload before returning a ts.
    """
    return None


def detect_ban(data: dict) -> tuple[bool, float | None]:
    """Generic best-effort (banned, mute_until). Always (False, None) until a
    real envelope is captured; kept so engine/account seams share a verdict.
    # TODO(unverified): Qwen ban envelope shape unknown.
    """
    try:
        if is_ban_error_text(str(data.get("message", ""))):
            return True, None
    except Exception:
        pass
    return False, None


def is_waf_challenge(resp) -> bool:
    """True when a 200 response is really the Aliyun WAF challenge page.

    The chat/qwen mutating endpoints answer a bot request with HTTP 200 +
    an HTML challenge (not 403) bearing ``aliyun_waf_aa`` / ``AliyunCaptcha``.
    The content-type check is cheap; the body check only applies on the
    completion path, where the caller has already drained lines. Never raises.
    """
    try:
        ctype = (resp.headers.get("content-type", "") or "") if hasattr(resp, "headers") else ""
        if "text/html" in ctype:
            return True
    except Exception:
        pass
    return False


def is_waf_body(body: str) -> bool:
    """True when drained body bytes hold WAF challenge markers."""
    low = (body or "").lower()
    return "aliyun_waf" in low or "aliyuncaptcha" in low or "nc-container" in low


def _merge_token_cookie(waf_cookies: str, token: str) -> str:
    """Merge ``token=<value>`` into a raw Cookie header string.

    Completions REQUIRE the ``token`` cookie: ``Authorization: Bearer`` alone
    makes the WAF return 200 text/html, while ``Cookie: token=<jwt>`` streams.
    Drop any existing ``token=...`` pair from browser/WAF cookies so the
    account token always wins over a profile token.
    """
    if not token:
        return waf_cookies or ""
    parts = []
    if waf_cookies:
        for piece in str(waf_cookies).split(";"):
            piece = piece.strip()
            if not piece:
                continue
            if piece.lower().startswith("token="):
                continue
            parts.append(piece)
    parts.append(f"token={token}")
    return "; ".join(parts)


def _omit_bearer() -> bool:
    """True when env asks to match web (source=web deletes Authorization).

    Real web (see recon) deletes ``Authorization`` for source=web/h5; session
    rides cookies. Default keeps Bearer+cookie (FastAPI accepts Bearer) so
    existing flows stay green; set DEEPSEAPORT_QWEN_NO_BEARER=1 to test the
    strict-web fingerprint when WAF persists with token+acw_tc.
    """
    try:
        return str(os.environ.get("DEEPSEAPORT_QWEN_NO_BEARER", "") or "").strip().lower() in ("1", "true", "yes")
    except Exception:
        return False


def _build_headers(*, user_agent: str = "", waf_cookies: str = "", bearer: str = "",
                   version: str = VERSION, source: str = "web") -> dict[str, str]:
    """Single header builder shared by base_headers/bridge_headers."""
    # Timezone must be latin-1-encodable: curl_cffi encodes headers as
    # latin-1, and some locales (e.g. Polish) return non-ASCII tzname
    # (u"\u015awit"). Fall back to UTC when the local name can't encode.
    try:
        tz = time.tzname[0] or "UTC"
        tz.encode("latin-1")
    except Exception:
        tz = "UTC"
    headers = {
        "Accept": "application/json, text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": BASE,
        "Referer": f"{BASE}/",
        "User-Agent": user_agent or CHROME_UA,
        "Version": version,
        "source": source,
        "X-Request-Id": str(uuid.uuid4()),
        "Timezone": tz,
    }
    # The web session authenticates via the ``token`` cookie (that is how the
    # JWT is planted on .qwen.ai after signin); always ship it alongside the
    # Authorization bearer so every hot path works the same (unless env asks
    # for strict-web: cookie-only, no Bearer).
    cookie = _merge_token_cookie(waf_cookies, bearer)
    if cookie:
        headers["Cookie"] = cookie
    if bearer and not (_omit_bearer() and source in ("web", "h5")):
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def base_headers(user_agent: str = "", waf_cookies: str = "", bearer: str = "",
                 version: str = VERSION, source: str = "web") -> dict[str, str]:
    return _build_headers(user_agent=user_agent, waf_cookies=waf_cookies,
                          bearer=bearer, version=version, source=source)


def cookie_names(cookie_header: str) -> list[str]:
    """Cookie names only, in order — safe for logging (never values)."""
    names: list[str] = []
    try:
        for piece in str(cookie_header or "").split(";"):
            piece = piece.strip()
            if not piece:
                continue
            name = piece.split("=", 1)[0].strip()
            if name:
                names.append(name)
    except Exception:
        pass
    return names


def describe_cookies(headers: dict | None) -> dict:
    """Names-only summary {names, has_token, has_acw_tc} — never values."""
    try:
        cookie = str((headers or {}).get("Cookie") or "")
    except Exception:
        cookie = ""
    names = cookie_names(cookie)
    low = [n.lower() for n in names]
    return {"names": names, "has_token": "token" in low,
            "has_acw_tc": "acw_tc" in low}


_BRIDGE_UNSET: object = object()


def bridge_headers(bearer: str = "", bridge: object = _BRIDGE_UNSET) -> dict[str, str]:
    """Base headers enriched with a warm bridge's WAF cookies + browser UA.

    Without the WAF ``acw_tc`` cookie + matching UA, chat.qwen.ai blocks
    mutating endpoints at the edge (Aliyun WAF). Our programmatic client DOES
    send ``Authorization: Bearer`` (FastAPI backend accepts it) even though
    the web app deletes it for source=web — design decision, see README.
    Falls back to plain base_headers when no bridge is registered.

    Pass an explicit ``bridge`` (e.g. ``app.state.bridge``) to avoid the
    ``app.state.bridge`` vs ``get_default_bridge()`` divergence: callers
    holding the request bridge must use its cookies, not a stale default.
    ``bridge=None`` forces the no-bridge fallback; omit to use the default.
    """
    if bridge is _BRIDGE_UNSET:
        try:
            from ...core.obscura_bridge import get_default_bridge
            bridge = get_default_bridge()
        except Exception:
            bridge = None
    if bridge is None:
        return base_headers(bearer=bearer)
    try:
        ua = bridge.state.user_agent  # type: ignore[union-attr]
        cookies = bridge.cookie_header()  # type: ignore[union-attr]
    except Exception:
        ua, cookies = "", ""
    return base_headers(user_agent=ua, waf_cookies=cookies, bearer=bearer)


def chat_new_payload(model: str, chat_id: str = "", chat_type: str = "t2t",
                     chat_mode: str = "normal") -> dict:
    """Verified POST /api/v2/chats/new body; success returns data.id."""
    return {
        "chatId": chat_id,
        "models": [model],
        "project_id": None,
        "timestamp": int(time.time()),
        "chat_type": chat_type,
        "chat_mode": chat_mode,
    }


def completion_payload(chat_id: str, prompt: str, model: str, thinking: bool,
                       search: bool, stream: bool = True) -> dict:
    """Live-verified POST /api/v2/chat/completions body (sendPromptOpenAI).

    ``version`` is the completions version (default "2.1", NOT the app header
    version). The user message item uses ``user_action:"chat"``, empty
    ``models``/``model`` (the model lives at top level), ``sub_chat_type`` +
    ``extra.meta.subChatType``, and NO ``files`` key for a plain text message.
    ``feature_config`` carries ``thinking_format`` only when the model meta has
    one AND thinking is on; ``auto_search`` only when the meta has the key —
    its value is the request's search flag (live-verified mapping).
    """
    now = int(time.time())
    meta = MODEL_META.get(model) or {}
    feature: dict = {
        "thinking_enabled": bool(thinking),
        "output_schema": "phase",
        "research_mode": "normal",
    }
    if meta.get("thinking_format") and thinking:
        feature["thinking_format"] = meta["thinking_format"]
    if "auto_search" in meta:
        feature["auto_search"] = bool(search)
    message = {
        "id": None,
        "fid": str(uuid.uuid4()),
        "parentId": "",
        "childrenIds": [],
        "role": "user",
        "content": prompt,
        "user_action": "chat",
        "timestamp": now,
        "models": [],
        "model": "",
        "chat_type": "t2t",
        "feature_config": feature,
        "extra": {"meta": {"subChatType": "t2t"}},
        "sub_chat_type": "t2t",
    }
    return {
        "stream": bool(stream),
        "version": COMPLETION_VERSION,
        "incremental_output": True,
        "chatId": chat_id,
        "chat_id": chat_id,
        "parentId": "",
        "parent_id": None,
        "chat_mode": "normal",
        "model": model,
        "messages": [message],
        "timestamp": now,
        "stream_options": {"include_usage": True},
    }


def parse_json(resp) -> dict:
    """Best-effort ``resp.json()``: {} on any error, never raises."""
    try:
        return resp.json()
    except Exception:
        return {}