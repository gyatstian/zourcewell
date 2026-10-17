"""Qwen web client: signin, session, chats, SSE completion (sync, thread-safe).

No PoW/wasm: Qwen's WAF is Aliyun anti-bot; recovery is a bridge warmup
(retry), not a proof-of-work solve. Direct-API login (``signin``) works from a
plain network — no browser or captcha — and yields the JWT that rides the
``token`` cookie on every hot path.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Iterator
from urllib.parse import quote

from curl_cffi import requests as crequests

from . import protocol as P
from ...core import transport as T
# Backward-compat re-exports: canonical SSE parsing lives in sse_parser.py,
# header/payload helpers in protocol.py — still importable from client.
from .protocol import (  # noqa: F401
    base_headers,
    bridge_headers,
    chat_new_payload,
    completion_payload,
    parse_json,
)
from .sse_parser import QwenStreamParser, StreamEvent, parse_sse_line  # noqa: F401

logger = logging.getLogger("deepseaport.client")

# Backwards-compatible alias: the fingerprint lives in protocol as the single
# source shared with the account pool.
IMPERSONATE = P.IMPERSONATE
# Per-request stream timeout. curl_cffi treats this as a low-speed/stall
# timeout for streaming (read timeout), not a hard wall-clock cap, but keep it
# large enough for long reasoning generations. The server's consumer wait is
# derived from this value so the two cannot drift apart.
DEFAULT_TIMEOUT = 300

# Shared session: reuse HTTP/2 connections + TLS across calls. curl_cffi
# thread-safe: one curl handle per thread (thread-local), so concurrent
# callers get isolation while each thread reuses its connection. The cookie
# jar is discarded: the Cookie header is set manually per request and WAF
# Set-Cookie responses must not mutate a shared jar.
_SESSION = crequests.Session(impersonate=IMPERSONATE, discard_cookies=True)

# Qwen-host WAF cache: cookies harvested from Set-Cookie on the qwen open
# endpoint (GET MODELS_URL) despite discard_cookies=True, plus full-jar
# obscura fetch to chat.qwen.ai (acw_tc 64-char solved + x-ap + sca/cna/
# tfstk/isg/xlly_s). Curl prime alone yields bare acw_tc (short, unsolved)
# which passes chats/new but NOT completions; the solved jar is required.
# The shared obscura bridge warms chat.deepseek.com (never qwen), so without
# this the Cookie header carries token= but has_acw_tc=False. Thread-safe,
# values never logged (names-only diagnostics).
_QWEN_WAF_CACHE: dict[str, str] = {}
_QWEN_WAF_LOCK = threading.Lock()
_ACW_TC_RE = re.compile(r"\bacw_tc=([^;,\s]+)", re.IGNORECASE)
_X_AP_RE = re.compile(r"\bx-ap=([^;,\s]+)", re.IGNORECASE)
# Allowlist for bridge-jar merge (never store token=/Authorization).
_QWEN_WAF_ALLOW = frozenset({"acw_tc", "x-ap", "sca", "cna", "tfstk", "isg", "xlly_s"})

# Qwen's WAF answers 200 + text/html, never 403 (see protocol.is_waf_challenge),
# so the shared transport default (DeepSeek: 403 == WAF challenge) must NOT be
# applied here: a real Qwen 403 carries a genuine upstream status — e.g. a
# suspension body — and labelling it HTTP_403_WAF makes the classifier pick the
# WAF branch, so the account is never cooled down or failed over. Empty mapping
# = report a plain HTTP_<status>.
_FORBIDDEN_CODES: dict[int, str] = {}


def _is_winerror5(exc: BaseException) -> bool:
    """True for WinError 5 / profile-dir access-denied (graceful, never raise)."""
    try:
        if getattr(exc, "winerror", None) == 5:
            return True
        if isinstance(exc, PermissionError):
            return True
        msg = str(exc) or ""
        if "WinError 5" in msg or "[WinError 5]" in msg:
            return True
        if "Access is denied" in msg and "profile" in msg.lower():
            return True
    except Exception:
        pass
    return False


def _raw_set_cookie_values(resp) -> list[str]:
    """Best-effort Set-Cookie extraction (dict or curl_cffi header object)."""
    try:
        headers = getattr(resp, "headers", None)
        if not headers:
            return []
        # curl_cffi / httpx style multi-value helpers first.
        for meth in ("get_list", "get_all", "getlist"):
            try:
                fn = getattr(headers, meth, None)
                if callable(fn):
                    for key in ("set-cookie", "Set-Cookie"):
                        try:
                            vals = fn(key)
                        except Exception:
                            continue
                        if vals:
                            if isinstance(vals, (list, tuple)):
                                return [str(v) for v in vals if v]
                            return [str(vals)]
            except Exception:
                pass
        # Plain dict-style single value (case-insensitive probe).
        try:
            if hasattr(headers, "get"):
                for key in ("Set-Cookie", "set-cookie", "SET-COOKIE"):
                    try:
                        val = headers.get(key)
                    except Exception:
                        val = None
                    if val:
                        if isinstance(val, (list, tuple)):
                            return [str(v) for v in val if v]
                        return [str(val)]
            elif isinstance(headers, dict):
                for k, v in headers.items():
                    try:
                        if str(k).lower() == "set-cookie" and v:
                            if isinstance(v, (list, tuple)):
                                return [str(x) for x in v if x]
                            return [str(v)]
                    except Exception:
                        continue
        except Exception:
            pass
    except Exception:
        pass
    return []


def _store_waf_cookies(mapping: dict) -> dict[str, str]:
    """Store allowlisted qwen WAF cookies into cache. Returns stored subset."""
    stored: dict[str, str] = {}
    try:
        for k, v in (mapping or {}).items():
            try:
                name = str(k or "").strip()
                if not name or name.lower() in ("token", "authorization"):
                    continue
                if name not in _QWEN_WAF_ALLOW and name.lower() not in _QWEN_WAF_ALLOW:
                    # Keep acw_tc/x-ap even with case drift; drop unknowns.
                    continue
                val = str(v or "").strip().strip('"')
                if val:
                    stored[name] = val
            except Exception:
                continue
        if stored:
            try:
                with _QWEN_WAF_LOCK:
                    _QWEN_WAF_CACHE.update(stored)
            except Exception:
                pass
            try:
                logger.debug("qwen harvested waf cookie_names=%s has_acw_tc=%s",
                             sorted(stored), "acw_tc" in [s.lower() for s in stored])
            except Exception:
                pass
    except Exception:
        pass
    return stored


def _harvest_waf_cookies(resp) -> dict[str, str]:
    """Harvest acw_tc (+x-ap) from Set-Cookie into _QWEN_WAF_CACHE. Names-only."""
    try:
        vals = _raw_set_cookie_values(resp)
        found: dict[str, str] = {}
        for raw in vals:
            try:
                text = str(raw)
                m = _ACW_TC_RE.search(text)
                if m and m.group(1):
                    v = m.group(1).strip().strip('"')
                    if v and "acw_tc" not in found:
                        found["acw_tc"] = v
                m2 = _X_AP_RE.search(text)
                if m2 and m2.group(1):
                    v2 = m2.group(1).strip().strip('"')
                    if v2 and "x-ap" not in found:
                        found["x-ap"] = v2
            except Exception:
                continue
        if found:
            return _store_waf_cookies(found)
    except Exception:
        pass
    return {}


def qwen_waf_cookie_dict() -> dict[str, str]:
    """Copy of cached qwen WAF cookies (never token). Values caller-guarded."""
    try:
        with _QWEN_WAF_LOCK:
            return dict(_QWEN_WAF_CACHE)
    except Exception:
        return {}


def qwen_waf_cookie_header() -> str:
    """Cached 'acw_tc=...' fragment ('' when empty). Never logs values."""
    try:
        with _QWEN_WAF_LOCK:
            val = _QWEN_WAF_CACHE.get("acw_tc") or ""
        val = str(val).strip()
        return f"acw_tc={val}" if val else ""
    except Exception:
        return ""


def _merged_waf_cookies(base: str) -> str:
    """Merge cached qwen jar into a Cookie string (no dup, no token change)."""
    try:
        cached = qwen_waf_cookie_dict()
        if not cached:
            return base or ""
        have = set(n.lower() for n in P.cookie_names(base or ""))
        missing = []
        for k, v in cached.items():
            try:
                if str(k).lower() in have:
                    continue
                vv = str(v or "").strip()
                if vv:
                    missing.append(f"{k}={vv}")
            except Exception:
                continue
        if not missing:
            return base or ""
        base = (base or "").strip().rstrip(";").strip()
        frag = "; ".join(missing)
        return f"{base}; {frag}" if base else frag
    except Exception:
        return base or ""


def prime_qwen_bridge(bridge, timeout: int = 30) -> bool:
    """Best-effort solved-jar fetch via obscura to chat.qwen.ai (NOT deepseek).

    Runs ``bridge._run fetch <BASE/> --dump cookies`` in the bridge engine so
    Aliyun JS executes and plants the solved acw_tc (64-char) + x-ap + sca/
    cna/tfstk/isg/xlly_s. Parses via bridge._parse_cookie_dump and stores
    allowlisted names. WinError5/profile-lock is graceful, names-only, never
    raises. True when acw_tc present afterwards.
    """
    try:
        if bridge is None:
            return False
        try:
            from ...core.obscura_bridge import profile_lock as _plock
            try:
                prof = getattr(bridge, "profile", "") or ""
                lock = _plock(prof) if prof else None
            except Exception:
                lock = None
        except Exception:
            lock = None
        url = f"{P.BASE}/"
        args = ["fetch", url, "--dump", "cookies", "--timeout", "20", "--quiet"]
        proc = None
        try:
            if lock is not None:
                with lock:
                    proc = bridge._run(*args, timeout=timeout)  # type: ignore[attr-defined]
            else:
                proc = bridge._run(*args, timeout=timeout)  # type: ignore[attr-defined]
        except OSError as exc:
            if _is_winerror5(exc):
                logger.warning("qwen bridge fetch WinError5 (profile locked), "
                               "continuing with curl prime")
            else:
                logger.debug("qwen bridge fetch OSError: %s", exc)
            return bool(qwen_waf_cookie_header())
        except Exception as exc:
            logger.debug("qwen bridge fetch ignored: %s", exc)
            return bool(qwen_waf_cookie_header())
        try:
            dump = getattr(proc, "stdout", "") or ""
            cookies = bridge._parse_cookie_dump(dump)  # type: ignore[attr-defined]
        except Exception:
            cookies = {}
        if cookies:
            _store_waf_cookies(cookies)
        return bool(qwen_waf_cookie_header())
    except Exception:
        return bool(qwen_waf_cookie_header())


def _clear_waf_cache() -> None:
    """Test-only helper to reset the harvested cache."""
    try:
        with _QWEN_WAF_LOCK:
            _QWEN_WAF_CACHE.clear()
    except Exception:
        pass


def _json(resp) -> dict:
    return P.parse_json(resp)


def _ensure_token_cookie(headers: dict) -> dict:
    """Ensure Cookie carries ``token=<bearer>`` (account token always wins).

    Defensive: callers build headers via ``_headers`` (which merges), but a
    stale/cached dict may carry an old profile ``token=`` or none at all.
    Re-merge on every hot-path call so ``token=`` always equals the
    ``Authorization`` bearer. Mutates + returns.
    """
    try:
        hdrs = headers or {}
        auth = str(hdrs.get("Authorization") or "")
        if auth.startswith("Bearer "):
            jwt = auth[len("Bearer "):].strip()
            if jwt:
                hdrs["Cookie"] = P._merge_token_cookie(
                    str(hdrs.get("Cookie") or ""), jwt)
        # Completions carry X-Accel-Buffering:no in the web app (see recon).
        # Harmless on chats/new, required for SSE path fidelity.
        try:
            if "X-Accel-Buffering" not in hdrs:
                hdrs["X-Accel-Buffering"] = "no"
        except Exception:
            pass
    except Exception:
        pass
    return headers


def _log_waf_headers(where: str, headers: dict) -> None:
    """Cookie-names-only WAF diagnostic (never values/secrets)."""
    try:
        desc = P.describe_cookies(headers)
        logger.warning("qwen WAF %s cookie_names=%s has_token=%s has_acw_tc=%s",
                       where, desc["names"], desc["has_token"], desc["has_acw_tc"])
    except Exception:
        pass


def prime_waf(headers: dict, timeout: int = 20) -> bool:
    """Best-effort qwen WAF priming via qwen-host open GET models/ (NOT gated).

    Targets chat.qwen.ai (P.MODELS_URL) — never the DeepSeek host — so the
    edge returns Set-Cookie acw_tc. Harvests acw_tc despite
    discard_cookies=True into _QWEN_WAF_CACHE. Never raises; WinError5
    (profile-dir lock) is graceful with names-only logging. True when the
    models envelope parses.
    """
    try:
        resp = _SESSION.get(P.MODELS_URL, headers=headers or {},
                            impersonate=IMPERSONATE, timeout=timeout)
        try:
            # Harvest before parse so acw_tc lands even on envelope miss.
            try:
                _harvest_waf_cookies(resp)
            except Exception:
                pass
            data = P.parse_json(resp)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        inner = data.get("data") if isinstance(data, dict) and isinstance(
            data.get("data"), dict) else {}
        models = inner.get("data")
        return isinstance(models, list)
    except OSError as exc:
        if _is_winerror5(exc):
            try:
                _log_waf_headers("prime_waf_winerror5", headers or {})
            except Exception:
                pass
            logger.warning("qwen prime_waf WinError5 (profile dir locked), "
                           "continuing without fresh acw_tc")
        else:
            logger.debug("prime_waf ignored: %s", exc)
        return False
    except Exception as exc:
        logger.debug("prime_waf ignored: %s", exc)
        return False


def _with_chat_id(base: str, payload: dict | None) -> str:
    """Append ``?chat_id=<id>`` when the payload carries a chat id.

    Live-verified: POST /api/v2/chat/completions REQUIRES the query param
    (missing → ``RequestValidationError: Field 'query -> chat_id': Field
    required``). The stop endpoint takes the same param. Falls back to
    ``chatId`` when ``chat_id`` is absent; bare base URL when neither key
    is present (preserves minimal-payload callers).
    """
    try:
        data = payload or {}
        cid = data.get("chat_id") or data.get("chatId")
    except Exception:
        cid = None
    if not cid:
        return base
    return f"{base}?chat_id={quote(str(cid), safe='')}"


def signin(email: str, password: str, timeout: int = 30) -> dict:
    """POST /api/v1/auths/signin -> flat user dict (JWT in ``token``).

    Direct-API login, no browser/captcha needed (live-verified). The password
    is SHA-256 hex-encoded (the web app does the same). Success is a FLAT user
    object (NOT ``{success,...}``) keyed ``id, email, name, role, ...,
    token, token_type:"Bearer", expires_at``. Failures:
      400 -> ``{"detail":{"code":"INVALID_CRED","details":"..."}}`` or
            ``{"detail":"email not found"}`` (single string).
      200 + text/html -> Aliyun WAF challenge page.
    Raises RuntimeError with the surfaced code/details; never crashes on odd
    shapes or non-JSON bodies.
    """
    digest = hashlib.sha256((password or "").encode("utf-8")).hexdigest()
    body = {"email": (email or "").strip(), "password": digest}
    resp = _SESSION.post(P.SIGNIN_URL, headers=P.base_headers(source="web"),
                         json=body, impersonate=IMPERSONATE, timeout=timeout)
    if resp.status_code == 200 and P.is_waf_challenge(resp):
        resp.close()
        raise RuntimeError(
            "WAF_CHALLENGE: signin returned the Aliyun challenge page "
            "(200 text/html) — warm up the bridge and retry")
    data = _json(resp)
    status = resp.status_code
    resp.close()
    if status == 200 and isinstance(data, dict) and data.get("token"):
        return data
    detail = data.get("detail", data) if isinstance(data, dict) else data
    code: str = ""
    msg: str = ""
    if isinstance(detail, dict):
        code = str(detail.get("code") or "")
        msg = str(detail.get("details") or detail.get("message") or "")
    elif isinstance(detail, str):
        msg = detail
    if code or msg:
        raise RuntimeError(f"Qwen signin failed {code}: {msg}".replace("  ", " "))
    raise RuntimeError(
        f"Qwen signin failed: unexpected response (HTTP {status}): {str(data)[:200]}")


def _raise_biz_error(data: dict) -> None:
    """Raise RuntimeError on a biz error envelope (success:false / top error).

    Qwen envelopes are ``{success, request_id, data}``; failures put a code in
    ``data.code`` (e.g. ``{data:{code:"Unauthorized"}}``). Ban shapes are
    unverified — this never claims a ban, just surfaces code+message.
    """
    if not isinstance(data, dict):
        return
    success = data.get("success")
    if success is True or success in (1, "1"):
        return
    inner = data.get("data")
    code = ""
    msg = ""
    if isinstance(inner, dict):
        code = str(inner.get("code") or "")
        msg = str(inner.get("details") or inner.get("message") or "")
    err = data.get("error")
    if isinstance(err, dict):
        code = code or str(err.get("code") or err.get("type") or "")
        msg = msg or str(err.get("message") or "")
    elif isinstance(err, str):
        msg = msg or err
    if code or msg:
        raise RuntimeError(f"Qwen API error {code}: {msg}")


def create_chat(headers: dict, model: str, chat_type: str = "t2t") -> str:
    """POST /api/v2/chats/new; return the chat id (data.id).

    WAF-gated: a bot request returns HTTP 200 with an HTML challenge page.
    Raise RuntimeError carrying WAF + challenge so the engine classifier
    matches it as a WAF retry.
    """
    _ensure_token_cookie(headers)
    resp = _SESSION.post(P.CHATS_NEW_URL, headers=headers,
                         json=P.chat_new_payload(model, chat_type=chat_type),
                         impersonate=IMPERSONATE, timeout=30)
    if resp.status_code == 200 and P.is_waf_challenge(resp):
        resp.close()
        _log_waf_headers("create_chat", headers)
        raise RuntimeError(
            "WAF_CHALLENGE: /api/v2/chats/new returned the Aliyun WAF "
            "challenge page (200 text/html) — warm up the bridge and retry")
    data = _json(resp)
    resp.close()
    try:
        _raise_biz_error(data)
    except RuntimeError as exc:
        raise RuntimeError(f"create_chat failed: {exc}") from exc
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    chat_id = inner.get("id")
    if not chat_id:
        raise RuntimeError(
            f"create_chat: no data.id (HTTP {resp.status_code}): {resp.text[:200]}")
    return str(chat_id)


def delete_chat(headers: dict, chat_id: str) -> None:
    """Best-effort DELETE /api/v2/chats/{id}; swallow exceptions."""
    try:
        _SESSION.delete(P.CHAT_DELETE_URL.format(id=chat_id), headers=headers,
                        impersonate=IMPERSONATE, timeout=10)
    except Exception as exc:
        logger.debug("delete_chat ignored: %s", exc)


def fetch_models(headers: dict, timeout: int = 20) -> list[dict]:
    """GET qwen-host /api/v2/models/ -> data.data list; harvest acw_tc.

    Open endpoint on chat.qwen.ai (never DeepSeek host). Harvests Set-Cookie
    acw_tc despite discard_cookies=True. WinError5 graceful, names-only logs.
    Best-effort, [] on failure.
    """
    try:
        resp = _SESSION.get(P.MODELS_URL, headers=headers,
                            impersonate=IMPERSONATE, timeout=timeout)
        try:
            try:
                _harvest_waf_cookies(resp)
            except Exception:
                pass
            data = _json(resp)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        inner = data.get("data") if isinstance(data, dict) and isinstance(
            data.get("data"), dict) else {}
        models = inner.get("data")
        if isinstance(models, list):
            return models
    except OSError as exc:
        if _is_winerror5(exc):
            try:
                _log_waf_headers("fetch_models_winerror5", headers or {})
            except Exception:
                pass
            logger.warning("qwen fetch_models WinError5 (profile dir locked), "
                           "continuing without fresh acw_tc")
        else:
            logger.debug("fetch_models ignored: %s", exc)
    except Exception as exc:
        logger.debug("fetch_models ignored: %s", exc)
    return []


def check_ban(headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
    """Return (is_banned, mute_until); best-effort, never raises.

    Uses GET /api/v2/auths/ (read endpoint, NOT challenge-gated): a valid
    session returns success:true. When the headers carry an Authorization
    bearer, merge it into the Cookie ``token`` as well so the probe works
    with a real token (completions require the cookie). ``success:false``
    runs the shared ``detect_ban`` verdict: explicit mute/ban flags or ban
    wording -> (True, mute_until-or-None) so banned accounts cool down and
    fail over; plain ``Unauthorized``/``401`` (invalid token) and HTML
    challenge reads stay (False, None) — an invalid token is not a ban.
    # TODO(unverified): no verified Qwen ban envelope yet — the verdict is
    # best-effort (flags + generic wording) until a real suspended-account
    # payload is captured.
    """
    try:
        hdrs = dict(headers or {})
        auth = str(hdrs.get("Authorization") or "")
        if auth.startswith("Bearer "):
            jwt = auth[len("Bearer "):].strip()
            hdrs["Cookie"] = P._merge_token_cookie(
                str(hdrs.get("Cookie") or ""), jwt)
        resp = _SESSION.get(P.AUTH_URL, headers=hdrs,
                            impersonate=IMPERSONATE, timeout=timeout)
        try:
            if P.is_waf_challenge(resp):
                return False, None
            data = _json(resp)
            if isinstance(data, dict):
                if data.get("success") is True:
                    return False, None
                try:
                    return P.detect_ban(data)
                except Exception:
                    return False, None
            return False, None
        finally:
            try:
                resp.close()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("check_ban ignored: %s", exc)
        return False, None


def stream_completion(headers: dict, payload: dict,
                      timeout: int = DEFAULT_TIMEOUT) -> Iterator[StreamEvent]:
    """POST /api/v2/chat/completions?chat_id=<id> and yield parsed events.

    Caller must close on break. Maps: 200+HTML challenge -> WAF_CHALLENGE
    error event; 200+JSON biz error -> envelope error (with ban_until from
    the shared detect_ban verdict when the envelope carries a ban signal);
    200+non-SSE empty body -> UPSTREAM_ERROR (a drained stream must never be
    re-iterated); non-200 -> build_http_error_event; else SSE parse via
    QwenStreamParser.
    """
    _ensure_token_cookie(headers)
    resp = _SESSION.post(_with_chat_id(P.CHAT_COMPLETIONS_URL, payload),
                         headers=headers, json=payload,
                         impersonate=IMPERSONATE, timeout=timeout, stream=True)
    if resp.status_code == 200 and P.is_waf_challenge(resp):
        resp.close()
        _log_waf_headers("stream_completion", headers)
        yield StreamEvent(
            kind="error", code="WAF_CHALLENGE",
            message="Qwen completions returned the Aliyun WAF challenge page "
                    "(200 text/html) — warm up the bridge and retry")
        return
    if resp.status_code == 200 and not T.is_sse_response(resp):
        # Some errors arrive as JSON with HTTP 200 (e.g. unauthorized
        # envelope). With stream=True the body is not buffered, so resp.json()
        # is empty — drain via iter_lines (shared cap/helpers).
        try:
            body = T.drain_json_text(resp, T.JSON_DRAIN_CAP)
            if P.is_waf_body(body):
                T.close_quietly(resp)
                _log_waf_headers("stream_completion_body", headers)
                yield StreamEvent(
                    kind="error", code="WAF_CHALLENGE",
                    message="Qwen completions returned a WAF challenge body")
                return
            data = T.loads_lenient(body) if body else {}
            if isinstance(data, dict) and data.get("success") is False:
                T.close_quietly(resp)
                inner = data.get("data") if isinstance(data.get("data"), dict) else {}
                code = str(inner.get("code") or data.get("code") or "UPSTREAM_ERROR")
                msg = str(inner.get("details") or inner.get("message")
                          or data.get("message") or body)
                ban_until: float | None = None
                try:
                    banned, ban_until = P.detect_ban(data)
                    if not banned:
                        ban_until = None
                except Exception:
                    ban_until = None
                yield StreamEvent(kind="error", code=code,
                                  message=T.truncate_message(msg),
                                  ban_until=ban_until)
                return
            if body:
                T.close_quietly(resp)
                yield StreamEvent(kind="error", code="UPSTREAM_ERROR",
                                  message=T.truncate_message(body))
                return
        except StopIteration:
            pass
        except Exception:
            pass
        T.close_quietly(resp)
        # The drain above consumed the only body this response has: falling
        # through to the SSE pump iterated a drained, closed curl_cffi stream
        # a SECOND time, which never yields EOF and never raises -- the worker
        # parked forever and its account slot was never released (reproduced
        # against curl_cffi 0.16.3).
        yield StreamEvent(
            kind="error", code="UPSTREAM_ERROR",
            message="empty non-SSE 200 body")
        return
    if resp.status_code != 200:
        event = T.build_http_error_event(resp, payload, _FORBIDDEN_CODES)
        T.close_quietly(resp)
        yield event
        return
    # Shared decode/close scaffolding (transport.pump_sse); the per-line WAF
    # guard + Qwen wire parser stay Qwen-specific (DeepSeek uses its own
    # parser + raw-JSON ban guard).
    def _qwen_waf_precheck(line: str) -> StreamEvent | None:
        if P.is_waf_body(line):
            _log_waf_headers("stream_completion_line", headers)
            return StreamEvent(
                kind="error", code="WAF_CHALLENGE",
                message="Qwen completions returned a WAF challenge line")
        return None

    yield from T.pump_sse(resp, QwenStreamParser(), precheck=_qwen_waf_precheck)
