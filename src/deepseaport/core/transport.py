"""HTTP transport helpers: status/body/error-drain handling for completion.

Behavior-preserving extraction from client.stream_completion (structural
split only). No ban classification, retry-code, string-matching, or cap
changes here — caps (20k/4k) and truncations are preserved verbatim.

Ownership:
- This module owns HTTP-level concerns: content-type checks, .text vs
  iter_lines draining (stream=True leaves .text/.json() empty), 422
  detail surfacing, the injectable 403/forbidden -> event-code mapping
  (default: DeepSeek's HTTP_403_WAF), and the warning log shape.
- SSE line/event parsing lives in sse_parser.py.
- Ban-envelope mapping (_biz_error/_ban_stream_event) stays in client.py;
  this module only moves the byte-handling primitives so client.py keeps
  its _SESSION hook (tests monkeypatch client._SESSION).
"""

from __future__ import annotations

import json as _jsonlib
import logging
import re as _re
from typing import Any, Callable, Iterator

from .types import StreamEvent

# Preserve the original log channel so existing filters see the same records.
logger = logging.getLogger("deepseaport.client")

# Drain caps (bytes-ish chars): 20000 for HTTP-200 JSON bodies arriving as
# raw JSON lines, 4000 for non-200 error drains. Do not change (later batch).
JSON_DRAIN_CAP = 20000
ERROR_DRAIN_CAP = 4000
# Truncation limits for surfaced error bodies. Do not change (later batch).
ERROR_BODY_MAX = 2000
DETAIL_SLICE_MAX = 1500
UPSTREAM_MESSAGE_MAX = 300


def decode_line(raw: bytes | str) -> str:
    """Decode one iter_lines item as utf-8/replace (bytes) or str()."""
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def content_type_of(resp: Any) -> str:
    """Best-effort content-type fetch: "" on any error, never raises."""
    try:
        return resp.headers.get("content-type", "") or ""
    except Exception:
        return ""


def is_sse_response(resp: Any) -> bool:
    """True when the response claims text/event-stream (may still be empty)."""
    return "text/event-stream" in (content_type_of(resp) or "")


def drain_lines(resp: Any, cap: int) -> list[str]:
    """Drain iter_lines: skip empties, decode, stop once chars exceed cap.

    With stream=True the body is not buffered, so resp.json()/.text can be
    empty — callers drain via iter_lines instead. Cap preserves the
    20k/4k bounds from client.stream_completion.
    """
    out: list[str] = []
    total = 0
    for raw in resp.iter_lines():
        if not raw:
            continue
        s = decode_line(raw)
        out.append(s)
        total += len(s)
        if total > cap:
            break
    return out


def read_text_body(resp: Any) -> str:
    """Best-effort resp.text: "" on any error, never raises."""
    try:
        return resp.text or ""
    except Exception:
        return ""


def read_error_body(resp: Any) -> tuple[str, str]:
    """Return (ctype, body) for non-200 paths: .text, else drain (cap 4k).

    stream=True can leave .text empty: drain a few lines instead.
    Body is stripped and truncated to 2000 chars (preserved).
    """
    ctype = content_type_of(resp)
    raw_body = read_text_body(resp)
    if not raw_body:
        # stream=True can leave .text empty: drain a few lines instead.
        try:
            raw_body = "".join(drain_lines(resp, ERROR_DRAIN_CAP))
        except Exception:
            pass
    return ctype, (raw_body or "").strip()[:ERROR_BODY_MAX]


def enrich_422_body(body: str) -> str:
    """Surface FastAPI-style 422 {"detail": [...]} compactly.

    Preserves: only when body starts with "{", detail appended as
    " | detail=<json>" with 1500-char slices, total capped at 2000.
    """
    # FastAPI-style 422 carries {"detail": [...]}: surface it compactly
    # so the server log shows the failing field, not just "HTTP_422:".
    if body.startswith("{"):
        try:
            parsed = _jsonlib.loads(body)
            if isinstance(parsed, dict) and parsed.get("detail") is not None:
                detail_s = _jsonlib.dumps(parsed.get("detail"), ensure_ascii=False)
                body = f"{body[:DETAIL_SLICE_MAX]} | detail={detail_s[:DETAIL_SLICE_MAX]}"[:ERROR_BODY_MAX]
        except Exception:
            pass
    return body


# Per-provider 403 -> event-code mapping (see http_error_code).
_DEEPSEEK_FORBIDDEN: dict[int, str] = {403: "HTTP_403_WAF"}


def http_error_code(status_code: Any,
                    forbidden: dict[int, str] | None = None) -> str:
    """Map an HTTP status to an event code.

    The 403 meaning is per-provider (PROVIDERS.md: these web APIs differ), so
    the mapping is injectable instead of hardcoded here. ``forbidden=None``
    keeps the historical DeepSeek default (403 -> ``HTTP_403_WAF``; a DeepSeek
    403 is always the AWS-WAF challenge), and a 1-arg call is unchanged.
    Qwen passes an empty mapping: its WAF answers 200 + text/html (see
    providers/qwen/protocol.py), so a 403 there is a real upstream status
    (e.g. a suspension body) and must stay a plain ``HTTP_403``.
    """
    table = _DEEPSEEK_FORBIDDEN if forbidden is None else forbidden
    try:
        code = (table or {}).get(int(status_code))
    except (TypeError, ValueError):
        code = None
    return code or f"HTTP_{status_code}"


def describe_payload(payload: Any) -> tuple[list[str], int]:
    """Return (sorted_keys, prompt_len) for the completion warning log."""
    try:
        payload_keys = sorted((payload or {}).keys()) if isinstance(payload, dict) else []
        prompt_len = len(str((payload or {}).get("prompt", ""))) if isinstance(payload, dict) else 0
    except Exception:
        payload_keys, prompt_len = [], 0
    return payload_keys, prompt_len


def log_http_error(status_code: Any, ctype: str, body: str, payload: Any) -> None:
    """Emit the completion warning with payload shape (preserved format).

    The body is de-tokenized here because this fires on EVERY non-200, and the
    request that provoked the error is precisely the one carrying the account's
    bearer token -- an upstream that echoes the ``Authorization`` header it was
    given puts it in the log here. The ``%.500s`` that used to bound this was a
    length guard, not a secret guard, so the bound is the scrubber's now and the
    format string no longer needs a length of its own.
    """
    from .engine import scrub_upstream_text

    payload_keys, prompt_len = describe_payload(payload)
    logger.warning("completion HTTP %s (ct=%s) body=%s payload_keys=%s prompt_len=%d",
                   status_code, ctype, scrub_upstream_text(body, 500),
                   payload_keys, prompt_len)


def build_http_error_event(resp: Any, payload: Any,
                           forbidden: dict[int, str] | None = None) -> StreamEvent:
    """Full non-200 mapping: drain body, enrich 422, log, return error event.

    ``forbidden`` forwards the provider-specific 403 mapping to
    :func:`http_error_code` (None = the DeepSeek default). Caller owns
    resp.close()/yield (lifecycle stays in client.py so the client._SESSION
    monkeypatch hook keeps working).
    """
    ctype, body = read_error_body(resp)
    body = enrich_422_body(body)
    code = http_error_code(resp.status_code, forbidden)
    log_http_error(resp.status_code, ctype, body, payload)
    return StreamEvent(kind="error", code=code, message=body)


def loads_lenient(body: str) -> Any:
    """Best-effort json.loads: {} on any error, never raises."""
    try:
        return _jsonlib.loads(body)
    except Exception:
        return {}


def truncate_message(text: str, limit: int = UPSTREAM_MESSAGE_MAX) -> str:
    """Truncate a surfaced upstream body (default 300, preserved)."""
    return (text or "")[:limit]


# --- Shared singletons (F9: one copy, providers alias/delve) -----------------

# Word-ish ban match (shared DeepSeek/Qwen): plain substring "mute"
# false-positives on "commute"/"transmute". Letter boundaries (not \b: "_"
# must count as a boundary so "is_muted"/"mute_until" still hit) keep true
# bans while ignoring "commute". "unbanned" intentionally does NOT match
# (preceded by a letter).
BAN_WORD_RE = _re.compile(
    r"(?<![a-z])(muted?|suspend\w*|bann?ed?|violat\w*)(?![a-z])",
    _re.IGNORECASE,
)


def has_ban_word(text: str) -> bool:
    """True when text carries a ban-family word (never raises)."""
    try:
        return bool(text and BAN_WORD_RE.search(text))
    except Exception:
        return False


def parse_json_response(resp: Any) -> dict:
    """Best-effort ``resp.json()``: {} on any error, never raises.

    Single home for the identical ``parse_json`` helpers in
    deepseek/qwen ``protocol.py`` (same 4 lines). Providers keep their
    ``parse_json`` names as thin aliases so existing imports keep working.
    """
    try:
        return resp.json()
    except Exception:
        return {}


def close_quietly(resp: Any) -> None:
    """Best-effort resp.close(), never raises."""
    try:
        resp.close()
    except Exception:
        pass


def drain_json_text(resp: Any, cap: int = JSON_DRAIN_CAP) -> str:
    """Drain a non-SSE 200 body via iter_lines -> stripped text.

    stream=True leaves .text/.json() empty, so both providers drain the
    same way (cap 20k preserved). Returns "" on any error, never raises.
    """
    try:
        return "".join(drain_lines(resp, cap)).strip()
    except Exception:
        return ""


def iter_text_lines(resp: Any) -> Iterator[str]:
    """Yield decoded non-empty SSE lines (bytes->utf-8/replace)."""
    for raw in resp.iter_lines():
        if not raw:
            continue
        yield decode_line(raw)


def pump_sse(resp: Any, parser: Any,
             precheck: Callable[[str], Any | None] | None = None,
             ) -> Iterator[StreamEvent]:
    """Shared SSE pump: decode lines, optional per-line guard, feed parser.

    Owns ``resp.close()`` via finally (callers no longer close manually).
    ``precheck(line)`` maps one decoded line to an error StreamEvent (or
    None to keep feeding): DeepSeek uses it for inline raw-JSON bans,
    Qwen for WAF-challenge lines. The wire parsers stay distinct; only
    this decode/close scaffolding is shared. Yields until
    finished/error, then returns.
    """
    try:
        for line in iter_text_lines(resp):
            if precheck is not None:
                try:
                    guarded = precheck(line)
                except Exception:
                    guarded = None
                if guarded is not None:
                    yield guarded
                    return
            for event in parser.feed(line):
                yield event
                if event.kind in ("finished", "error"):
                    return
    finally:
        close_quietly(resp)
