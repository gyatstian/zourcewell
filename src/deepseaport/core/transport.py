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
- The bounded read for the SSE pump (``_cancellable`` / ``_BoundedQueue`` /
  ``_abandon`` / ``_ClientGone``) is here because BOTH web2api providers need
  it. It changes the GRANULARITY of cancellation only, never the socket
  budget, and it reads the cancel token late -- see ``_cancellable``.
"""

from __future__ import annotations

import json as _jsonlib
import logging
import queue
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


def truncate_message(text: str) -> str:
    """Truncate a surfaced upstream body to ``UPSTREAM_MESSAGE_MAX`` (300)."""
    return (text or "")[:UPSTREAM_MESSAGE_MAX]


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


# --- The bounded read every SSE pump needs -----------------------------------
#
# curl_cffi offers NO per-read timeout -- ``iter_lines(chunk_size,
# decode_unicode, delimiter)`` has no ``timeout`` parameter (0.16.3) and
# ``iter_content`` blocks on an untimed ``queue.get`` -- so the response's queue
# is the only place a read parked on a silent socket can hand control back.
# Shared by both web2api providers: the stream function sits inside a BLOCKING
# read, and ``drain_attempt``'s per-event cancel check cannot run until an event
# is yielded out of the generator.

# How long one bounded read waits for the next socket chunk before handing
# control back. Not a timeout: an empty wait is re-entered and the TOTAL budget
# stays the provider's DEFAULT_TIMEOUT. It only decides how often cancellation
# is noticed.
_READ_SLICE_SECONDS = 2.0


class _ClientGone(Exception):
    """Internal: the client disconnected while we were blocked on a read."""


def _abandon(resp) -> None:
    """Stop the transfer without waiting for curl to notice.

    ``Response.close()`` joins curl's worker thread, and curl only re-checks
    ``quit_now`` from inside its WRITE callback -- verified against the
    installed curl_cffi 0.16.3: with the upstream still streaming, ``quit_now``
    ends the transfer on the next delta (close returned in 0.05s); with the
    upstream silent it is never called again and the join blocked for the rest
    of DEFAULT_TIMEOUT. A cancelled reply is discarded anyway, so set the flag
    and let go: curl ends the transfer on the next byte or on its own stall
    timer, in a thread curl_cffi owns, and the account slot comes back now.
    ``Curl.__del__`` reclaims the handle, so nothing is leaked.
    """
    if getattr(resp, "quit_now", None) is not None:
        resp.quit_now.set()

    def _close() -> None:
        pass

    resp.close = _close


class _BoundedQueue:
    """``Response.queue`` proxy whose ``get`` returns control every slice.

    curl_cffi offers no per-read timeout -- ``iter_lines`` takes no ``timeout``
    (0.16: ``chunk_size`` / ``decode_unicode`` / ``delimiter``) and
    ``iter_content`` blocks on an untimed ``queue.get`` -- so the queue is the
    only place a blocked read can be handed control back. The stall budget is
    NOT shortened: curl still aborts at its own timeout and puts its error in
    the queue (``iter_content`` re-raises it). Only the granularity changes, so
    a disconnected client gives its account slot back within a slice instead of
    after the whole budget.
    """

    def __init__(self, raw, cancel_event, resp) -> None:
        self._raw = raw
        self._cancel = cancel_event
        self._resp = resp
        self._fired = False

    def get(self, *a, **k):
        while True:
            try:
                return self._raw.get(timeout=_READ_SLICE_SECONDS)
            except queue.Empty:
                if self._cancel is not None and self._cancel.is_set():
                    if not self._fired:
                        self._fired = True
                        _abandon(self._resp)
                    raise _ClientGone from None

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _cancellable(resp) -> None:
    """Bound ``resp``'s blocking read to the cancel token in force, if any.

    No token (a direct/unit-test call) or no live stream queue (a test double, a
    non-stream response) means nothing to observe, so ``resp`` is left exactly
    as it was and nothing else about the pump changes.
    """
    from .engine import current_cancel_event

    cancel = current_cancel_event()
    q = getattr(resp, "queue", None)
    if cancel is None or q is None:
        return
    resp.queue = _BoundedQueue(q, cancel, resp)
