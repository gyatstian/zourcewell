"""DeepSeek web client: session, PoW, SSE completion (sync, thread-safe)."""

from __future__ import annotations

import logging
import queue
import time
from typing import Iterator

from curl_cffi import requests as crequests

from . import pow as PoW
from . import protocol as P
from ...core import engine as _engine
from ...core import transport as T
from .sse_parser import StreamParser

logger = logging.getLogger("deepseaport.client")

# Backwards-compatible alias: the fingerprint lives in protocol as the single
# source shared with the account pool.
IMPERSONATE = P.IMPERSONATE
# Per-request stream timeout. curl_cffi treats this as a low-speed/stall
# timeout for streaming (read timeout), not a hard wall-clock cap, but keep it
# large enough for long reasoning generations. The server's consumer wait is
# derived from this value so the two cannot drift apart.
DEFAULT_TIMEOUT = 300

# How long one bounded read waits for the next socket chunk before handing
# control back. Not a timeout: an empty wait is re-entered and the TOTAL budget
# stays DEFAULT_TIMEOUT. It only decides how often cancellation is noticed, and
# `iter_lines(timeout=...)` does not exist in curl_cffi 0.16 (`iter_lines(
# chunk_size=None, decode_unicode=False, delimiter=None)`, `iter_content`
# blocking on an untimed `queue.get`), so a per-read timeout is not available
# and the read loop has to be bounded here.
_READ_SLICE_SECONDS = 2.0

# Shared session: reuse HTTP/2 connections + TLS across calls. curl_cffi
# thread-safe: one curl handle per thread (thread-local), so concurrent
# callers get isolation while each thread reuses its connection. The cookie
# jar is discarded: the Cookie header is set manually per request and WAF
# Set-Cookie responses must not mutate a shared jar.
_SESSION = crequests.Session(impersonate=IMPERSONATE, discard_cookies=True)


def _json(resp) -> dict:
    return P.parse_json(resp)


def _biz_error(data: dict) -> tuple[str, str] | None:
    """Return (code, msg) when the envelope carries a biz error.

    Ban/mute envelopes (biz_code 5, "user is muted") embed
    biz_data {is_muted, mute_until}: the expiry is appended so callers
    surface "suspended until 16 September 2026 12:49" instead of a bare code.
    """
    if not isinstance(data, dict):
        return None
    inner = data.get("data")
    code = data.get("code")
    if isinstance(inner, dict):
        code = inner.get("biz_code", code)
        msg = inner.get("biz_msg", data.get("msg", ""))
    else:
        msg = data.get("msg", "")
    if code in (0, "0", None, ""):
        return None
    code_s, msg_s = str(code), str(msg or "")
    # Attach ban expiry when present (completion ban carries it here).
    msg_s = P.with_ban_expiry(msg_s, data)
    return code_s, msg_s


def fetch_current_user(headers: dict, timeout: int = 20) -> dict:
    """GET users/current. Raw envelope; caller parses ban/invalid itself."""
    resp = _SESSION.get(P.USERS_CURRENT_URL, headers=headers,
                        impersonate=IMPERSONATE, timeout=timeout)
    return _json(resp)


def check_ban(headers: dict, timeout: int = 20) -> tuple[bool, float | None]:
    """Return (is_banned, mute_until). Invalid token (40003) -> (False, None).

    Uses users/current chat.{is_muted,mute_until}. Best-effort: network or
    shape failures return (False, None), never raise.
    """
    try:
        data = fetch_current_user(headers, timeout=timeout)
        if not isinstance(data, dict):
            return False, None
        # Invalid/expired token is not a ban — just unauthenticated.
        if P.is_invalid_token_payload(data):
            return False, None
        # Single verdict: detect_ban pairs is_ban_payload + timestamp, so an
        # explicit non-muted chat shape also lands here as (False, None).
        return P.detect_ban(data)
    except Exception as exc:
        logger.debug("check_ban ignored: %s", exc)
        return False, None


def create_session(headers: dict) -> str:
    resp = _SESSION.post(P.SESSION_URL, headers=headers, json={"agent": "chat"},
                         impersonate=IMPERSONATE, timeout=30)
    data = _json(resp)
    err = _biz_error(data)
    if err:
        raise RuntimeError(f"create_session biz error {err[0]}: {err[1]}")
    inner = (data.get("data") or {}).get("biz_data", {}) if isinstance(data.get("data"), dict) else {}
    session_id = inner.get("id") or (inner.get("chat_session") or {}).get("id")
    if not session_id:
        raise RuntimeError(f"create_session: no id (HTTP {resp.status_code}): {resp.text[:200]}")
    return str(session_id)


def delete_session(headers: dict, session_id: str) -> None:
    try:
        _SESSION.post(P.DELETE_SESSION_URL, headers=headers,
                      json={"chat_session_id": session_id},
                      impersonate=IMPERSONATE, timeout=10)
    except Exception as exc:
        logger.debug("delete_session ignored: %s", exc)


def fetch_pow(headers: dict) -> dict:
    resp = _SESSION.post(P.POW_URL, headers=headers,
                         json={"target_path": P.COMPLETION_PATH},
                         impersonate=IMPERSONATE, timeout=30)
    data = _json(resp)
    err = _biz_error(data)
    if err:
        raise RuntimeError(f"pow challenge biz error {err[0]}: {err[1]}")
    inner = (data.get("data") or {}).get("biz_data", {}) if isinstance(data.get("data"), dict) else {}
    challenge = inner.get("challenge", inner)
    if not isinstance(challenge, dict) or "challenge" not in challenge:
        raise RuntimeError(f"pow challenge: bad shape: {resp.text[:200]}")
    return challenge


def solve_challenge(challenge: dict) -> str:
    """Solve an already-fetched PoW challenge; return the X-DS-PoW-Response value."""
    t0 = time.time()
    answer = PoW.solve(
        challenge.get("algorithm", PoW.ALGORITHM),
        challenge["challenge"], challenge["salt"],
        challenge.get("difficulty", 144000), challenge.get("expire_at", 0),
    )
    if answer is None:
        raise RuntimeError("PoW solver found no answer")
    challenge["answer"] = answer
    logger.debug("pow solved in %.1fs (difficulty %s)", time.time() - t0, challenge.get("difficulty"))
    return PoW.build_header(challenge)


def solve_pow(headers: dict) -> str:
    return solve_challenge(fetch_pow(headers))


def _ban_stream_event(data) -> P.StreamEvent | None:
    """Map a completion envelope to an error StreamEvent, else None.

    Handles the repeated biz-error / ban-payload sequence:
      - biz error present -> error event (ban expiry attached by _biz_error)
      - ban payload but no biz code -> explicit ban event (code "5")
    """
    if not isinstance(data, dict):
        return None
    banned, ts = P.detect_ban(data)
    err = _biz_error(data)
    if err:
        return P.StreamEvent(kind="error", code=err[0], message=err[1],
                             ban_until=ts if banned else None)
    if banned:
        return P.StreamEvent(kind="error", code="5",
                             message=P.ban_message(ts), ban_until=ts)
    return None


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
    NOT shortened: curl still aborts at DEFAULT_TIMEOUT and puts its error in
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
    cancel = _engine.current_cancel_event()
    q = getattr(resp, "queue", None)
    if cancel is None or q is None:
        return
    resp.queue = _BoundedQueue(q, cancel, resp)


def _raw_json_ban_precheck(line: str) -> P.StreamEvent | None:
    """Inline raw-JSON ban guard for the SSE pump (DeepSeek envelope only).

    Raw JSON bans can arrive inside an event-stream body (no ``data:``
    prefix). Returns the ban/error event to stop the pump, else None to
    keep feeding the wire parser (which stays DeepSeek-specific).
    """
    stripped = (line or "").strip()
    if not stripped.startswith("data:") and stripped.startswith("{"):
        data3 = T.loads_lenient(stripped)
        if isinstance(data3, dict):
            event3 = _ban_stream_event(data3)
            if event3 is not None:
                return event3
    return None


def stream_completion(headers: dict, payload: dict,
                      timeout: int = DEFAULT_TIMEOUT) -> Iterator[P.StreamEvent]:
    """POST completion and yield parsed events. Caller must close on break."""
    resp = _SESSION.post(P.COMPLETION_URL, headers=headers, json=payload,
                         impersonate=IMPERSONATE, timeout=timeout, stream=True)
    if resp.status_code == 200 and not T.is_sse_response(resp):
        # Some errors arrive as JSON with HTTP 200 (e.g. ban:
        # biz_code 5 "user is muted" + mute_until). With stream=True the body
        # is not buffered, so resp.json() is empty — drain via iter_lines.
        data = _json(resp)
        event = _ban_stream_event(data) if data else None
        if event is not None:
            T.close_quietly(resp)
            yield event
            return
        # Streaming mode: body arrives as raw JSON lines, not SSE "data:".
        try:
            body = T.drain_json_text(resp, T.JSON_DRAIN_CAP)
            T.close_quietly(resp)
            if body:
                data2 = T.loads_lenient(body)
                event2 = _ban_stream_event(data2) if isinstance(data2, dict) else None
                if event2 is not None:
                    yield event2
                    return
                yield P.StreamEvent(kind="error", code="UPSTREAM_ERROR",
                                    message=T.truncate_message(body))
                return
        except StopIteration:
            pass
        except Exception:
            pass
        T.close_quietly(resp)
        # The drain above consumed the only body this response has: the SSE
        # loop would iterate a drained, closed curl_cffi stream a SECOND time,
        # which never yields EOF and never raises -- the worker parked forever
        # and its account slot was never released (reproduced against
        # curl_cffi 0.16.3).
        yield P.StreamEvent(kind="error", code="UPSTREAM_ERROR",
                            message="empty non-SSE 200 body")
        return
    if resp.status_code != 200:
        event = T.build_http_error_event(resp, payload)
        T.close_quietly(resp)
        yield event
        return
    # Shared decode/close scaffolding; wire parser + ban precheck stay
    # DeepSeek-specific (Qwen uses its own parser + WAF guard). Bounding the
    # read is done here, on the only read that can park for the whole budget.
    _cancellable(resp)
    try:
        yield from T.pump_sse(resp, StreamParser(), precheck=_raw_json_ban_precheck)
    except _ClientGone:
        # Client left mid-read: stop cleanly so the caller releases the account
        # slot. The stream pump's own finally already closed the response, and
        # _abandon replaced that close with the non-blocking one.
        return
