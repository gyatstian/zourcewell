"""Opencode local-CLI client: session lifecycle over the v2 HTTP API (sync).

Stdlib ``urllib`` only (localhost HTTP, no TLS fingerprinting needed). v2
removed the blocking message endpoint, so a turn is prompt-then-poll:

``GET /api/info`` (alive) -> ``POST /api/session`` (create) ->
``POST /api/session/:id/model`` + ``/agent`` (v2 made the model, its
reasoning ``variant`` and the agent SESSION-scoped) ->
``POST /api/session/:id/prompt`` (returns at once, the agent loop runs
async) -> poll ``GET /api/session/:id/message`` until this turn's assistant
item carries ``time.completed`` -> best-effort ``DELETE /api/session/:id``.
:func:`stream_message` subscribes to the ``GET /api/event`` bus FIRST so no
early delta is missed, forwards ``session.text.delta`` /
``session.reasoning.delta`` through the callbacks, and then reads the same
authoritative timeline for the final answer.

Success bodies are enveloped as ``{"data": ...}``, failures are a flat
``{"_tag": "...", "message": "..."}`` object. Any transport or upstream
failure surfaces as ``RuntimeError`` carrying a short code prefix
(``OPENCODE_SERVE_DOWN`` / ``OPENCODE_HTTP_<status>`` / ``OPENCODE_UPSTREAM``)
that the engine classifier maps to a terminal. Never raises raw
``URLError``/``KeyError``.

Sessions are pinned to a working directory via ``location.directory`` in the
create body (see ``protocol.session_directory``): the agent's permitted file
tools run there, not in the serve's own cwd. Resolution is
``DEEPSEAPORT_OPENCODE_WORKDIR`` env > process cwd.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request

from . import protocol as P

logger = logging.getLogger("deepseaport.client")

# Per-request HTTP timeout, and in send_message the deadline for the whole
# prompt-then-poll turn (the prompt POST itself returns at once).
DEFAULT_TIMEOUT = 180

# Poll cadence for the turn loop. The first poll happens before any sleep;
# localhost, so a fixed interval is cheaper than a backoff ladder.
_POLL_INTERVAL = 0.5
# Timeline window per poll, newest first (see get_session_messages), so a long
# session keeps the current turn in view.
_POLL_LIMIT = 50


def _auth_header() -> dict[str, str]:
    user, password = P.basic_auth()
    if not password:
        return {}
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _request(method: str, path: str, body: dict | None = None,
             timeout: float = 20,
             session_id: str | None = None) -> tuple[int, dict]:
    """One JSON request against the serve instance -> (status, data).

    ``data`` is ``{}`` when the body is not JSON. Connection-level failures
    raise ``RuntimeError("OPENCODE_SERVE_DOWN: ...")`` with the start hint.
    Client-identity headers ride every request (see protocol.client_headers).
    """
    url = P.base_url() + path
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(_auth_header())
    headers.update(P.client_headers(session_id))
    req = urllib.request.Request(url, data=payload, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            try:
                raw = resp.read().decode("utf-8", "replace")
            except Exception:
                raw = ""
    except urllib.error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
    except Exception as exc:  # URLError, TimeoutError, OSError, ...
        raise RuntimeError(f"OPENCODE_SERVE_DOWN: {exc}. {P.serve_hint()}") from exc
    try:
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return status, data


def _serve_error_message(data: dict, fallback: str) -> str:
    """Pull the human message out of a serve error envelope.

    v2 failures are flat (``{"_tag": "...", "message": "..."}``); older or
    unexpected shapes nest it under ``data`` or carry it in ``error`` as a
    ``{type, message}`` object. Never raises.
    """
    try:
        inner = data.get("data")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        for key in ("message", "error"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict) and val.get("message"):
                return str(val["message"])
    except Exception:
        pass
    return fallback


def _error_detail(raw) -> str:
    """Human text of one v2 error value (string or ``{type, message}``)."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return str(raw.get("message") or raw.get("type")
                   or json.dumps(raw))[:300]
    return ""


def _raise_for_status(status: int, data: dict, action: str,
                      session_id: str) -> None:
    """Raise RuntimeError with the classifier code prefix, or return on 2xx.

    One home for the status mapping shared by the session-scoped calls
    (model/agent select, prompt): 401 auth, 404 unknown session vs missing
    v2 route, 429 rate limit, everything else as ``OPENCODE_HTTP_<status>``.
    """
    if 200 <= status < 300:
        return
    detail = _serve_error_message(data, json.dumps(data)[:300])[:300]
    if status == 401:
        raise RuntimeError(
            "OPENCODE_HTTP_401: serve rejected basic auth "
            "(check DEEPSEAPORT_OPENCODE_USERNAME/PASSWORD "
            "vs OPENCODE_SERVER_PASSWORD).")
    if status == 404:
        if P.is_session_error(detail):
            raise RuntimeError(
                f"OPENCODE_INVALID_SESSION: session {session_id} unknown "
                f"(stale session, recreate): {detail[:200]}")
        raise RuntimeError(
            f"OPENCODE_HTTP_404: {action} route missing at {P.base_url()} "
            f"(opencode v2 /api required, v1 routes are gone): {detail[:200]}")
    if status == 429:
        raise RuntimeError(f"OPENCODE_RATE_LIMITED: {detail}")
    raise RuntimeError(f"OPENCODE_HTTP_{status}: {action} failed: {detail}")


def _raise_turn_error(detail: str, provider_id: str, model_id: str) -> None:
    """One home for a turn the serve refused or failed -> RuntimeError.

    The free-tier gatekeeper is terminal and gets its own prefix; everything
    else is ``OPENCODE_UPSTREAM``. The detail keeps the serve's own words
    ("Variant unavailable for ...", "rate limit exceeded") because
    completion_service.classify_error and its variant-resend check read them.
    """
    detail = str(detail or "")[:500]
    if P.is_free_tier_error(detail):
        raise RuntimeError(
            "OPENCODE_FREE_TIER_DENIED: opencode free tier refused the "
            "request (upstream client-fingerprint check). Upgrade the "
            "`opencode` CLI and retry; key-gated models need "
            "`opencode auth login`. "
            f"Upstream: {detail[:300]}")
    raise RuntimeError(
        f"OPENCODE_UPSTREAM: agent loop failed for {provider_id}/{model_id}: "
        f"{detail}")


def health(timeout: float = 10) -> bool:
    """True when ``GET /api/info`` answers 200 (never raises)."""
    try:
        status, _data = _request("GET", "/api/info", timeout=timeout)
        return status == 200
    except Exception:
        return False


def create_session(timeout: float = 20,
                    directory: str | None = None) -> str:
    """POST /api/session -> session id (raises RuntimeError on failure).

    ``directory`` pins the session worktree (native tools run here) as
    ``location.directory``; None resolves via
    ``protocol.session_directory()`` (WORKDIR env > cwd). The model, variant
    and agent are NOT set here -- v2 can only be born with a model, and
    :func:`send_message` sets all three per turn (see :func:`_select`).
    """
    resolved = directory if directory is not None else P.session_directory()
    body: dict = {"location": {"directory": resolved}} if resolved else {}
    status, data = _request("POST", "/api/session", body, timeout=timeout)
    if status != 200:
        raise RuntimeError(
            f"OPENCODE_HTTP_{status}: create_session failed: "
            f"{_serve_error_message(data, json.dumps(data)[:300])[:300]}. "
            f"{P.serve_hint()}")
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    sid = str(inner.get("id") or "").strip()
    if not sid:
        raise RuntimeError(
            f"OPENCODE_UPSTREAM: create_session returned no id: "
            f"{json.dumps(data)[:300]}")
    return sid


def delete_session(session_id: str | None, timeout: float = 10,
                   directory: str | None = None) -> None:
    """Best-effort DELETE /api/session/:id (204); never raises."""
    if not session_id:
        return
    try:
        _request("DELETE", f"/api/session/{session_id}", timeout=timeout,
                 session_id=session_id)
    except Exception:
        pass


def abort_session(session_id: str | None,
                  directory: str | None = None) -> None:
    """Best-effort POST /api/session/:id/interrupt; never raises."""
    if not session_id:
        return
    try:
        _request("POST", f"/api/session/{session_id}/interrupt", {},
                 timeout=10, session_id=session_id)
    except Exception:
        pass


def get_session_messages(session_id: str, timeout: float = 20,
                         directory: str | None = None,
                         limit: int = _POLL_LIMIT) -> list:
    """Session timeline, OLDEST FIRST; [] on failure, never raises.

    v2 answers ``{"data": [items], "cursor": {...}}``. ``order=desc`` plus a
    limit keeps the newest window (the current turn) in view on a long
    session, so the list is reversed back into chronological order here.
    ``directory`` is accepted and ignored: v2 has no ``?directory=`` query.
    """
    del directory
    try:
        status, data = _request(
            "GET", f"/api/session/{session_id}/message"
                   f"?limit={int(limit)}&order=desc",
            timeout=timeout, session_id=session_id)
        if status != 200:
            return []
        items = data.get("data")
        if not isinstance(items, list):
            return []
        return items[::-1]
    except Exception:
        return []


def _select(session_id: str, provider_id: str, model_id: str, agent: str,
            variant: str | None, timeout: float) -> None:
    """POST the session's model (with ``variant``) and its agent; 204 = ok.

    v2 made all three session-scoped. ``POST /api/session`` is the only place
    a session can be born with a model and :func:`create_session`'s signature
    cannot carry one, so the turn sets them here instead -- without it the
    serve answers with whatever model the session happens to carry (nothing,
    for a session created with an empty body), so a request for
    ``anthropic/claude-x`` would silently come back from another model.

    An unusable model or variant is accepted here and fails the turn with
    ``Model/Variant unavailable for ...`` on the timeline, which is why the
    "variant" word has to survive into the error text:
    completion_service keys its variant-less resend off it.
    """
    model: dict = {"id": str(model_id or ""),
                   "providerID": str(provider_id or "")}
    if variant:
        model["variant"] = str(variant)
    if model["id"]:
        status, data = _request(
            "POST", f"/api/session/{session_id}/model", {"model": model},
            timeout=timeout, session_id=session_id)
        _raise_for_status(status, data, "select model", session_id)
    if agent:
        status, data = _request(
            "POST", f"/api/session/{session_id}/agent", {"agent": str(agent)},
            timeout=timeout, session_id=session_id)
        _raise_for_status(status, data, "select agent", session_id)


def _turn_outcome(items: list, user_id: str, provider_id: str,
                  model_id: str) -> tuple[str, object]:
    """State of the turn that starts at ``user_id`` -> (outcome, payload).

    ``("done", result_dict)`` once its assistant item carries
    ``time.completed``, ``("error", detail)`` on a failed turn, else
    ``("pending", "")``. Only items AFTER the user item count, so a resend on
    the same session (the variant-less retry) can never read the previous
    turn's answer or its error. ``user_id`` is the id the prompt POST
    returned; when it is not in the window the whole window is used, which
    still resolves to the newest completed assistant item.
    """
    start = 0
    if user_id:
        for idx, item in enumerate(items):
            if isinstance(item, dict) and str(item.get("id") or "") == user_id:
                start = idx + 1
                break
    tail = [i for i in items[start:] if isinstance(i, dict)]
    failed = ""
    for item in tail:
        detail = _error_detail(item.get("error"))
        if detail:
            failed = detail
    for item in reversed(tail):
        if str(item.get("type") or "") != "assistant":
            continue
        time_info = item.get("time")
        if isinstance(time_info, dict) and time_info.get("completed"):
            return "done", _parse_message_result(item, provider_id, model_id)
    if failed:
        return "error", failed
    return "pending", ""


def _parse_message_result(item: dict, provider_id: str,
                          model_id: str) -> dict:
    """One v2 assistant item -> {content, thinking, usage_total, finish}.

    ``item`` is the timeline entry itself (``content[]`` parts, ``tokens``,
    ``finish``, ``error``). Raises RuntimeError with the classifier code
    prefix on error/empty, never KeyError.
    """
    parts = item.get("content")
    if not isinstance(parts, list):
        parts = []
    content = "".join(str(p.get("text") or "")
                      for p in parts
                      if isinstance(p, dict) and p.get("type") == "text")
    thinking = "".join(str(p.get("text") or "")
                       for p in parts
                       if isinstance(p, dict) and p.get("type") == "reasoning")
    usage_total = 0
    try:
        tokens = item.get("tokens")
        if isinstance(tokens, dict):
            usage_total = int(tokens.get("output") or 0)
    except (TypeError, ValueError):
        usage_total = 0
    finish = str(item.get("finish") or "stop")
    detail = _error_detail(item.get("error"))
    if detail:
        _raise_turn_error(detail, provider_id, model_id)
    if not content and not thinking:
        raise RuntimeError(
            f"OPENCODE_EMPTY: empty completion for {provider_id}/{model_id} "
            "(transient contentless turn; retry on a fresh session)")
    return {"content": content, "thinking": thinking,
            "usage_total": usage_total, "finish": finish}


def prompt_async(session_id: str, provider_id: str, model_id: str,
                 agent: str, prompt: str,
                 timeout: float = 30,
                 variant: str | None = None) -> str:
    """Select model+agent, POST the prompt; returns this turn's user item id.

    v2 has no separate async endpoint: ``POST /api/session/:id/prompt``
    enqueues the turn and answers at once with the created user message, so
    the caller polls the timeline for the reply. Raises RuntimeError (with
    the classifier code prefix) when the prompt itself is refused.
    """
    _select(session_id, provider_id, model_id, agent, variant, timeout)
    status, data = _request(
        "POST", f"/api/session/{session_id}/prompt",
        P.message_payload(prompt),
        timeout=timeout, session_id=session_id)
    _raise_for_status(status, data, "prompt", session_id)
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    return str(inner.get("id") or "")


def _await_turn(session_id: str, user_id: str, provider_id: str,
                model_id: str, timeout: float) -> dict:
    """Poll the timeline until this turn finishes, fails or times out."""
    deadline = time.monotonic() + max(5.0, float(timeout))
    while True:
        items = get_session_messages(session_id, timeout=20)
        outcome, payload = _turn_outcome(items, user_id, provider_id, model_id)
        if outcome == "done":
            return payload
        if outcome == "error":
            _raise_turn_error(str(payload), provider_id, model_id)
        if time.monotonic() >= deadline:
            logger.warning("opencode turn produced no assistant message "
                           "within %.0fs for %s/%s",
                           float(timeout), provider_id, model_id)
            # "timed out" on purpose: classify_error reads it as a retryable
            # overload, so a slow-but-healthy serve gets a fresh attempt.
            raise RuntimeError(
                f"OPENCODE_UPSTREAM: turn timed out after "
                f"{float(timeout):.0f}s for {provider_id}/{model_id} "
                f"(session {session_id})")
        time.sleep(_POLL_INTERVAL)


def send_message(session_id: str, provider_id: str, model_id: str,
                 agent: str, prompt: str,
                 timeout: float = DEFAULT_TIMEOUT,
                 variant: str | None = None) -> dict:
    """Run one turn; return {content, thinking, usage_total, finish}.

    Prompt, then poll: v2 returns from the prompt at once and runs the agent
    loop in the background, so the authoritative reply is the newest
    assistant item carrying ``time.completed``. Raises RuntimeError on
    transport/upstream failure (the message carries a code prefix for the
    classifier) and ``OPENCODE_EMPTY`` on a contentless reply, so the engine
    retries once on a fresh session.
    """
    user_id = prompt_async(
        session_id, provider_id, model_id, agent, prompt,
        timeout=min(30.0, max(5.0, float(timeout))),
        variant=variant)
    return _await_turn(session_id, user_id, provider_id, model_id, timeout)


def stream_message(session_id: str, provider_id: str, model_id: str,
                   agent: str, prompt: str,
                   timeout: float = DEFAULT_TIMEOUT,
                   variant: str | None = None,
                   on_content=None, on_thinking=None,
                   cancel_event=None) -> dict:
    """/api/event SSE tail + prompt -> final result dict (live deltas).

    Subscribes to ``GET /api/event`` first (so early deltas are not missed),
    selects the session's model+agent, sends the prompt, forwards
    ``session.text.delta`` / ``session.reasoning.delta`` through the callbacks
    as they arrive, and waits for this session's
    ``session.execution.succeeded`` (or ``session.execution.failed``). The
    returned value is the authoritative final read from the timeline, never
    the assembled tail, so a partial stream can never be served as the
    answer. Raises RuntimeError on transport/upstream failure with the
    classifier code prefix; partial text only ever rides the callbacks.
    """

    def _cancelled() -> bool:
        try:
            return bool(cancel_event is not None and cancel_event.is_set())
        except Exception:
            return False

    # --- SSE reader state (writer thread -> callbacks) ---------------------
    _connected = threading.Event()
    _idle = threading.Event()
    _sse_error: list[str] = []
    _stop_reader = threading.Event()

    def _emit_content(text: str) -> None:
        if not text or on_content is None:
            return
        try:
            on_content(text)
        except Exception:
            pass

    def _emit_thinking(text: str) -> None:
        if not text or on_thinking is None:
            return
        try:
            on_thinking(text)
        except Exception:
            pass

    def _handle_event(payload: dict) -> None:
        """One ``data: {...}`` frame; payload fields live under ``data``."""
        try:
            etype = str(payload.get("type") or "")
            props = payload.get("data")
            if not isinstance(props, dict):
                return
            if str(props.get("sessionID") or "") != session_id:
                return
            if etype == "session.reasoning.delta":
                _emit_thinking(str(props.get("delta") or ""))
            elif etype == "session.text.delta":
                _emit_content(str(props.get("delta") or ""))
            elif etype == "session.execution.succeeded":
                _idle.set()
            elif etype in ("session.execution.failed", "session.error"):
                # Stop waiting on the first failure; the timeline carries the
                # same error and the final read below turns it into the raise.
                detail = _error_detail(props.get("error")) or json.dumps(props)
                _sse_error.append(detail[:500])
                _idle.set()
        except Exception:
            pass

    def _read_sse() -> None:
        try:
            url = P.base_url() + "/api/event"
            headers = {"Accept": "text/event-stream"}
            headers.update(_auth_header())
            headers.update(P.client_headers(session_id))
            req = urllib.request.Request(url, method="GET", headers=headers)
            # Short socket timeout so cancel/idle checks stay responsive;
            # the serve sends ": heartbeat" comments, so silence is expected.
            with urllib.request.urlopen(req, timeout=10) as resp:
                _connected.set()
                buf = b""
                data_lines: list[str] = []
                while not _stop_reader.is_set() and not _idle.is_set():
                    if _cancelled():
                        break
                    try:
                        chunk = resp.read(1)
                    except Exception:
                        # Socket timeout: re-check stop conditions, keep going.
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    if buf.endswith(b"\n"):
                        try:
                            line = buf.decode("utf-8", "replace").rstrip("\r\n")
                        except Exception:
                            line = ""
                        buf = b""
                        if not line:
                            if data_lines:
                                raw = "\n".join(data_lines)
                                data_lines = []
                                # Frames are `data: {...}`; ignore comments.
                                if raw.startswith("data:"):
                                    raw = raw[5:].strip()
                                try:
                                    _handle_event(json.loads(raw))
                                except Exception:
                                    pass
                        elif line.startswith("data:"):
                            data_lines.append(line)
                        elif line.startswith(":"):
                            continue
                        # `event:` lines carry no payload; ignore.
        except Exception as exc:
            if not _connected.is_set():
                _sse_error.append(f"SSE_CONNECT: {exc}"[:300])
        finally:
            _connected.set()

    reader = threading.Thread(target=_read_sse, daemon=True)
    reader.start()
    # Wait for the SSE subscription before admitting the prompt; otherwise
    # the first deltas race the subscription and streaming starts mid-reply.
    _connected.wait(timeout=10.0)
    try:
        user_id = prompt_async(session_id, provider_id, model_id, agent,
                               prompt,
                               timeout=min(30.0, max(5.0, float(timeout))),
                               variant=variant)
    except Exception:
        _stop_reader.set()
        raise

    deadline = time.monotonic() + max(10.0, float(timeout) + 30.0)
    while time.monotonic() < deadline:
        if _cancelled():
            try:
                abort_session(session_id)
            except Exception:
                pass
            _stop_reader.set()
            try:
                from ...server_concurrency import _StreamCancelled as _Cancelled
            except Exception:  # pragma: no cover
                class _Cancelled(Exception):  # type: ignore[no-redef]
                    pass

            raise _Cancelled()
        if _idle.is_set():
            break
        time.sleep(0.05)
    _stop_reader.set()
    try:
        reader.join(timeout=5.0)
    except Exception:
        pass

    if _cancelled():
        try:
            from ...server_concurrency import _StreamCancelled as _Cancelled2
        except Exception:  # pragma: no cover
            class _Cancelled2(Exception):  # type: ignore[no-redef]
                pass

        raise _Cancelled2()
    # Authoritative final: this turn's completed assistant item (covers
    # multi-step tool loops; deltas already streamed live via callbacks).
    items = get_session_messages(session_id, timeout=20)
    outcome, payload = _turn_outcome(items, user_id, provider_id, model_id)
    if outcome == "done":
        return payload
    if outcome == "error":
        _raise_turn_error(str(payload), provider_id, model_id)
    if _sse_error and not _idle.is_set():
        # SSE-level failure without a usable message: surface for retry.
        raise RuntimeError(f"OPENCODE_UPSTREAM: live stream failed: "
                           f"{_sse_error[0][:300]}")
    raise RuntimeError(
        f"OPENCODE_EMPTY: empty completion for {provider_id}/{model_id} "
        "(live stream ended without an assistant message)")