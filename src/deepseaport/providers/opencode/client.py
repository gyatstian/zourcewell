"""Opencode local-CLI client: session lifecycle over plain HTTP (sync).

Stdlib ``urllib`` only (localhost HTTP, no TLS fingerprinting needed):
``health`` -> ``create_session`` -> ``send_message`` (blocking: serve waits
for the agent loop) -> best-effort ``delete_session``. Any transport or
upstream failure surfaces as ``RuntimeError`` carrying a short code prefix
(``OPENCODE_SERVE_DOWN`` / ``OPENCODE_HTTP_<status>``) that the engine
classifier maps to a terminal. Never raises raw ``URLError``/``KeyError``.

Sessions are pinned to a working directory via ``?directory=`` (see
``protocol.session_directory``): native tools (bash/read) run there, not in
the serve's own cwd. Resolution is ``DEEPSEAPORT_OPENCODE_WORKDIR`` env >
process cwd; the serve registers the project on demand.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request

from . import protocol as P

logger = logging.getLogger("deepseaport.client")

# Per-request stream/HTTP timeout for the blocking message POST. The server's
# consumer wait is derived from this so the two cannot drift apart.
DEFAULT_TIMEOUT = 180

# Sentinel: caller did not choose a native tool allowlist -> use the
# gatekeeper-minimal bash+read pair (historical behaviour). Pass an explicit
# None to omit the `tools` key entirely (plain chat; live-verified on
# opencode 1.18.32: tool-less passes via agent defaults, only an explicit
# all-false allowlist triggers the free-tier FreeTierError envelope).
_DEFAULT_TOOLS: object = object()


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

    Failures look like ``{"name":"UnknownError","data":{"message":...}}``;
    fall back to the raw body when the shape is unknown. Never raises.
    """
    try:
        inner = data.get("data")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        for key in ("message", "error"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
    except Exception:
        pass
    return fallback


def health(timeout: float = 10) -> bool:
    """True when the serve instance answers healthy (never raises)."""
    try:
        status, data = _request("GET", "/global/health", timeout=timeout)
        return status == 200 and bool(data.get("healthy", True))
    except Exception:
        return False


def create_session(timeout: float = 20,
                     directory: str | None = None) -> str:
    """POST /session {} -> session id (raises RuntimeError on failure).

    ``directory`` pins the session worktree (native tools run here);
    None resolves via ``protocol.session_directory()`` (WORKDIR env > cwd).
    """
    query = P.session_query(
        directory if directory is not None else P.session_directory())
    status, data = _request("POST", f"/session{query}", {}, timeout=timeout)
    if status != 200:
        raise RuntimeError(
            f"OPENCODE_HTTP_{status}: create_session failed: "
            f"{_serve_error_message(data, json.dumps(data)[:300])[:300]}. "
            f"{P.serve_hint()}")
    sid = str(data.get("id") or "").strip()
    if not sid:
        raise RuntimeError(
            f"OPENCODE_UPSTREAM: create_session returned no id: "
            f"{json.dumps(data)[:300]}")
    return sid


def send_message(session_id: str, provider_id: str, model_id: str,
                 agent: str, prompt: str,
                 timeout: float = DEFAULT_TIMEOUT,
                 variant: str | None = None,
                 directory: str | None = None,
                 tools: dict | None | object = _DEFAULT_TOOLS) -> dict:
    """POST one message; return {content, thinking, usage_total, finish}.

    Raises RuntimeError on transport/upstream failure (the message carries a
    code prefix for the classifier). Empty agent replies surface as
    ``OPENCODE_EMPTY`` so the engine retries once on a fresh session.
    ``tools`` selects the per-message native allowlist: the default keeps
    the gatekeeper-minimal bash+read pair (everything else hidden so
    pi-harness tools with the same names, via the DSML prompt, stay
    unambiguous); an explicit None omits the key (plain chat, no pi tools).
    """
    body = P.message_payload(
        provider_id, model_id, agent, prompt,
        variant=variant,
        tools=(P.native_tools_minimal() if tools is _DEFAULT_TOOLS
               else (dict(tools) if isinstance(tools, dict) else None)))
    query = P.session_query(
        directory if directory is not None else P.session_directory())
    status, data = _request(
        "POST", f"/session/{session_id}/message{query}", body, timeout=timeout,
        session_id=session_id)
    if status == 401:
        raise RuntimeError(
            "OPENCODE_HTTP_401: serve rejected basic auth "
            "(check DEEPSEAPORT_OPENCODE_USERNAME/PASSWORD "
            "vs OPENCODE_SERVER_PASSWORD).")
    if status == 404:
        raise RuntimeError(
            f"OPENCODE_INVALID_SESSION: session {session_id} unknown "
            f"(stale session, recreate): {json.dumps(data)[:200]}")
    if status == 429:
        raise RuntimeError(
            f"OPENCODE_RATE_LIMITED: {json.dumps(data)[:300]}")
    if status != 200:
        raise RuntimeError(
            f"OPENCODE_HTTP_{status}: message failed: "
            f"{_serve_error_message(data, json.dumps(data)[:500])[:500]}")
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    # 200 with an error envelope (flaky agent loop): fail loudly instead of
    # serving the error text as a normal reply.
    info_error = info.get("error") if isinstance(info, dict) else None
    parts = data.get("parts")
    if not isinstance(parts, list):
        parts = []
    content = "".join(str(p.get("text") or "")
                      for p in parts
                      if isinstance(p, dict) and p.get("type") == "text")
    thinking = "".join(str(p.get("text") or "")
                       for p in parts
                       if isinstance(p, dict) and p.get("type") in ("reasoning",))
    usage_total = 0
    try:
        tokens = info.get("tokens") if isinstance(info, dict) else None
        if isinstance(tokens, dict):
            usage_total = int(tokens.get("output") or 0)
    except (TypeError, ValueError):
        usage_total = 0
    finish = str((info.get("finish") if isinstance(info, dict) else "") or "stop")
    if info_error:
        detail = info_error if isinstance(info_error, str) else json.dumps(info_error)
        if P.is_free_tier_error(detail):
            raise RuntimeError(
                "OPENCODE_FREE_TIER_DENIED: opencode free tier refused the "
                "request (upstream client-fingerprint check). Upgrade the "
                "`opencode` CLI and retry; key-gated models need "
                "`opencode auth login`. "
                f"Upstream: {detail[:300]}")
        raise RuntimeError(
            f"OPENCODE_UPSTREAM: agent loop failed for {provider_id}/{model_id}: "
            f"{detail[:500]}")
    if not content and not thinking:
        raise RuntimeError(
            f"OPENCODE_EMPTY: empty completion for {provider_id}/{model_id} "
            "(transient contentless turn; retry on a fresh session)")
    return {"content": content, "thinking": thinking,
            "usage_total": usage_total, "finish": finish}


def delete_session(session_id: str | None, timeout: float = 10,
                     directory: str | None = None) -> None:
    """Best-effort DELETE /session/:id; never raises."""
    if not session_id:
        return
    try:
        query = P.session_query(
            directory if directory is not None else P.session_directory())
        _request("DELETE", f"/session/{session_id}{query}", timeout=timeout,
                 session_id=session_id)
    except Exception:
        pass


def _parse_message_result(data: dict, provider_id: str,
                          model_id: str) -> dict:
    """Shared blocking-result parser -> {content, thinking, usage_total, finish}.

    Raises RuntimeError with the classifier code prefix on error/empty
    (same contract as :func:`send_message`). Never raises KeyError.
    """
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    info_error = info.get("error") if isinstance(info, dict) else None
    parts = data.get("parts")
    if not isinstance(parts, list):
        parts = []
    content = "".join(str(p.get("text") or "")
                      for p in parts
                      if isinstance(p, dict) and p.get("type") == "text")
    thinking = "".join(str(p.get("text") or "")
                       for p in parts
                       if isinstance(p, dict) and p.get("type") in ("reasoning",))
    usage_total = 0
    try:
        tokens = info.get("tokens") if isinstance(info, dict) else None
        if isinstance(tokens, dict):
            usage_total = int(tokens.get("output") or 0)
    except (TypeError, ValueError):
        usage_total = 0
    finish = str((info.get("finish") if isinstance(info, dict) else "") or "stop")
    if info_error:
        detail = info_error if isinstance(info_error, str) else json.dumps(info_error)
        if P.is_free_tier_error(detail):
            raise RuntimeError(
                "OPENCODE_FREE_TIER_DENIED: opencode free tier refused the "
                "request (upstream client-fingerprint check). Upgrade the "
                "`opencode` CLI and retry; key-gated models need "
                "`opencode auth login`. "
                f"Upstream: {detail[:300]}")
        raise RuntimeError(
            f"OPENCODE_UPSTREAM: agent loop failed for {provider_id}/{model_id}: "
            f"{detail[:500]}")
    if not content and not thinking:
        raise RuntimeError(
            f"OPENCODE_EMPTY: empty completion for {provider_id}/{model_id} "
            "(transient contentless turn; retry on a fresh session)")
    return {"content": content, "thinking": thinking,
            "usage_total": usage_total, "finish": finish}


def _session_message_path(session_id: str, directory: str,
                          extra: str = "") -> str:
    """`/session/:id/message` path with directory + extra query merged."""
    base_q = P.session_query(directory)
    if not extra:
        return f"/session/{session_id}/message{base_q}"
    sep = "&" if base_q else "?"
    return f"/session/{session_id}/message{base_q}{sep}{extra.lstrip('?&')}"


def prompt_async(session_id: str, provider_id: str, model_id: str,
                 agent: str, prompt: str,
                 timeout: float = 30,
                 variant: str | None = None,
                 directory: str | None = None,
                 tools: dict | None | object = _DEFAULT_TOOLS) -> None:
    """POST prompt_async (204 on accept); raises RuntimeError otherwise.

    ``tools`` mirrors :func:`send_message` (default minimal, None omits).
    """
    body = P.message_payload(
        provider_id, model_id, agent, prompt,
        variant=variant,
        tools=(P.native_tools_minimal() if tools is _DEFAULT_TOOLS
               else (dict(tools) if isinstance(tools, dict) else None)))
    resolved = directory if directory is not None else P.session_directory()
    query = P.session_query(resolved)
    status, data = _request(
        "POST", f"/session/{session_id}/prompt_async{query}", body,
        timeout=timeout, session_id=session_id)
    if status == 204:
        return
    if status == 401:
        raise RuntimeError(
            "OPENCODE_HTTP_401: serve rejected basic auth "
            "(check DEEPSEAPORT_OPENCODE_USERNAME/PASSWORD "
            "vs OPENCODE_SERVER_PASSWORD).")
    if status == 404:
        raise RuntimeError(
            f"OPENCODE_INVALID_SESSION: session {session_id} unknown "
            f"(stale session, recreate): {json.dumps(data)[:200]}")
    if status == 429:
        raise RuntimeError(
            f"OPENCODE_RATE_LIMITED: {json.dumps(data)[:300]}")
    raise RuntimeError(
        f"OPENCODE_HTTP_{status}: prompt_async failed: "
        f"{_serve_error_message(data, json.dumps(data)[:500])[:500]}")


def abort_session(session_id: str | None,
                  directory: str | None = None) -> None:
    """Best-effort POST /session/:id/abort; never raises."""
    if not session_id:
        return
    try:
        resolved = directory if directory is not None else P.session_directory()
        query = P.session_query(resolved)
        _request("POST", f"/session/{session_id}/abort{query}", {},
                 timeout=10, session_id=session_id)
    except Exception:
        pass


def get_session_messages(session_id: str, timeout: float = 20,
                         directory: str | None = None,
                         limit: int = 20) -> list:
    """GET session messages (newest last); [] on failure, never raises."""
    try:
        resolved = directory if directory is not None else P.session_directory()
        path = _session_message_path(session_id, resolved, f"limit={int(limit)}")
        # Messages endpoint returns a JSON list (not an object), so fetch
        # raw instead of via _request (which coerces non-dict bodies to {}).
        url = P.base_url() + path
        headers = {"Accept": "application/json"}
        headers.update(_auth_header())
        headers.update(P.client_headers(session_id))
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if int(getattr(resp, "status", 200) or 200) != 200:
                return []
            raw = resp.read().decode("utf-8", "replace")
        parsed = json.loads(raw) if raw.strip() else []
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _final_result_from_messages(messages: list, provider_id: str,
                                model_id: str) -> dict | None:
    """Newest assistant message with text/reasoning -> parsed result.

    Returns None when no usable assistant message exists (caller falls back
    to the blocking path or surfaces OPENCODE_EMPTY). Never raises.
    """
    try:
        for msg in reversed(messages or []):
            if not isinstance(msg, dict):
                continue
            info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
            if str(info.get("role") or "") != "assistant":
                continue
            parts = msg.get("parts")
            if not isinstance(parts, list) or not parts:
                continue
            try:
                return _parse_message_result(
                    {"info": info, "parts": parts}, provider_id, model_id)
            except Exception:
                continue
    except Exception:
        pass
    return None


def stream_message(session_id: str, provider_id: str, model_id: str,
                   agent: str, prompt: str,
                   timeout: float = DEFAULT_TIMEOUT,
                   variant: str | None = None,
                   directory: str | None = None,
                   on_content=None, on_thinking=None,
                   cancel_event=None,
                   tools: dict | None | object = _DEFAULT_TOOLS) -> dict:
    """prompt_async + /event SSE tail -> final result dict (live deltas).

    Subscribes to ``GET /event?directory=`` first (so early deltas are not
    missed), sends ``prompt_async``, forwards ``message.part.delta`` text /
    reasoning deltas via the callbacks as they arrive, waits for
    ``session.idle`` for this session, then returns the authoritative final
    result fetched via ``GET /session/:id/message`` (same shape as
    :func:`send_message`). Raises RuntimeError on transport/upstream failure
    with the classifier code prefix; never returns partial text as the
    result (partials only ride the callbacks). ``tools`` mirrors
    :func:`send_message` (default minimal, None omits).
    """
    import threading
    import time as _time

    resolved = directory if directory is not None else P.session_directory()
    query = P.session_query(resolved)

    def _cancelled() -> bool:
        try:
            return bool(cancel_event is not None and cancel_event.is_set())
        except Exception:
            return False

    # --- SSE reader state (writer thread -> callbacks) ---------------------
    _connected = threading.Event()
    _idle = threading.Event()
    _sse_error: list[str] = []
    _finish_seen: list[str] = []
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
        try:
            etype = str(payload.get("type") or "")
            props = payload.get("properties")
            if not isinstance(props, dict):
                return
            if str(props.get("sessionID") or "") != session_id:
                return
            if etype == "message.part.delta":
                field = str(props.get("field") or "").lower()
                delta = props.get("delta")
                delta_text = str(delta) if isinstance(delta, str) else ""
                if not delta_text:
                    return
                if field in ("reasoning", "thinking", "reasoning_text"):
                    _emit_thinking(delta_text)
                elif field in ("text", ""):
                    _emit_content(delta_text)
            elif etype == "message.updated":
                info = props.get("info")
                if isinstance(info, dict):
                    fin = str(info.get("finish") or "").strip()
                    if fin:
                        _finish_seen.append(fin)
                    err = info.get("error")
                    if err:
                        detail = err if isinstance(err, str) else json.dumps(err)
                        _sse_error.append(detail[:500])
            elif etype in ("session.idle",):
                _idle.set()
            elif etype in ("session.error", "session.next.step.failed"):
                try:
                    _sse_error.append(json.dumps(props)[:500])
                except Exception:
                    _sse_error.append(str(props)[:500])
        except Exception:
            pass

    def _read_sse() -> None:
        try:
            url = P.base_url() + f"/event{query}"
            headers = {"Accept": "text/event-stream"}
            headers.update(_auth_header())
            headers.update(P.client_headers(session_id))
            req = urllib.request.Request(url, method="GET", headers=headers)
            # Short socket timeout so cancel/idle checks stay responsive;
            # the outer loop re-reads until idle/timeout.
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
        prompt_async(session_id, provider_id, model_id, agent, prompt,
                     timeout=min(30.0, timeout), variant=variant,
                     directory=resolved, tools=tools)
    except Exception:
        _stop_reader.set()
        raise

    deadline = _time.monotonic() + max(10.0, float(timeout) + 30.0)
    while _time.monotonic() < deadline:
        if _cancelled():
            try:
                abort_session(session_id, directory=resolved)
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
        _time.sleep(0.05)
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
    # Authoritative final: newest assistant message (covers multi-step tool
    # loops; deltas already streamed live via callbacks).
    messages = get_session_messages(session_id, timeout=20,
                                    directory=resolved, limit=20)
    final = _final_result_from_messages(messages, provider_id, model_id)
    if final is not None:
        return final
    if _sse_error and not _idle.is_set():
        # SSE-level failure without a usable message: surface for retry.
        raise RuntimeError(f"OPENCODE_UPSTREAM: live stream failed: "
                           f"{_sse_error[0][:300]}")
    raise RuntimeError(
        f"OPENCODE_EMPTY: empty completion for {provider_id}/{model_id} "
        "(live stream ended without an assistant message)")


def list_provider_models(timeout: float = 20) -> list[str]:
    """Best-effort ``provider/model`` ids from GET /config/providers.

    Used for diagnostics; the chat path does not depend on it (model ids
    pass through). [] on any failure, never raises.
    """
    try:
        status, data = _request("GET", "/config/providers", timeout=timeout)
        if status != 200:
            return []
        providers = data.get("providers")
        if not isinstance(providers, list):
            return []
        out: list[str] = []
        for prov in providers:
            if not isinstance(prov, dict):
                continue
            pid = str(prov.get("id") or "").strip()
            models = prov.get("models")
            if not pid or not isinstance(models, dict):
                continue
            for mid in models:
                mid = str(mid or "").strip()
                if mid:
                    out.append(f"{pid}/{mid}")
        return out
    except Exception:
        return []
