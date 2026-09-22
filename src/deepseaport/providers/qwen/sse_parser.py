"""Qwen SSE event parsing: state-free lenient QwenStreamParser + one-liner.

Qwen's stream (POST /api/v2/chat/completions) is OpenAI-style JSON per
``data:`` line — NOT the DeepSeek ``{p,v}`` path protocol, so this parser is
intentionally separate. Event shapes below are LIVE-VERIFIED (2026-09-22 real
completions streams); the parser stays lenient and ignores unknown keys.

Live stream facts:
- FINISH signal = ``choices[0].delta.status == "finished"`` (empty content).
  No ``[DONE]`` line / ``response.stopped`` / ``done:true`` observed — the
  stream simply ends after the finished chunk (done:[DONE] still handled).
- usage keys are ``input_tokens``/``output_tokens``/``total_tokens``.
- error envelope: ``{"error":{"code":"quota_limit","details":"..."}}`` — the
  message lives in ``details``, not ``message``.

Ownership:
- This module owns SSE line/event parsing for Qwen.
- Header/payload helpers and WAF/ban detection live in protocol.py
  (re-exported only via deferred local import to avoid a module cycle).
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from ...core.types import StreamEvent

logger = logging.getLogger("deepseaport.protocol")

# delta.phase values observed in the stream client (lowercased here; the
# parser lowercases incoming phases before matching).
_PHASE_THINKING = ("think", "think_summary")
_PHASE_IGNORED = ("keepalive",)


class QwenStreamParser:
    """Lenient, state-free SSE parser for Qwen's OpenAI-style stream."""

    def feed(self, line: str) -> Iterator[StreamEvent]:
        yield from parse_sse_line(line)

    @staticmethod
    def _usage_event(usage) -> StreamEvent | None:
        if not isinstance(usage, dict):
            return None
        # Live usage keys: input_tokens/output_tokens/total_tokens. Also accept
        # the classic prompt_tokens/completion_tokens. Prefer total.
        total = usage.get("total_tokens")
        if total is None:
            total = usage.get("output_tokens")
        if total is None:
            total = usage.get("completion_tokens")
        if isinstance(total, int):
            return StreamEvent(kind="usage", text=str(total))
        try:
            total = int(total)
            return StreamEvent(kind="usage", text=str(total))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_event(data: dict) -> StreamEvent | None:
        # Local import: protocol.py imports this module at top level, so a
        # top-level import here would cycle.
        try:
            from .protocol import detect_ban as _detect_ban
        except Exception:
            _detect_ban = None  # type: ignore[assignment]
        err = data.get("error")
        if isinstance(err, dict):
            # Live envelope: {"error":{"code":"quota_limit","details":"..."}}.
            # details is the verified message key; message is the fallback.
            code = str(err.get("code") or err.get("type") or "UPSTREAM_ERROR")
            message = str(err.get("details") or err.get("message") or "")
            ban_until: float | None = None
            try:
                if _detect_ban is not None:
                    banned, ban_until = _detect_ban(
                        {"message": message, "code": code,
                         "data": err if isinstance(err, dict) else {}})
                    if not banned:
                        ban_until = None
            except Exception:
                ban_until = None
            return StreamEvent(kind="error", code=code, message=message,
                               ban_until=ban_until)
        if isinstance(err, str):
            ban_until = None
            try:
                if _detect_ban is not None:
                    banned, ban_until = _detect_ban({"message": err})
                    if not banned:
                        ban_until = None
            except Exception:
                ban_until = None
            return StreamEvent(kind="error", code="UPSTREAM_ERROR", message=err,
                               ban_until=ban_until)
        return None


def parse_sse_line(line: str) -> Iterator[StreamEvent]:
    """Stateless one-line parse of a Qwen ``data:`` SSE line.

    Handles ``data: [DONE]`` and the live-verified top-level keys:
    response.created / choices / usage / error / done. The FINISH signal is
    ``choices[0].delta.status == "finished"``. Unknown or non-JSON lines are
    ignored.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return
    data = line[5:].strip()
    if data == "[DONE]":
        yield StreamEvent(kind="finished")
        return
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        logger.debug("skip non-JSON SSE line: %.80s", data)
        return
    if not isinstance(chunk, dict):
        return
    # Terminal markers.
    if chunk.get("done") is True:
        yield StreamEvent(kind="finished")
        return
    if "response.stopped" in chunk:
        # Turn finished (JS-grep, not seen live). Do NOT emit finished here:
        # a final choices/usage line may still follow; stay lenient.
        return
    if "response.created" in chunk or "response.info" in chunk:
        # Start marker / skip_think info: nothing to surface.
        return
    # Error envelope: {"error": {...}} or {"error": "..."}.
    err = QwenStreamParser._error_event(chunk)
    if err is not None:
        yield err
        return
    # Usage-only line: {"usage": {...}} without choices.
    if "usage" in chunk and "choices" not in chunk:
        usage = QwenStreamParser._usage_event(chunk.get("usage"))
        if usage is not None:
            yield usage
        return
    # Content chunks: {"choices":[{delta:{...}, "usage":{...}}], ...}.
    choices = chunk.get("choices")
    if isinstance(choices, list):
        for item in choices:
            if not isinstance(item, dict):
                continue
            delta = item.get("delta")
            if not isinstance(delta, dict):
                continue
            # Live FINISH signal: delta.status == "finished" (empty content).
            if str(delta.get("status") or "").lower() == "finished":
                yield StreamEvent(kind="finished")
                return
            phase = str(delta.get("phase") or "").lower()
            content = delta.get("content")
            if phase in _PHASE_THINKING or (phase == "" and delta.get("thinking_content")):
                text = delta.get("thinking_content", content)
                if isinstance(text, str) and text:
                    yield StreamEvent(kind="thinking", text=text)
                continue
            if phase in _PHASE_IGNORED:
                continue
            # answer / absent / tool_call phases -> surface as content; a
            # tool_call payload may also carry function_call JSON — surfaced
            # as content text so the tool parser can recover it.
            if isinstance(content, str) and content:
                yield StreamEvent(kind="content", text=content)
        # Usage rides on the choices line: parse best-effort (prefer total).
        if "usage" in chunk:
            usage = QwenStreamParser._usage_event(chunk.get("usage"))
            if usage is not None:
                yield usage
        return
    # Unknown top-level keys (sources, selected_model_id, {phase,value} tool
    # output): ignore.
    return
