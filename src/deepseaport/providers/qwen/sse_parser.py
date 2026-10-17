"""Qwen SSE event parsing: state-free ``QwenStreamParser`` + one-liner.

Qwen's stream (``POST /api/v2/chat/completions``) is OpenAI-style JSON per
``data:`` line — NOT the DeepSeek ``{p,v}`` path protocol, so this parser is
intentionally separate. Event shapes are LIVE-VERIFIED (2026-09-22 real
completion streams); the parser stays lenient and ignores unknown keys.

- FINISH signal = ``choices[0].delta.status == "finished"`` (empty content).
  No ``[DONE]`` line / ``response.stopped`` / ``done:true`` observed — the
  stream simply ends after the finished chunk (done:[DONE] still handled).
- usage keys are ``input_tokens``/``output_tokens``/``total_tokens``; the
  usage event carries the COMPLETION count — see
  :meth:`QwenStreamParser._usage_count` for the key order and why.
- error envelope: ``{"error":{"code":"quota_limit","details":"..."}}`` — the
  message lives in ``details``, not ``message``.

Ownership: SSE line/event parsing here; header/payload helpers and WAF/ban
detection stay in protocol.py (deferred local import, to avoid the module
cycle).
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
    def _usage_count(usage) -> int | None:
        """Completion-token count from a Qwen usage object, else None.

        Live keys are ``input_tokens``/``output_tokens``/``total_tokens``; the
        classic ``prompt_tokens``/``completion_tokens`` are also accepted.
        ``output_tokens``/``completion_tokens`` is preferred because the
        consumer (``core.openai.build_usage``) treats this number as the
        COMPLETION count, and Qwen's ``total_tokens`` is input+output — using
        it here would report the prompt as completion tokens too and roughly
        double the usage of a long agentic session. ``total_tokens`` is only a
        last resort for a payload that omits both output keys.
        """
        try:
            for key in ("output_tokens", "completion_tokens", "total_tokens"):
                if key not in usage:
                    continue
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    continue  # unusable value: fall through to the next key
        except TypeError:
            return None
        return None

    @classmethod
    def _usage_event(cls, usage) -> StreamEvent | None:
        count = cls._usage_count(usage)
        if count is None:
            return None
        return StreamEvent(kind="usage", text=str(count))

    @staticmethod
    def _merge_usage(*usages) -> dict | None:
        """Union of usage objects, summing numeric counters (multi-choice)."""
        merged: dict = {}
        seen = False
        for usage in usages:
            if not isinstance(usage, dict) or not usage:
                continue
            seen = True
            for key, val in usage.items():
                if isinstance(val, bool):
                    merged[key] = val
                    continue
                if isinstance(val, (int, float)):
                    try:
                        merged[key] = merged.get(key, 0) + val
                    except TypeError:
                        merged[key] = val
                else:
                    merged[key] = val
        return merged if seen else None

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
        finished = False
        per_choice: list = []
        for item in choices:
            if not isinstance(item, dict):
                continue
            # Usage may ride on the choice object (the module contract), not
            # only on the chunk: collect every choice's usage so the union is
            # available below.
            choice_usage = item.get("usage")
            if isinstance(choice_usage, dict):
                per_choice.append(choice_usage)
            delta = item.get("delta")
            if not isinstance(delta, dict):
                continue
            # Live FINISH signal: delta.status == "finished" (empty content).
            if str(delta.get("status") or "").lower() == "finished":
                # Record, do NOT return: the usage block for this chunk still
                # has to run, and a choice after the finished delta (never seen
                # live, but the wire allows it) must not be dropped. The finish
                # event is emitted after usage — the consumer stops reading at
                # ``finished`` (core.engine.drain_attempt / transport.pump_sse),
                # so usage emitted afterwards would be lost.
                finished = True
                continue
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
        # Usage rides on the choices line: parse best-effort (chunk-level first,
        # then the union of any per-choice usage). Prefer the completion count
        # (output_tokens) — see _usage_count.
        usage = QwenStreamParser._usage_event(chunk.get("usage"))
        if usage is None and per_choice:
            usage = QwenStreamParser._usage_event(
                QwenStreamParser._merge_usage(*per_choice))
        if usage is not None:
            yield usage
        if finished:
            yield StreamEvent(kind="finished")
        return
    # Unknown top-level keys (sources, selected_model_id, {phase,value} tool
    # output): ignore.
    return
