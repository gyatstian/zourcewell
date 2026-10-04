"""Provider-neutral OpenAI response shaping: shared by all web2api providers.

Tool parsing resolves to ``core.tools`` directly (no core->server import,
no cycle). When ``deepseaport.server`` is already loaded, its (possibly
monkeypatched) ``parse_tool_calls`` wins via a soft ``sys.modules`` lookup
so tests patching ``deepseaport.server.parse_tool_calls`` keep working;
otherwise the real parser is used. An explicit ``parse_fn`` overrides both.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from .tools import looks_truncated_tool_attempt

logger = logging.getLogger("deepseaport.server")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for web upstreams (no tokenizer available).

    Upstream web SSE gives at most an accumulated total (DeepSeek) or
    output total (Qwen); prompt tokens are never reported. ~4 chars/token
    is the historical heuristic — kept explicit here so every caller shares
    one formula instead of drifting len//4 copies.
    """
    try:
        return max(1, len(text or "") // 4)
    except Exception:
        return 1


def build_usage(prep: dict, content: str, thinking: str,
                usage_total: int | None = None) -> dict:
    """Build an OpenAI usage dict with one shared estimate.

    ``usage_total`` (upstream accumulated/output total when the SSE parser
    saw one) wins for completion_tokens; otherwise the len//4 estimate is
    used. prompt_tokens is always estimated — the web APIs never report it.
    """
    pt = estimate_tokens(str(prep.get("prompt", "") or ""))
    try:
        ct = int(usage_total) if usage_total else 0
    except (TypeError, ValueError):
        ct = 0
    if not ct:
        # content+thinking concatenated then //4 would undercount by the
        # join; estimate them separately would double the max(1,·) floor —
        # use the sum of lengths instead.
        try:
            ct = max(1, (len(content or "") + len(thinking or "")) // 4)
        except Exception:
            ct = 1
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct}


def format_miss_trail(diagnostics: list[dict] | None,
                      valid: set[str] | None = None) -> str:
    """Compact one-line summary of parse_tool_calls_ex diagnostics.

    Turns the per-format trail into an operator-actionable hint:
    which format almost matched (matched but kept 0 = unknown-name or
    oversize drop), vs no format matched at all (format drift / plain
    answer). Used by both buffered and streaming miss logs (single home;
    engine imports this instead of an inline copy).
    """
    try:
        parts: list[str] = []
        for d in diagnostics or []:
            try:
                parts.append(
                    f"{d.get('format')}:matched={int(bool(d.get('matched')))} "
                    f"kept={d.get('kept', 0)} dropped={list(d.get('dropped') or [])} "
                    f"raw={d.get('raw', 0)}")
            except Exception:
                continue
        trail = "; ".join(parts) if parts else "no formats matched"
        if valid:
            return f"{trail} valid={sorted(valid)}"
        return trail
    except Exception:
        return "diagnostics unavailable"


def _resolve_parse_fn(explicit=None):
    """Return (parse_fn, real_parse, real_ex) without importing server.

    Soft seam: when ``deepseaport.server`` is already in ``sys.modules``
    (normal runtime + tests), its ``parse_tool_calls`` attribute wins so
    monkeypatched fakes keep working. No import is forced, so core never
    depends on server and no cycle is created. Explicit arg wins over both.
    """
    from .tools.parser import parse_tool_calls as _real_parse
    from .tools.parser import parse_tool_calls_ex as _real_ex
    if explicit is not None:
        return explicit, _real_parse, _real_ex
    try:
        import sys as _sys
        _mod = _sys.modules.get("deepseaport.server")
        _cand = getattr(_mod, "parse_tool_calls", None) if _mod is not None else None
        if callable(_cand):
            return _cand, _real_parse, _real_ex
    except Exception:
        pass
    return _real_parse, _real_parse, _real_ex


def _log_tool_miss(prep: dict, content: str, where: str = "response") -> None:
    """Warning-level miss log with diagnostics (unknown vs oversize vs drift).

    Runs the real ``parse_tool_calls_ex`` for logging only — the response
    itself still comes from the (possibly monkeypatched) ``parse_tool_calls``
    seam above, so tests keep patching that seam. Never raises.
    """
    try:
        from .tools.parser import parse_tool_calls_ex as _ex
        from .tools import _valid_names
    except Exception:
        logger.debug("no tool call parsed %s model=%s preview=%.200s",
                     where, prep.get("model"), (content or "")[:200])
        return
    try:
        limit = prep.get("tool_args_max_chars")
        forgiving = bool(prep.get("forgiving_toolcalls", False))
        tools = prep.get("tools") or []
        kwargs: dict = {"forgiving": forgiving}
        if limit:
            kwargs["max_args_chars"] = limit
        res = _ex(content or "", tools, **kwargs)
        valid = _valid_names(tools)
        summary = format_miss_trail(res.diagnostics, valid)
        logger.warning(
            "tool-call miss (%s) model=%s forgiving=%s %s preview=%.300s",
            where, prep.get("model"), forgiving, summary,
            (content or "")[:300])
    except Exception:
        logger.debug("no tool call parsed %s model=%s preview=%.200s",
                     where, prep.get("model"), (content or "")[:200])


async def _alog_tool_miss(prep: dict, content: str,
                          diagnostics: list[dict] | None = None,
                          *, where: str = "stream") -> None:
    """Async twin of :func:`_log_tool_miss` for the streaming parse path.

    Same warning-level miss log with the diagnostics trail. Takes the
    ``diagnostics`` the stream parse already produced; when it has none (the
    parse ran through a monkeypatched seam) it re-runs ``parse_tool_calls_ex``
    for logging only, exactly like the sync helper. The parse itself stays on
    the caller's off-thread call, so this stays off the event loop. Never
    raises.
    """
    try:
        from .tools import _valid_names
        from .tools.parser import parse_tool_calls_ex as _ex

        limit = prep.get("tool_args_max_chars")
        forgiving = bool(prep.get("forgiving_toolcalls", False))
        tools = prep.get("tools") or []
        if diagnostics is None:
            kwargs: dict = {"forgiving": forgiving}
            if limit:
                kwargs["max_args_chars"] = limit
            res = await asyncio.to_thread(_ex, content or "", tools, **kwargs)
            diagnostics = res.diagnostics
        summary = format_miss_trail(diagnostics, _valid_names(tools))
        logger.warning(
            "tool-call miss (%s) model=%s forgiving=%s %s preview=%.300s",
            where, prep.get("model"), forgiving, summary, (content or "")[:300])
    except Exception:
        logger.debug("no tool call parsed %s model=%s preview=%.200s",
                     where, prep.get("model"), (content or "")[:200])


def _log_tool_dropped(prep: dict, dropped: list[int] | None,
                      where: str = "response") -> None:
    """Warn when some (but not all) calls were dropped (partial miss)."""
    try:
        if dropped:
            logger.warning(
                "tool-call partial drop (%s) model=%s dropped=%s "
                "(unknown-name or oversize args; see prior drop lines)",
                where, prep.get("model"), list(dropped))
    except Exception:
        pass


def _openai_response(prep: dict, result: dict, parse_fn=None) -> dict:
    # No core->server import: soft sys.modules lookup keeps the
    # deepseaport.server.parse_tool_calls monkeypatch seam without a cycle.
    parse_tool_calls, _real_parse, _real_ex = _resolve_parse_fn(parse_fn)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    limit = prep.get("tool_args_max_chars")
    forgiving = bool(prep.get("forgiving_toolcalls", False))
    _dropped: list[int] | None = None
    if prep["tools"]:
        if parse_tool_calls is _real_parse:
            # Unpatched production path: single parse returns calls + the
            # drop trail (no double-parse cost). ``diagnostics`` from the same
            # result is NOT captured here: the miss log re-runs the parser via
            # _log_tool_miss, so an unused local only looked like it did work.
            kwargs: dict = {"forgiving": forgiving}
            if limit:
                kwargs["max_args_chars"] = limit
            res = _real_ex(result["content"], prep["tools"], **kwargs)
            calls, remaining = res.calls, res.rest
            _dropped = res.dropped
        else:
            if limit:
                calls, remaining = parse_tool_calls(
                    result["content"], prep["tools"],
                    max_args_chars=limit, forgiving=forgiving)
            else:
                calls, remaining = parse_tool_calls(
                    result["content"], prep["tools"], forgiving=forgiving)
    else:
        calls, remaining = None, result["content"]
    if prep["tools"] and not calls:
        _log_tool_miss(prep, result.get("content", ""), where="response")
    elif _dropped:
        _log_tool_dropped(prep, _dropped, where="response")
    message: dict[str, Any] = {"role": "assistant", "content": remaining}
    if result["thinking"]:
        message["reasoning_content"] = result["thinking"]
    finish = "stop"
    if calls:
        message["tool_calls"] = calls
        finish = "tool_calls"
    elif prep["tools"] and looks_truncated_tool_attempt(
            result["content"], prep["tools"], forgiving=forgiving):
        # The upstream web protocol has no finish_reason. An unbalanced tool
        # attempt means the reply was cut mid-call; report OpenAI's length
        # signal so the harness can retry/split instead of showing raw markup.
        finish = "length"
    # Both token counts are computed once inside build_usage (single estimate
    # formula); do not recompute them here.
    usage = build_usage(prep, result.get("content", ""), result.get("thinking", ""),
                        result.get("usage_total"))
    return {"id": cid, "object": "chat.completion", "created": created, "model": prep["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage}
