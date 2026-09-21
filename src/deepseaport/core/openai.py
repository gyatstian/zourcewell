"""Provider-neutral OpenAI response shaping: shared by all web2api providers. Tool parsing resolves through `deepseaport.server.parse_tool_calls` at call time so tests keep patching it."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from .tools import looks_truncated_tool_attempt

logger = logging.getLogger("deepseaport.server")


def _openai_response(prep: dict, result: dict) -> dict:
    # Late-bound so tests can monkeypatch deepseaport.server.parse_tool_calls.
    from ..server import parse_tool_calls
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    limit = prep.get("tool_args_max_chars")
    forgiving = bool(prep.get("forgiving_toolcalls", False))
    if prep["tools"]:
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
        logger.debug("no tool call parsed model=%s preview=%.200s",
                     prep.get("model"), (result["content"] or "")[:200])
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
    pt = max(1, len(prep["prompt"]) // 4)
    ct = result.get("usage_total") or max(1, (len(result["content"]) + len(result["thinking"])) // 4)
    return {"id": cid, "object": "chat.completion", "created": created, "model": prep["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}
