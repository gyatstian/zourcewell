"""Format 4: fenced ```json block describing a call."""

from __future__ import annotations

import logging

from ..common import (
    _FENCED_BLOCK_RE,
    _LOOSE_JSON_MISS,
    _balanced_json,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_fence(text, valid_or_none, limit_args, forgiving_on):
    # Balanced scan inside each fence so nested arguments survive
    # (non-greedy \{.*?\} stopped at the first inner "}").
    for m in _FENCED_BLOCK_RE.finditer(text):
        inner = m.group(1)
        inner_pos = 0
        while inner_pos < len(inner):
            obj_start = inner.find("{", inner_pos)
            if obj_start < 0:
                break
            obj_text = _balanced_json(inner, obj_start)
            if not obj_text:
                inner_pos = obj_start + 1
                continue
            obj = _loads_json_lenient(obj_text)
            if obj is _LOOSE_JSON_MISS:
                inner_pos = obj_start + 1
                continue
            raw = None
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                raw = obj["tool_calls"]
            elif isinstance(obj, dict) and obj.get("name"):
                raw = [{"function": obj}]
            elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
                raw = [obj]
            if raw:
                calls, _ = _normalize_calls(raw, valid_or_none, limit_args)
                if calls:
                    rest = _clean_rest(text[:m.start()] + text[m.end():], forgiving_on)
                    return calls, rest
            inner_pos = obj_start + len(obj_text)
    return None, None
