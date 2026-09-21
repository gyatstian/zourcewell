"""Format 1: {"tool_calls": [...]} balanced scan."""

from __future__ import annotations

import logging

from ..common import (
    _LOOSE_JSON_MISS,
    _balanced_json,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_json_array(text, valid_or_none, limit_args, forgiving_on):
    """First match wins: single left-to-right balanced scan (linear)."""
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("{", pos)
        if start < 0:
            break
        obj_text = _balanced_json(text, start)
        if not obj_text:
            pos = start + 1
            continue
        if '"tool_calls"' not in obj_text:
            pos = start + len(obj_text)
            continue
        obj = _loads_json_lenient(obj_text)
        if obj is _LOOSE_JSON_MISS:
            pos = start + 1
            continue
        raw = obj.get("tool_calls") if isinstance(obj, dict) else None
        if isinstance(raw, list) and raw:
            calls, _ = _normalize_calls(raw, valid_or_none, limit_args)
            if calls:
                rest = _clean_rest(text[:start] + text[start + len(obj_text):], forgiving_on)
                return calls, rest
        pos = start + len(obj_text)
    return None, None
