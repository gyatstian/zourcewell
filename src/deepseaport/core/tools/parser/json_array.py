"""Format 1: {"tool_calls": [...]} balanced scan."""

from __future__ import annotations

import logging

from ..common import (
    MAX_SCAN_OBJECTS,
    _LOOSE_JSON_MISS,
    _balanced_json,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_json_array(text, valid_or_none, limit_args, forgiving_on,
                         diagnostics=None):
    """First match wins: single left-to-right balanced scan (linear)."""
    # Fast C-level exit: no tool_calls marker means no work (plain answers
    # with many "{" skip Python-loop balanced scans entirely).
    if '"tool_calls"' not in text:
        return None, None
    pos = 0
    n = len(text)
    scanned = 0
    while pos < n:
        if scanned >= MAX_SCAN_OBJECTS:
            break
        # Remaining text can't match once the marker is gone.
        if text.find('"tool_calls"', pos) < 0:
            break
        start = text.find("{", pos)
        if start < 0:
            break
        scanned += 1
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
            calls, dropped = _normalize_calls(raw, valid_or_none, limit_args)
            if diagnostics is not None:
                diagnostics.append({"format": "json_array", "matched": bool(calls),
                                    "kept": len(calls), "dropped": list(dropped),
                                    "raw": len(raw)})
            if calls:
                rest = _clean_rest(text[:start] + text[start + len(obj_text):], forgiving_on)
                return calls, rest
        pos = start + len(obj_text)
    return None, None
