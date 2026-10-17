"""Format 1: {"tool_calls": [...]} balanced scan."""

from __future__ import annotations

import logging

from ..common import (
    MAX_SCAN_OBJECTS,
    _LOOSE_JSON_MISS,
    _balanced_json,
    _finish,
    _loads_json_lenient,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_json_array(text, valid_or_none, limit_args, forgiving_on,
                         diagnostics=None, skip_spans=None):
    """First match wins: single left-to-right balanced scan (linear).

    ``skip_spans`` (orchestrator-supplied DSML invoke spans) marks regions
    whose content is raw text, not markup: a sample inside one must not be
    read as a call ahead of the DSML parser. ``None`` = no skipping.
    """
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
        if skip_spans and any(s <= start < e for s, e in skip_spans):
            pos = start + 1
            continue
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
            hit = _finish(raw, valid_or_none, limit_args, "json_array",
                          forgiving_on, diagnostics=diagnostics,
                          on_hit=lambda: text[:start] + text[start + len(obj_text):])
            if hit[0]:
                return hit
        pos = start + len(obj_text)
    return None, None
