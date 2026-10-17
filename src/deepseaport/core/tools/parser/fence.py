"""Format 4: fenced ```json block describing a call."""

from __future__ import annotations

import logging

from ..common import (
    MAX_SCAN_OBJECTS,
    _FENCED_BLOCK_RE,
    _LOOSE_JSON_MISS,
    _balanced_json,
    _finish,
    _loads_json_lenient,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_fence(text, valid_or_none, limit_args, forgiving_on,
                    diagnostics=None):
    # Balanced scan inside each fence so nested arguments survive
    # (non-greedy \{.*?\} stopped at the first inner "}").
    # Bounded: at most MAX_SCAN_OBJECTS balanced objects total across fences.
    scanned = 0
    for m in _FENCED_BLOCK_RE.finditer(text):
        if scanned >= MAX_SCAN_OBJECTS:
            break
        inner = m.group(1)
        # Fast skip: fence without tool shape avoids balanced+lenient work.
        if ('"name"' not in inner and '"function"' not in inner
                and '"tool_calls"' not in inner):
            continue
        inner_pos = 0
        while inner_pos < len(inner):
            if scanned >= MAX_SCAN_OBJECTS:
                break
            obj_start = inner.find("{", inner_pos)
            if obj_start < 0:
                break
            scanned += 1
            obj_text = _balanced_json(inner, obj_start)
            if not obj_text:
                inner_pos = obj_start + 1
                continue
            # Skip lenient repair for objects without tool shape.
            if ('"name"' not in obj_text and '"function"' not in obj_text
                    and '"tool_calls"' not in obj_text):
                inner_pos = obj_start + len(obj_text)
                continue
            obj = _loads_json_lenient(obj_text)
            if obj is _LOOSE_JSON_MISS:
                inner_pos = obj_start + 1
                continue
            raw = None
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                raw = obj["tool_calls"]
            # Arguments/parameters key required: a fenced name-only object is
            # an illustrative example (same guard as the bare parser), and
            # executing it with "{}" would both fail and eat the example.
            elif (isinstance(obj, dict) and obj.get("name")
                  and ("arguments" in obj or "parameters" in obj)):
                raw = [{"function": obj}]
            elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
                raw = [obj]
            if raw:
                hit = _finish(raw, valid_or_none, limit_args, "fence",
                              forgiving_on, diagnostics=diagnostics,
                              on_hit=lambda: text[:m.start()] + text[m.end():])
                if hit[0]:
                    return hit
            inner_pos = obj_start + len(obj_text)
    return None, None
