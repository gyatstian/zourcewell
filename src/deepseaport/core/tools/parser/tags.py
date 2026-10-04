"""Format 2: <tool_call>/<function_call>/<invoke> JSON tags."""

from __future__ import annotations

import logging
import re

from ..common import (
    MAX_SCAN_OBJECTS,
    _LOOSE_JSON_MISS,
    _finish,
    _loads_json_lenient,
)

logger = logging.getLogger("deepseaport.tools")

_TAG_RE = re.compile(
    r"<(?:tool_call|function_call|invoke)>(.*?)</(?:tool_call|function_call|invoke)>",
    re.DOTALL | re.IGNORECASE,
)


def try_parse_tags(text, valid_or_none, limit_args, forgiving_on,
                   diagnostics=None):
    tagged = []
    scanned = 0
    for m in _TAG_RE.finditer(text):
        if scanned >= MAX_SCAN_OBJECTS:
            break
        scanned += 1
        obj = _loads_json_lenient(m.group(1).strip())
        if obj is _LOOSE_JSON_MISS:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            tagged.append({"function": obj})
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            tagged.append(obj)
    if tagged:
        hit = _finish(tagged, valid_or_none, limit_args, "tags", forgiving_on,
                      diagnostics=diagnostics,
                      on_hit=lambda: _TAG_RE.sub("", text))
        if hit[0]:
            return hit
    return None, None
