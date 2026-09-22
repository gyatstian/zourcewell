"""Format 2: <tool_call>/<function_call>/<invoke> JSON tags."""

from __future__ import annotations

import logging
import re

from ..common import (
    _LOOSE_JSON_MISS,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
)

logger = logging.getLogger("deepseaport.tools")


def try_parse_tags(text, valid_or_none, limit_args, forgiving_on,
                   diagnostics=None):
    tag_pat = re.compile(
        r"<(?:tool_call|function_call|invoke)>(.*?)</(?:tool_call|function_call|invoke)>",
        re.DOTALL | re.IGNORECASE,
    )
    tagged = []
    for m in tag_pat.finditer(text):
        obj = _loads_json_lenient(m.group(1).strip())
        if obj is _LOOSE_JSON_MISS:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            tagged.append({"function": obj})
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            tagged.append(obj)
    if tagged:
        calls, dropped = _normalize_calls(tagged, valid_or_none, limit_args)
        if diagnostics is not None:
            diagnostics.append({"format": "tags", "matched": bool(calls),
                                "kept": len(calls), "dropped": list(dropped),
                                "raw": len(tagged)})
        if calls:
            rest = _clean_rest(tag_pat.sub("", text), forgiving_on)
            return calls, rest
    return None, None
