"""Format 2: <tool_call>/<function_call>/<invoke> JSON tags."""

from __future__ import annotations

import logging
import re

from ..common import (
    MAX_SCAN_OBJECTS,
    _LOOSE_JSON_MISS,
    _finish,
    _loads_json_lenient,
    _strip_spans,
)

logger = logging.getLogger("deepseaport.tools")

# Open/close literals in one alternation. The old lazy-body regex retried at
# every later position when an open had no close (quadratic: 176KB of bare
# `<tool_call>` took minutes); this is a single linear pass.
_TAG_TOKEN_RE = re.compile(
    r"<(?:tool_call|function_call|invoke)>|</(?:tool_call|function_call|invoke)>",
    re.IGNORECASE,
)


def _tag_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, body) per matched pair, in one linear pass.

    Pairing reproduces the old non-greedy regex exactly: the first open
    starts a pair, the next close of ANY of the three names ends it, and
    opens seen while a pair is pending are body text. Capped at
    MAX_SCAN_OBJECTS pairs as before.
    """
    if "</" not in text:
        return []
    spans: list[tuple[int, int, str]] = []
    pending: re.Match | None = None
    for m in _TAG_TOKEN_RE.finditer(text):
        if m.group(0).startswith("</"):
            if pending is not None:
                spans.append((pending.start(), m.end(),
                              text[pending.end():m.start()]))
                pending = None
                if len(spans) >= MAX_SCAN_OBJECTS:
                    break
        elif pending is None:
            pending = m
    return spans


def try_parse_tags(text, valid_or_none, limit_args, forgiving_on,
                   diagnostics=None, skip_spans=None):
    tagged = []
    spans: list[tuple[int, int]] = []
    for start, end, body in _tag_spans(text):
        # A tag sample inside a DSML raw value is content, not a call.
        if skip_spans and any(s <= start < e for s, e in skip_spans):
            continue
        spans.append((start, end))
        obj = _loads_json_lenient(body.strip())
        if obj is _LOOSE_JSON_MISS:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            tagged.append({"function": obj})
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            tagged.append(obj)
    if tagged:
        hit = _finish(tagged, valid_or_none, limit_args, "tags", forgiving_on,
                      diagnostics=diagnostics,
                      on_hit=lambda: _strip_spans(text, spans))
        if hit[0]:
            return hit
    return None, None
