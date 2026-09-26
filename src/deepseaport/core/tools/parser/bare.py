"""Format 5: bare unfenced name/arguments objects."""

from __future__ import annotations

import logging
import re

from ..common import (
    MAX_BARE_OBJECTS,
    _LOOSE_JSON_MISS,
    _balanced_json,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
    _strip_spans,
)

logger = logging.getLogger("deepseaport.tools")

# Near-miss counter for diagnostics only (unknown-name {"name": ...} shapes).
# Module-level so the success/miss path never recompiles per call.
_BARE_NAME_RE = re.compile(r'["\']name["\']\s*:\s*["\']([^"\']{1,256})["\']')


def _parse_bare_name_objects(
    text: str,
    valid: set[str] | None,
    max_objects: int | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Unfenced {"name":..., "arguments":...} objects (parallel-safe).

    Generic scan, so gated by valid-names match to avoid false positives on
    normal code samples. Returns (raw_calls, spans_to_strip).
    """
    limit = max_objects if max_objects is not None else MAX_BARE_OBJECTS
    lowers = {n.lower() for n in valid} if valid else set()
    raw: list[dict] = []
    spans: list[tuple[int, int]] = []
    count = 0
    pos = 0
    n = len(text)
    # Fast exit: no name/function marker means no bare tool shape.
    if '"name"' not in text and "'name'" not in text and '"function"' not in text:
        return raw, spans
    while pos < n:
        if count >= limit:  # bound scan on long replies (tunable)
            break
        start = text.find("{", pos)
        if start < 0:
            break
        obj_text = _balanced_json(text, start)
        if not obj_text:
            pos = start + 1
            continue
        count += 1
        pos = start + len(obj_text)
        # Skip lenient repair for objects without tool shape (code samples).
        if ('"name"' not in obj_text and "'name'" not in obj_text
                and '"function"' not in obj_text):
            continue
        obj = _loads_json_lenient(obj_text)
        if obj is _LOOSE_JSON_MISS:
            continue
        if not isinstance(obj, dict):
            continue
        candidate = None
        if isinstance(obj.get("name"), str) and ("arguments" in obj or "parameters" in obj):
            args = obj.get("arguments", obj.get("parameters", {}))
            candidate = {"function": {"name": obj["name"], "arguments": args if args is not None else "{}"}}
            if isinstance(obj.get("id"), str):
                candidate["id"] = obj["id"]
        elif isinstance(obj.get("function"), dict) and isinstance(obj["function"].get("name"), str):
            candidate = obj
        if candidate is None:
            continue
        nm = candidate["function"]["name"] if isinstance(candidate.get("function"), dict) else None
        if not isinstance(nm, str):
            continue
        if valid and nm not in valid and nm.lower() not in lowers:
            continue
        raw.append(candidate)
        spans.append((start, start + len(obj_text)))
    return raw, spans


def try_parse_bare(text, valid, valid_or_none, limit_args, limit_bare, forgiving_on,
                   diagnostics=None):
    if valid:
        bare_raw, bare_spans = _parse_bare_name_objects(text, valid, limit_bare)
        if bare_raw:
            calls, dropped = _normalize_calls(bare_raw, valid_or_none, limit_args)
            if diagnostics is not None:
                diagnostics.append({"format": "bare", "matched": bool(calls),
                                    "kept": len(calls), "dropped": list(dropped),
                                    "raw": len(bare_raw)})
            if calls:
                # Strip only the spans that survived normalization.
                rest = _clean_rest(_strip_spans(text, bare_spans), forgiving_on)
                return calls, rest
            return None, None
        # No valid-name candidates: record unknown-name near-miss so
        # diagnostics distinguishes plain answers (no trail) from
        # near-misses (matched=False trail). Bounded C-level scan only.
        if diagnostics is not None and ('"name"' in text or "'name'" in text
                                        or '"function"' in text):
            cap = limit_bare if limit_bare is not None else MAX_BARE_OBJECTS
            # Double + single quoted name values (lenient shapes included).
            found = 0
            for _m in _BARE_NAME_RE.finditer(text):
                found += 1
                if found >= cap:
                    break
            if found:
                diagnostics.append({"format": "bare", "matched": False,
                                    "kept": 0, "dropped": list(range(found)),
                                    "raw": found})
    return None, None
