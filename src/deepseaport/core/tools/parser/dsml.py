"""Format 3: DSML / plain invoke+parameter blocks."""

from __future__ import annotations

import json
import logging
import re

from ..common import (
    _LOOSE_JSON_MISS,
    _balanced_json,
    _clean_rest,
    _loads_json_lenient,
    _normalize_calls,
    _normalize_dsml_text,
    _strip_spans,
)

logger = logging.getLogger("deepseaport.tools")


def _coerce_dsml_value(value: str, string_attr: str | None):
    v = value.strip()
    if string_attr is not None and string_attr.lower() == "true":
        return v
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        try:
            return json.loads(v if v[0] == '"' else f'"{v[1:-1]}"')
        except json.JSONDecodeError:
            return v[1:-1]
    return v


_LT = chr(60)
_GT = chr(62)
_DS = "||DSML||"
_SP = "[ ]*"
# Forgiving bucket (Settings.forgiving_toolcalls). Strict patterns stay byte
# compatible; forgiving patterns accept common model slips in place (no text
# rewrite, so match spans stay valid):
# - 0-2 pipes + inner spaces: <|DSML|>, <DSML invoke>, <|| DSML ||invoke>
#   (fullwidth ｜ is already normalized to ASCII above).
# - HTML-escaped brackets: &lt; &gt; (models escaping "XML").
# - Any whitespace (incl. newlines, NBSP) between marker parts.
_LT_STRICT = re.escape(_LT)
_LT_FORGIVING = r"(?:<|&[lL][tT];)"
_GT_STRICT = re.escape(_GT)
_GT_FORGIVING = r"(?:>|&[gG][tT];)"
_SP_STRICT = _SP
_SP_FORGIVING = r"\s*"
_DS_CORE_STRICT = r"\|\|DSML\|\|"
_DS_CORE_FORGIVING = r"\|{0,2}\s*DSML\s*\|{0,2}"
# Strict attrs stop at the first ">"; forgiving must not stop at an "&gt;"
# inside the tag (HTML-escaped form has no literal ">" until the real end).
_ATTR_STRICT = r"([^>]*)"
_ATTR_FORGIVING = r"((?:(?!&[gG][tT];)[^>])*)"


def _tag_re(name: str, *, closing: bool = False, forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    prefix = lt + ("/" if closing else "")
    return re.compile(
        prefix + sp + "(?:" + ds_core + ")?" + sp
        + name + sp + gt,
        re.IGNORECASE,
    )


def _invoke_open_re(forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    attr = _ATTR_FORGIVING if forgiving else _ATTR_STRICT
    return re.compile(
        lt + sp + "(?:" + ds_core + ")?" + sp
        + "invoke" + attr + gt,
        re.IGNORECASE,
    )


def _param_open_re(forgiving: bool = False) -> re.Pattern:
    lt = _LT_FORGIVING if forgiving else _LT_STRICT
    sp = _SP_FORGIVING if forgiving else _SP_STRICT
    gt = _GT_FORGIVING if forgiving else _GT_STRICT
    ds_core = _DS_CORE_FORGIVING if forgiving else _DS_CORE_STRICT
    attr = _ATTR_FORGIVING if forgiving else _ATTR_STRICT
    return re.compile(
        lt + sp + "(?:" + ds_core + ")?" + sp
        + "parameter" + attr + gt,
        re.IGNORECASE,
    )


_INVOKE_OPEN_RE = _invoke_open_re(False)
_INVOKE_OPEN_FORGIVING_RE = _invoke_open_re(True)
_PARAM_OPEN_RE = _param_open_re(False)
_PARAM_OPEN_FORGIVING_RE = _param_open_re(True)
_INVOKE_CLOSE_RE = _tag_re("invoke", closing=True)
_INVOKE_CLOSE_FORGIVING_RE = _tag_re("invoke", closing=True, forgiving=True)
_PARAM_CLOSE_RE = _tag_re("parameter", closing=True)
_PARAM_CLOSE_FORGIVING_RE = _tag_re("parameter", closing=True, forgiving=True)
_GENERIC_CLOSE_RE = re.compile(re.escape(_LT + "/>"))
_GENERIC_CLOSE_FORGIVING_RE = re.compile(
    _LT_FORGIVING + "/" + _SP_FORGIVING + _GT_FORGIVING)


# Forgiving fix 1+2 (markup side only; parameter *values* stay verbatim so
# intentional file content is never rewritten): smart quotes and fullwidth
# "=" as attribute delimiters, HTML entities inside tags and values.
_SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_HTML_ATTRS = (("&nbsp;", " "), ("&NBSP;", " "), ("&quot;", '"'),
               ("&QUOT;", '"'), ("&#39;", "'"), ("&#x27;", "'"),
               ("&apos;", "'"), ("&APOS;", "'"))
_HTML_VALUES = (("&lt;", "<"), ("&LT;", "<"), ("&gt;", ">"), ("&GT;", ">"),
                ("&quot;", '"'), ("&QUOT;", '"'), ("&#39;", "'"),
                ("&#x27;", "'"), ("&apos;", "'"), ("&APOS;", "'"))


def _preprocess_dsml_attrs(attr_text: str) -> str:
    """Normalize attribute text for forgiving parsing (spans unaffected)."""
    for smart, plain in _SMART_QUOTES.items():
        attr_text = attr_text.replace(smart, plain)
    attr_text = attr_text.replace("\uFF1D", "=")
    for ent, ch in _HTML_ATTRS:
        attr_text = attr_text.replace(ent, ch)
    # "&amp;" last so "&amp;quot;" stays "&quot;" (single-unescape rule).
    return attr_text.replace("&amp;", "&").replace("&AMP;", "&")


def _unescape_dsml_value(value: str) -> str:
    """Unescape HTML entities inside an extracted argument value."""
    for ent, ch in _HTML_VALUES:
        value = value.replace(ent, ch)
    return value.replace("&amp;", "&").replace("&AMP;", "&")


def _tool_param_info(tools: list[dict] | None) -> tuple[dict, dict]:
    """Schema lookup for forgiving inference: {name: {required, props}}.

    Second item maps lowercased names back to canonical ones.
    """
    info: dict[str, dict] = {}
    lowers: dict[str, str] = {}
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if not isinstance(fn, dict):
            continue
        nm = fn.get("name")
        if not isinstance(nm, str) or not nm or nm in info:
            continue
        params = fn.get("parameters", {}) or {}
        if not isinstance(params, dict):
            params = {}
        props = params.get("properties", {}) or {}
        req = params.get("required", []) or []
        info[nm] = {
            "required": [r for r in req if isinstance(r, str)],
            "props": [p for p in props] if isinstance(props, dict) else [],
        }
        lowers.setdefault(nm.lower(), nm)
    return info, lowers


def _assign_unnamed_param(schema: dict | None, args: dict) -> str | None:
    """Target for a nameless <parameter>: single required (else single prop).

    Returns None when ambiguous so nothing is guessed into the wrong slot.
    """
    if not schema:
        return None
    required = [r for r in schema.get("required", []) if r not in args]
    if len(schema.get("required", [])) == 1 and required:
        return required[0]
    props = [p for p in schema.get("props", []) if p not in args]
    if len(schema.get("props", [])) == 1 and props:
        return props[0]
    return None


def _parse_dsml_attrs(attr_text: str, forgiving: bool = False) -> dict[str, str]:
    """Parse ``name="value"`` / ``name='value'`` / ``name=value`` attributes.

    Forgiving also accepts smart quotes (“ ” ‘ ’), fullwidth ＝ and HTML
    entities as delimiters (attribute text only; values stay verbatim).
    """
    if forgiving:
        attr_text = _preprocess_dsml_attrs(attr_text)
    attrs: dict[str, str] = {}
    i = 0
    n = len(attr_text)
    while i < n:
        while i < n and attr_text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (attr_text[j].isalnum() or attr_text[j] in "_-:."):
            j += 1
        name = attr_text[i:j].lower()
        while j < n and attr_text[j].isspace():
            j += 1
        if name and j < n and attr_text[j] == "=":
            j += 1
            while j < n and attr_text[j].isspace():
                j += 1
            if j < n and attr_text[j] in "'\"":
                quote = attr_text[j]
                j += 1
                k = attr_text.find(quote, j)
                if k < 0:
                    k = n
                value = attr_text[j:k]
                j = min(k + 1, n)
            else:
                k = j
                while k < n and not attr_text[k].isspace():
                    k += 1
                value = attr_text[j:k]
                j = k
            attrs[name] = value
        i = j + (0 if j > i else 1)
    return attrs


def _find_close(text: str, start: int, end: int,
                specific: re.Pattern,
                generic: re.Pattern | None = None) -> re.Match | None:
    m = specific.search(text, start, end)
    g = (generic or _GENERIC_CLOSE_RE).search(text, start, end)
    if m and g:
        return m if m.start() <= g.start() else g
    return m or g


def _scan_dsml_calls(text: str, forgiving: bool = False,
                     tools: list[dict] | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Tolerant DSML scanner (canonical, plain tags, generic close tags).

    Forgiving repairs (Settings.forgiving_toolcalls):
    - single/zero-pipe, spaced and HTML-escaped ``<|DSML|>`` markers;
    - missing invoke name, but only when exactly one tool is registered
      (multi-tool stays strict: guessing the tool would be unsafe);
    - nameless ``<parameter>`` assigned to a lone required/single property;
    - JSON-object invoke body with no parameter tags used as arguments.
    Nameless invokes with no parameters and no JSON body are still skipped
    (likely prose mentioning markup, not a call).
    """
    norm = _normalize_dsml_text(text)
    invoke_open = _INVOKE_OPEN_FORGIVING_RE if forgiving else _INVOKE_OPEN_RE
    param_open = _PARAM_OPEN_FORGIVING_RE if forgiving else _PARAM_OPEN_RE
    invoke_close = _INVOKE_CLOSE_FORGIVING_RE if forgiving else _INVOKE_CLOSE_RE
    param_close = _PARAM_CLOSE_FORGIVING_RE if forgiving else _PARAM_CLOSE_RE
    generic_close = _GENERIC_CLOSE_FORGIVING_RE if forgiving else _GENERIC_CLOSE_RE
    info, lowers = _tool_param_info(tools) if forgiving else ({}, {})
    single_tool = next(iter(info)) if len(info) == 1 else None
    opens = list(invoke_open.finditer(norm))
    out: list[dict] = []
    spans: list[tuple[int, int]] = []
    for idx, inv in enumerate(opens):
        body_start = inv.end()
        body_end = opens[idx + 1].start() if idx + 1 < len(opens) else len(norm)
        attrs = _parse_dsml_attrs(inv.group(1) or "", forgiving=forgiving)
        raw_name = attrs.get("name")
        # Fix 4: missing name. Tentative single-tool assumption, confirmed
        # below by parameter/JSON intent signals.
        eff_name = raw_name or single_tool
        if not eff_name:
            continue
        schema = info.get(eff_name)
        if schema is None and isinstance(eff_name, str):
            schema = info.get(lowers.get(eff_name.lower(), eff_name))
        named_close = invoke_close.search(norm, body_start, body_end)
        param_limit = named_close.start() if named_close else body_end
        args: dict = {}
        last_param_end = body_start
        params = list(param_open.finditer(norm, body_start, param_limit))
        for pidx, pm in enumerate(params):
            value_start = pm.end()
            value_limit = (params[pidx + 1].start()
                           if pidx + 1 < len(params) else param_limit)
            pclose = _find_close(norm, value_start, value_limit, param_close,
                                 generic_close)
            value_stop = pclose.start() if pclose else value_limit
            pattrs = _parse_dsml_attrs(pm.group(1) or "", forgiving=forgiving)
            pname = pattrs.get("name")
            raw_value = norm[value_start:value_stop]
            if forgiving:
                raw_value = _unescape_dsml_value(raw_value)
            if pname:
                args[pname] = _coerce_dsml_value(
                    raw_value, pattrs.get("string"))
            elif forgiving:
                # Fix 5: nameless parameter with an unambiguous target.
                target = _assign_unnamed_param(schema, args)
                if target:
                    args[target] = _coerce_dsml_value(
                        raw_value, pattrs.get("string"))
            last_param_end = max(last_param_end,
                                 pclose.end() if pclose else value_stop)
        body_json_found = False
        if forgiving and not params:
            # Fix 6: JSON-object body, no parameter tags (model mixed JSON
            # into a DSML invoke). Dict only; anything else keeps {}.
            body = _unescape_dsml_value(norm[body_start:param_limit]).strip()
            if body:
                parsed = _LOOSE_JSON_MISS
                if body[:1] == "{":
                    obj_text = _balanced_json(body, 0) or ""
                    if obj_text:
                        parsed = _loads_json_lenient(obj_text)
                    else:
                        parsed = _loads_json_lenient(body)
                if parsed is not _LOOSE_JSON_MISS and isinstance(parsed, dict):
                    args = dict(parsed)
                    body_json_found = True
        if not raw_name and not params and not body_json_found:
            continue
        if named_close:
            inv_close = named_close
        else:
            # Generic closers pair in order: each parameter consumes the
            # closer after its value; the next one closes the invoke.
            generic = generic_close.search(norm, last_param_end, body_end)
            inv_close = generic
        out.append({"function": {"name": eff_name, "arguments": args}})
        span_end = inv_close.end() if inv_close else body_end
        spans.append((inv.start(), span_end))
    return out, spans


def _parse_dsml_calls(text: str, forgiving: bool = False,
                      tools: list[dict] | None = None) -> list[dict]:
    """Parse DSML / plain invoke+parameter blocks into raw tool calls."""
    calls, _ = _scan_dsml_calls(text, forgiving=forgiving, tools=tools)
    return calls


def _dsml_spans(text: str, forgiving: bool = False,
                tools: list[dict] | None = None) -> list[tuple[int, int]]:
    """Spans to strip on DSML success (length-preserving normalization)."""
    norm = _normalize_dsml_text(text)
    spans: list[tuple[int, int]] = []
    for tag in ("calls", "tool_calls"):
        for rx in (_tag_re(tag, forgiving=forgiving),
                   _tag_re(tag, closing=True, forgiving=forgiving)):
            for m in rx.finditer(norm):
                spans.append((m.start(), m.end()))
    # The tolerant scanner owns invoke/parameter/plain/generic blocks. Its
    # normalized copy is length-preserving, so positions map to `text`.
    _, invoke_spans = _scan_dsml_calls(text, forgiving=forgiving, tools=tools)
    spans.extend(invoke_spans)
    return spans


def try_parse_dsml(text, valid_or_none, limit_args, forgiving_on, tools):
    """Gate on invoke/DSML marker, then scan. Returns (calls, rest) or (None, None)."""
    low = text.lower()
    if ("dsml" in low or "<invoke" in low
            or (forgiving_on and "&lt;invoke" in low)):
        dsml_raw = _parse_dsml_calls(text, forgiving=forgiving_on, tools=tools)
        if dsml_raw:
            calls, dropped = _normalize_calls(dsml_raw, valid_or_none, limit_args)
            if calls:
                # Strip on normalized text then map back by length-stable replace:
                # fullwidth->ASCII is length-stable (1 char -> 1 char), so spans align.
                spans = _dsml_spans(text, forgiving=forgiving_on, tools=tools)
                rest = _clean_rest(_strip_spans(text, spans), forgiving_on)
                # If spans came from normalized copy with same length, safe.
                return calls, rest
            # DSML present but names unknown: don't fall through to bare scan
            # producing junk; log and keep scanning fenced (explicit user intent).
    return None, None
