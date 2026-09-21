"""Shared base for tool parsing: env, limits, lenient JSON, normalize."""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("deepseaport.tools")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    return val if val > 0 else default


MAX_ARGS_CHARS = _int_env("DEEPSEAPORT_MAX_ARGS_CHARS", 200_000)
MAX_BARE_OBJECTS = _int_env("DEEPSEAPORT_MAX_BARE_OBJECTS", 40)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_forgiving(explicit: bool | None) -> bool:
    """One toggle for all lenient tool-call repairs.

    Explicit per-call value wins; otherwise env
    DEEPSEAPORT_FORGIVING_TOOLCALLS (or Settings.forgiving_toolcalls plumbed
    through ``parse_tool_calls(forgiving=...)``). Strict by default so
    historical behaviour is unchanged.
    """
    if explicit is not None:
        return bool(explicit)
    return _bool_env("DEEPSEAPORT_FORGIVING_TOOLCALLS", False)


def _valid_names(tools: list[dict] | None) -> set[str]:
    valid: set[str] = set()
    for t in tools or []:
        if isinstance(t, dict):
            name = t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else None
            if name:
                valid.add(name)
    return valid


def _balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced {...} object starting at index `start`."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


_LOOSE_JSON_MISS = object()


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters that models commonly put inside JSON strings.

    ``json.loads`` rejects a literal newline/tab inside a quoted value. Models
    frequently emit file content and shell commands that way, which is one of
    the main reasons a structurally valid tool call turns into "no tool call
    parsed". This repair keeps already-escaped sequences untouched.
    """
    out: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append("\\u%04x" % ord(ch))
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    """Drop trailing commas before ``}``/``]`` outside string literals."""
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # skip comma, whitespace is handled on the next loop
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _quote_unquoted_keys(text: str) -> str:
    """Quote bare object keys (``{foo: 1}``) outside strings."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch in "{,":
            out.append(ch)
            i += 1
            while i < n and text[i].isspace():
                out.append(text[i])
                i += 1
            if i < n and (text[i].isalpha() or text[i] == "_"):
                j = i
                while j < n and (text[j].isalnum() or text[j] in "_.-"):
                    j += 1
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k < n and text[k] == ":":
                    out.append('"' + text[i:j] + '"')
                    i = j
                    continue
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _replace_python_literals(text: str) -> str:
    """Replace ``True``/``False``/``None`` outside strings with JSON values."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            repl = {"True": "true", "False": "false", "None": "null"}.get(word, word)
            out.append(repl)
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _loads_json_lenient(text: str):
    """Parse JSON with small, safe repairs for common model-output damage.

    Returns ``_LOOSE_JSON_MISS`` when nothing usable could be produced. Only
    used after strict ``json.loads`` failed, so valid JSON is never changed.
    """
    if not isinstance(text, str):
        return _LOOSE_JSON_MISS
    candidate = text.strip()
    if not candidate:
        return _LOOSE_JSON_MISS
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    repaired = _replace_python_literals(
        _quote_unquoted_keys(
            _remove_trailing_commas(
                _escape_control_chars_in_strings(candidate))))
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Python-style dicts (single quotes / True / None) still occur when a
    # model is asked for an "arguments" object.
    try:
        import ast
        value = ast.literal_eval(candidate)
        if isinstance(value, (dict, list)):
            return value
    except Exception:
        pass
    return _LOOSE_JSON_MISS


class _OversizeArgs(ValueError):
    """Internal signal: arguments exceed the configured limit; caller drops."""


def _normalize_args(args, call_idx: int, max_args_chars: int | None = None) -> str:
    """Normalize arguments to a JSON string.

    Oversize payloads are rejected (raise _OversizeArgs) instead of being
    truncated: slicing JSON mid-object then wrapping the fragment in
    {"_raw": ...} silently corrupts data and produces invalid arguments.
    """
    limit = max_args_chars if max_args_chars is not None else MAX_ARGS_CHARS
    if isinstance(args, dict):
        dumped = json.dumps(args, ensure_ascii=False)
        if len(dumped) > limit:
            logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(dumped), limit)
            raise _OversizeArgs(f"arguments exceed {limit} chars")
        return dumped
    if isinstance(args, str):
        candidate = args.strip()
        if len(candidate) > limit:
            logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(candidate), limit)
            raise _OversizeArgs(f"arguments exceed {limit} chars")
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            repaired = _loads_json_lenient(candidate)
            if repaired is not _LOOSE_JSON_MISS:
                dumped = json.dumps(repaired, ensure_ascii=False)
                if len(dumped) > limit:
                    logger.warning("rejecting oversize tool arguments (%d chars > %d)",
                                   len(dumped), limit)
                    raise _OversizeArgs(f"arguments exceed {limit} chars")
                logger.info("tool call %d arguments repaired by lenient JSON parser", call_idx)
                return dumped
            logger.warning("tool call %d has non-JSON arguments; wrapping", call_idx)
            return json.dumps({"_raw": candidate}, ensure_ascii=False)
    dumped = json.dumps(args, ensure_ascii=False)
    if len(dumped) > limit:
        logger.warning("rejecting oversize tool arguments (%d chars > %d)", len(dumped), limit)
        raise _OversizeArgs(f"arguments exceed {limit} chars")
    return dumped


def _normalize_calls(
    raw_calls: list,
    valid_names: set[str] | None,
    max_args_chars: int | None = None,
) -> tuple[list[dict], list[int]]:
    """Validate + normalize; returns (calls, dropped_indexes).

    Exact name match wins; case-insensitive fallback maps to the canonical
    name so harnesses don't reject `Read` vs `read`. Unknown names and
    oversize arguments are dropped.
    """
    lower_map: dict[str, str] = {}
    if valid_names:
        for n in valid_names:
            if isinstance(n, str) and n.lower() not in lower_map:
                lower_map[n.lower()] = n
    calls: list[dict] = []
    dropped: list[int] = []
    for i, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            dropped.append(i)
            continue
        fn = call.get("function", call if "name" in call else {})
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            dropped.append(i)
            continue
        canonical = name
        if valid_names:
            if name in valid_names:
                canonical = name
            elif name.lower() in lower_map:
                canonical = lower_map[name.lower()]
            else:
                dropped.append(i)
                continue
        args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
        try:
            norm_args = _normalize_args(args, i, max_args_chars)
        except _OversizeArgs:
            dropped.append(i)
            continue
        calls.append({
            "id": call.get("id", f"call_{len(calls) + 1:03d}") if isinstance(call.get("id"), str) else f"call_{len(calls) + 1:03d}",
            "type": "function",
            "function": {"name": canonical, "arguments": norm_args},
        })
    return calls, dropped


_THOUGHT_LINE_RE = re.compile(r"^\s*(thought|thinking|reasoning)\s*:.*$", re.IGNORECASE)


def _clean_rest(rest: str, forgiving: bool = False) -> str:
    """Strip Thought preamble + leftover tool markers when a call was found.

    Only called on the success path; plain answers return untouched.
    Forgiving also strips single/zero-pipe, spaced and HTML-escaped
    ``<|DSML|>`` leftovers.
    """
    cleaned = rest.strip()
    # Drop leading Thought:/Thinking: preamble lines (model narration, not answer).
    lines = cleaned.splitlines()
    while lines and (_THOUGHT_LINE_RE.match(lines[0]) or lines[0].strip() == ""):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    # Remove stray DSML marker lines left after extraction.
    if forgiving:
        cleaned = re.sub(r"(?:<|&[lL][tT];)/?\|{0,2}\s*DSML\s*\|{0,2}[^>]*?(?:>|&[gG][tT];)",
                         "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"(?:<|&[lL][tT];)/?\|{1,2}[^>]*?(?:>|&[gG][tT];)",
                         "", cleaned).strip()
    else:
        cleaned = re.sub(r"<\/?\|\|DSML\|\|[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"<\/?\|\|[^>]*>", "", cleaned).strip()
    return cleaned


def _normalize_dsml_text(text: str) -> str:
    # Models emit fullwidth variants (｜＜＞); normalize to ASCII for parsing.
    return (text.replace("\uFF5C", "|").replace("\uFF1C", "<").replace("\uFF1E", ">"))


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    parts: list[str] = []
    last = 0
    for s, e in spans:
        if s < last:  # overlapping bare objects; skip inner
            continue
        parts.append(text[last:s])
        last = max(last, e)
    parts.append(text[last:])
    return "".join(parts)


def _looks_like_tool_attempt(text: str, valid: set[str] | None) -> bool:
    low = text.lower()
    if '"tool_calls"' in low or "<tool_call" in low or "<function_call" in low or "dsml" in low or "<invoke" in low:
        return True
    if valid:
        for n in valid:
            if isinstance(n, str) and n:
                # Word match: substring "read" must not fire on "already"/"bread".
                if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.IGNORECASE):
                    return True
    return False


def looks_truncated_tool_attempt(text: str, tools: list[dict] | None = None,
                                 forgiving: bool | None = None) -> bool:
    """Best-effort signal that a tool attempt was cut off mid-output.

    DeepSeek web has no finish_reason, so a model reply truncated by the
    upstream token limit arrives exactly like a normal stop. If the reply
    clearly looks like a tool call and JSON/DSML delimiters are unbalanced,
    callers can report ``finish_reason: "length"`` so harnesses retry or
    split the request instead of treating raw markup as the final answer.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    low = text.lower()
    strong = ('"tool_calls"' in low or "<tool_calls" in low or "<tool_call"
              in low or "dsml" in low or "<invoke" in low
              or "<function_call" in low)
    if not strong:
        return False
    stack: list[str] = []
    in_str = False
    esc = False
    closes = {"{": "}", "[": "]", "(": ")"}
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ord(ch) == 92:  # backslash
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in closes:
            stack.append(closes[ch])
        elif ch in "}])":
            if not stack or stack.pop() != ch:
                return True
    if stack or in_str:
        return True
    # DSML/XML-style blocks can be truncated without unbalanced braces.
    if "dsml" in low or "<invoke" in low:
        forgiving_on = _resolve_forgiving(forgiving)
        # Normalize fullwidth first so ｜ variants count the same as ASCII.
        norm_low = _normalize_dsml_text(text).lower()
        lt = chr(60)
        generic_close = norm_low.count(lt + "/" + chr(62))
        if forgiving_on:
            for tag in ("tool_calls", "calls", "invoke", "parameter"):
                opened = len(re.findall(
                    r"(?:<|&[lL][tT];)\|{0,2}\s*dsml\s*\|{0,2}" + tag, norm_low))
                closed = len(re.findall(
                    r"(?:<|&[lL][tT];)/\|{0,2}\s*dsml\s*\|{0,2}" + tag, norm_low))
                if opened > closed + generic_close:
                    return True
        else:
            ds = "||dsml||"
            for tag in ("tool_calls", "calls", "invoke", "parameter"):
                opened = norm_low.count(lt + ds + tag)
                closed = norm_low.count(lt + "/" + ds + tag)
                if opened > closed + generic_close:
                    return True
    return False


_FENCED_BLOCK_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)


def _recover_loose_openai_call(text: str, valid: set[str] | None,
                                limit_args: int):
    """Last-resort recovery for an unescaped OpenAI tool-call JSON attempt.

    Models often write ``"arguments": "{"path": ..., "content": ...}"`` (the
    inner JSON was not escaped).  Strict parsing rejects it, but the intended
    payload is usually recoverable by taking the outer quote before the final
    call delimiters and parsing the inner object leniently.  Only attempted
    after every normal format failed, so valid replies are unchanged.
    """
    if not valid or text.count('"arguments"') != 1:
        return None
    low = text.lower()
    if '"tool_calls"' not in low and '"function"' not in low:
        return None
    m = re.search(r'"arguments"[ ]*:[ ]*', text)
    if m is None:
        return None
    value_start = m.end()
    if value_start >= len(text):
        return None
    if text[value_start] == '"':
        end = text.rfind('"')
        if end <= value_start:
            return None
        raw = text[value_start + 1:end]
    elif text[value_start] in "{[":
        raw = _balanced_json(text, value_start)
        if not raw:
            return None
    else:
        return None
    parsed = _loads_json_lenient(raw)
    if parsed is _LOOSE_JSON_MISS and isinstance(raw, str):
        parsed = _loads_json_lenient(raw.strip())
    if parsed is _LOOSE_JSON_MISS:
        return None
    if isinstance(parsed, str):
        parsed = _loads_json_lenient(parsed)
    if parsed is _LOOSE_JSON_MISS or not isinstance(parsed, (dict, list)):
        return None
    lower_map = {n.lower(): n for n in valid if isinstance(n, str)}
    name = None
    for candidate in reversed(re.findall(r'"name"[ ]*:[ ]*"([^"]+)"',
                                         text[:value_start])):
        if candidate in valid:
            name = candidate
            break
        mapped = lower_map.get(candidate.lower())
        if mapped:
            name = mapped
            break
    if not name:
        return None
    calls, _ = _normalize_calls([{"function": {"name": name,
                                               "arguments": parsed}}],
                                valid, limit_args)
    return calls or None
