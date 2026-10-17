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


# Default argument-size limit, used only when a caller passes no
# `max_args_chars`. LIFETIME: a SNAPSHOT read once at first import and never
# refreshed, so `Settings.tool_args_max_chars` and this constant CAN
# disagree — documented, not a bug: the real request path does not disagree,
# because `server_routes._prep_tool_limits()` reads the Setting per request
# and threads it through the prep dict to `parse_tool_calls(...,
# max_args_chars=...)`. The snapshot is only the DEFAULT for callers that pass
# no limit (`tool_system_prompt()`, direct `parse_tool_calls()` from a script
# or a test), which keep whatever the environment held at first import.
# Import this name (re-exported by `deepseaport.core.tools`); it will never
# track the environment — to honour a live value, read it from `Settings` and
# pass `max_args_chars` explicitly.
MAX_ARGS_CHARS = _int_env("DEEPSEAPORT_MAX_ARGS_CHARS", 200_000)
MAX_BARE_OBJECTS = _int_env("DEEPSEAPORT_MAX_BARE_OBJECTS", 40)
# Universal bound for parser scans (json_array/fence/tags/DSML) and total
# normalized calls. Reuses MAX_BARE_OBJECTS so one env knob bounds work.
MAX_SCAN_OBJECTS = MAX_BARE_OBJECTS
MAX_TOTAL_CALLS = MAX_BARE_OBJECTS
MAX_PARAMS_PER_INVOKE = 64
# Lenient-repair budget: skip Python-loop repairs on huge candidates that
# would be dropped as oversize anyway (strict path still attempted).
# Derived from the import-time snapshot (see MAX_ARGS_CHARS): a repair budget
# is a work heuristic, not the accept/reject limit, so re-reading it per call
# would only add env lookups to the hot path. The limit itself is applied in
# _normalize_args, which uses the live value whenever a caller passes one.
_LENIENT_REPAIR_BUDGET = MAX_ARGS_CHARS + 16_384

# C-level fast guards (avoid Python-loop repair passes that would be no-ops).
_TRAILING_COMMA_RE = re.compile(r",\s*[}\]]")
_UNQUOTED_KEY_HINT_RE = re.compile(r"[{,]\s*[A-Za-z_]")
_OTHER_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CLEAN_DSML_STRICT_1 = re.compile(r"<\/?\|\|DSML\|\|[^>]*>", re.IGNORECASE)
_CLEAN_DSML_STRICT_2 = re.compile(r"<\/?\|\|[^>]*>", re.IGNORECASE)
_CLEAN_DSML_FORGIVING_1 = re.compile(
    r"(?:<|&[lL][tT];)/?\|{0,2}\s*DSML\s*\|{0,2}[^>]*?(?:>|&[gG][tT];)",
    re.IGNORECASE)
_CLEAN_DSML_FORGIVING_2 = re.compile(
    r"(?:<|&[lL][tT];)/?\|{1,2}[^>]*?(?:>|&[gG][tT];)", re.IGNORECASE)

# Forgiving tag-open/tag-close pairs for the truncation heuristic. Precompiled
# like its neighbours above; ``re`` would cache them anyway, but keeping them
# here means one place owns every pattern in this module.
_DSML_TAG_COUNTS_RE = tuple(
    (re.compile(r"(?:<|&[lL][tT];)\|{0,2}\s*dsml\s*\|{0,2}" + tag),
     re.compile(r"(?:<|&[lL][tT];)/\|{0,2}\s*dsml\s*\|{0,2}" + tag))
    for tag in ("tool_calls", "calls", "invoke", "parameter"))


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


def _try_lenient_repairs(candidate: str):
    """Repair path assuming strict ``json.loads`` already failed.

    Split out so ``_normalize_args`` avoids a duplicate strict parse.
    Fast C-level guards skip Python-loop passes that would be no-ops;
    huge candidates skip repairs (would be dropped as oversize anyway).
    """
    if len(candidate) > _LENIENT_REPAIR_BUDGET:
        return _LOOSE_JSON_MISS
    text = candidate
    # Escape control chars only when raw controls exist.
    if ("\n" in text or "\r" in text or "\t" in text
            or _OTHER_CONTROL_RE.search(text) is not None):
        text = _escape_control_chars_in_strings(text)
    # Trailing commas only when ",}" / ",]" pattern exists.
    if "," in text and _TRAILING_COMMA_RE.search(text) is not None:
        text = _remove_trailing_commas(text)
    # Bare keys only when "{word" / ",word" hint exists.
    if _UNQUOTED_KEY_HINT_RE.search(text) is not None:
        text = _quote_unquoted_keys(text)
    # Python literals only when those words appear.
    if ("True" in text or "False" in text or "None" in text):
        text = _replace_python_literals(text)
    if text is not candidate:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    else:
        # No repair applied and strict already failed: second loads would
        # fail identically, skip it and try literal_eval below when plausible.
        pass
    # Python-style dicts (single quotes / True / None) still occur when a
    # model is asked for an "arguments" object.
    if ("'" in candidate or "True" in candidate or "False" in candidate
            or "None" in candidate):
        if len(candidate) > _LENIENT_REPAIR_BUDGET:
            return _LOOSE_JSON_MISS
        try:
            import ast
            value = ast.literal_eval(candidate)
            if isinstance(value, (dict, list)):
                return value
        except Exception:
            pass
    return _LOOSE_JSON_MISS


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
    return _try_lenient_repairs(candidate)


class _OversizeArgs(ValueError):
    """Internal signal: arguments exceed the configured limit; caller drops."""


def _normalize_args(args, call_idx: int, max_args_chars: int | None = None) -> str:
    """Normalize arguments to a JSON string.

    Oversize payloads are rejected (raise _OversizeArgs) instead of being
    truncated: slicing JSON mid-object then wrapping the fragment in
    {"_raw": ...} silently corrupts data and produces invalid arguments.

    ``max_args_chars`` is the live per-call limit (the request path passes
    ``Settings.tool_args_max_chars``); ``None`` falls back to the
    import-time ``MAX_ARGS_CHARS`` snapshot, which does NOT track the
    environment afterwards.
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
        except (json.JSONDecodeError, ValueError):
            # Strict already failed: repair directly without a second strict
            # attempt inside _loads_json_lenient.
            repaired = _try_lenient_repairs(candidate)
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
    oversize arguments are dropped. Total calls are capped at
    MAX_TOTAL_CALLS so per-argument limits can't combine into unbounded
    memory (first N win, rest dropped with indexes preserved).

    ``valid_names`` is tri-state, and the difference is the whole
    hallucination guard: ``None`` means the request declared NO tools (a
    plain chat — there is no allowlist, so nothing is name-checked), while a
    possibly-empty set means tools WERE declared. ``parse_tool_calls_ex``
    (core/tools/parser/__init__.py:132-133) is what keeps the two apart: it
    passes ``None`` only when ``tools`` is falsy, and the set otherwise — so
    an empty set can only mean "declared, but no name could be extracted"
    (``_valid_names`` returns set() for a malformed or Anthropic-shaped
    array). An unverifiable name is therefore treated as unknown and dropped
    rather than forwarded: a missing call costs the agent one retry, a wrong
    name costs it a dispatch of a tool that does not exist.
    """
    if len(raw_calls) > MAX_TOTAL_CALLS:
        # Bound total work: callers see first N, extras reported as dropped.
        # Defer logging to per-call path; keep indexes stable.
        truncated_extra = list(range(MAX_TOTAL_CALLS, len(raw_calls)))
        raw_calls = raw_calls[:MAX_TOTAL_CALLS]
    else:
        truncated_extra: list[int] = []
    lower_map: dict[str, str] = {}
    if valid_names:
        for n in valid_names:
            if isinstance(n, str) and n.lower() not in lower_map:
                lower_map[n.lower()] = n
    calls: list[dict] = []
    dropped: list[int] = []
    # Ids already handed out in THIS response. Model-supplied ids are
    # arbitrary strings and routinely repeat (the JSON example injected in
    # tool_system_prompt (core/tools/prompt.py:91) always shows "call_001",
    # so a model echoing that shape for two invokes emits it twice). A
    # duplicate id is worse than a renamed one: harnesses map role:"tool"
    # results back to their call by id, so a collision makes the second
    # result unmatchable. Renaming is deterministic (no randomness, derived
    # from position), so the same call keeps the same id in the stream and
    # non-stream paths.
    used_ids: set[str] = set()
    for i, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            logger.debug("dropping tool call %d: not an object", i)
            dropped.append(i)
            continue
        fn = call.get("function", call if "name" in call else {})
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            logger.debug("dropping tool call %d: missing name", i)
            dropped.append(i)
            continue
        canonical = name
        # `is not None`, not truthiness: an empty set means tools were
        # declared but no usable name could be read out of them, so nothing
        # is checkable and every name is unverifiable.
        if valid_names is not None:
            if name in valid_names:
                canonical = name
            elif name.lower() in lower_map:
                canonical = lower_map[name.lower()]
            elif not valid_names:
                # Declared-but-unusable tool schemas (malformed entries, or
                # Anthropic-shaped [{"name":..,"input_schema":{..}}] that has
                # no OpenAI `function` wrapper). Suppress instead of
                # forwarding: the agent would dispatch a tool that does not
                # exist. Warning (not debug) so the malformed `tools` array
                # is visible in the log, as the unknown-name case is.
                logger.warning("dropping tool call %d: tools were declared but "
                               "no usable tool name could be read from them "
                               "(malformed tool schemas); cannot verify %r",
                               i, name)
                dropped.append(i)
                continue
            else:
                # Unknown tool names are dropped so the harness never
                # executes a hallucinated call — warning (not debug) so a
                # miss surfaces as unknown-name vs oversize vs format drift.
                logger.warning("dropping tool call %d: unknown tool %r "
                               "(valid=%s)", i, name, sorted(valid_names))
                dropped.append(i)
                continue
        args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
        try:
            norm_args = _normalize_args(args, i, max_args_chars)
        except _OversizeArgs:
            dropped.append(i)
            continue
        raw_id = call.get("id")
        if isinstance(raw_id, str) and raw_id not in used_ids:
            call_id = raw_id
        else:
            # Mint `call_NNN` that collides with nothing already emitted: the
            # positional guess can itself duplicate a model-supplied id.
            n = len(calls) + 1
            call_id = f"call_{n:03d}"
            while call_id in used_ids:
                n += 1
                call_id = f"call_{n:03d}"
        used_ids.add(call_id)
        calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": canonical, "arguments": norm_args},
        })
    if truncated_extra:
        dropped.extend(truncated_extra)
        dropped.sort()
    return calls, dropped


_THOUGHT_LINE_RE = re.compile(r"^\s*(thought|thinking|reasoning)\s*:.*$", re.IGNORECASE)


def _finish(raw_calls, valid_or_none, limit_args, fmt, forgiving_on, *,
            diagnostics=None, on_hit=None):
    """Normalize, record the format trail, strip the matched span.

    The tail every parser format shares: ``_normalize_calls`` drops unknown
    names and oversize args, the diagnostics entry records
    kept/dropped/raw, and a non-empty result returns ``(calls, rest)``. An
    empty result returns ``(None, None)`` so the first-match-wins chain falls
    through to the next format.

    ``on_hit`` computes the text to strip and is called only when calls
    survive, so a format that matched nothing pays nothing.
    """
    calls, dropped = _normalize_calls(raw_calls, valid_or_none, limit_args)
    if diagnostics is not None:
        diagnostics.append({"format": fmt, "matched": bool(calls),
                            "kept": len(calls), "dropped": list(dropped),
                            "raw": len(raw_calls)})
    if calls:
        return calls, _clean_rest(on_hit(), forgiving_on)
    return None, None


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
    # Remove stray DSML marker lines left after extraction. Guard with
    # substring checks so plain answers skip regex work; patterns are
    # precompiled (no per-call re.compile).
    low = cleaned.lower()
    if "<" not in cleaned and "&lt;" not in low and "dsml" not in low:
        return cleaned
    if forgiving:
        cleaned = _CLEAN_DSML_FORGIVING_1.sub("", cleaned).strip()
        # Second pass only when pipe markers may remain.
        if "|" in cleaned or "&lt;" in cleaned.lower():
            cleaned = _CLEAN_DSML_FORGIVING_2.sub("", cleaned).strip()
    else:
        cleaned = _CLEAN_DSML_STRICT_1.sub("", cleaned).strip()
        if "|" in cleaned:
            cleaned = _CLEAN_DSML_STRICT_2.sub("", cleaned).strip()
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


def _contains_word(low_text: str, low_name: str) -> bool:
    """Case-folded word match. Same semantics as ``(?<!\\w)name(?!\\w)``.

    ``re`` caches compiled patterns (512 entries), so building the pattern
    per call costs a dict lookup, not a recompile.
    """
    if not low_name:
        return False
    return re.search(rf"(?<!\w){re.escape(low_name)}(?!\w)", low_text) is not None


def _looks_like_tool_attempt(text: str, valid: set[str] | None) -> bool:
    low = text.lower()
    if '"tool_calls"' in low or "<tool_call" in low or "<function_call" in low or "dsml" in low or "<invoke" in low:
        return True
    if valid:
        for n in valid:
            if isinstance(n, str) and n:
                # Word match: substring "read" must not fire on "already"/"bread".
                if _contains_word(low, n.lower()):
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
            for _open_re, _close_re in _DSML_TAG_COUNTS_RE:
                opened = len(_open_re.findall(norm_low))
                closed = len(_close_re.findall(norm_low))
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
    elif text[value_start] == "{":
        # Object only: _balanced_json balances braces, so an array-shaped
        # arguments value (starting "[") is not recoverable.
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
