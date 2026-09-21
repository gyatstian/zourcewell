"""Parser orchestrator: parse_tool_calls (first match wins)."""

from __future__ import annotations

import logging

from ..common import (
    MAX_ARGS_CHARS,
    MAX_BARE_OBJECTS,
    _looks_like_tool_attempt,
    _recover_loose_openai_call,
    _resolve_forgiving,
    _valid_names,
)
from .bare import try_parse_bare
from .dsml import try_parse_dsml
from .fence import try_parse_fence
from .json_array import try_parse_json_array
from .tags import try_parse_tags

logger = logging.getLogger("deepseaport.tools")


def parse_tool_calls(
    text: str,
    tools: list[dict] | None = None,
    *,
    max_args_chars: int | None = None,
    max_bare_objects: int | None = None,
    forgiving: bool | None = None,
) -> tuple[list[dict] | None, str]:
    """Return (tool_calls or None, remaining_text).

    Accepted formats (first match wins):
    1. {"tool_calls": [...]} — balanced scan, Thought prefix tolerated.
    2. <tool_call>/<function_call>/<invoke> JSON tags.
    3. <||DSML|| invoke/parameter blocks (ASCII + fullwidth variants).
    4. fenced ```json block describing a call.
    5. bare unfenced {"name":..., "arguments":...} (valid-names gated).

    Plain answers return (None, original_text) unchanged.

    Limits are tunable: `max_args_chars` defaults to MAX_ARGS_CHARS
    (env DEEPSEAPORT_MAX_ARGS_CHARS), `max_bare_objects` defaults to
    MAX_BARE_OBJECTS (env DEEPSEAPORT_MAX_BARE_OBJECTS).

    Forgiving bucket (`forgiving=True`, env DEEPSEAPORT_FORGIVING_TOOLCALLS,
    or Settings.forgiving_toolcalls): extra repairs for model slips.
    Strict (default) keeps historical behaviour. Forgiving repairs:
    1. single/zero-pipe, spaced and HTML-escaped DSML markers;
    2. smart quotes / fullwidth ＝ / entities as attribute delimiters
       (parameter values stay verbatim);
    3. missing invoke name when exactly one tool is registered;
    4. nameless parameter assigned to a lone required/single property;
    5. JSON-object invoke body with no parameter tags used as arguments.
    Markup only: values are never invented, truncated calls never executed.
    """
    forgiving_on = _resolve_forgiving(forgiving)
    valid = _valid_names(tools) if tools else set()
    valid_or_none = valid if tools else None
    limit_args = max_args_chars if max_args_chars is not None else MAX_ARGS_CHARS
    limit_bare = max_bare_objects if max_bare_objects is not None else MAX_BARE_OBJECTS

    # Format 1: {"tool_calls": [...]}
    calls, rest = try_parse_json_array(text, valid_or_none, limit_args, forgiving_on)
    if calls:
        return calls, rest

    # Format 2: <tool_call>/<function_call>/<invoke> tags.
    calls, rest = try_parse_tags(text, valid_or_none, limit_args, forgiving_on)
    if calls:
        return calls, rest

    # Format 3: DSML / plain invoke+parameter blocks.
    calls, rest = try_parse_dsml(text, valid_or_none, limit_args, forgiving_on, tools)
    if calls:
        return calls, rest

    # Format 4: fenced ```json block describing a call.
    calls, rest = try_parse_fence(text, valid_or_none, limit_args, forgiving_on)
    if calls:
        return calls, rest

    # Format 5: bare unfenced name/arguments object.
    calls, rest = try_parse_bare(text, valid, valid_or_none, limit_args, limit_bare, forgiving_on)
    if calls:
        return calls, rest

    # Last resort: recover an unescaped OpenAI-style JSON attempt, e.g.
    # "arguments": "{"path": ...}" where the inner object was not escaped.
    recovered = _recover_loose_openai_call(text, valid_or_none, limit_args)
    if recovered:
        logger.info("recovered loose OpenAI tool call (%d call(s))", len(recovered))
        return recovered, ""

    if tools and _looks_like_tool_attempt(text, valid):
        logger.debug("tool parse miss preview=%.300s", text[:300])
    return None, text
